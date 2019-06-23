#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <iostream>
#include <stdio.h>
#include <stdarg.h>
using namespace std;
using namespace cv;

////���еĴ���
//int main( int argc, char** argv )
//{
//	IplImage* image=cvLoadImage("fruits.jpg");
//	IplImage* rotateimg = cvCloneImage( image );
//
//	int  delta = 1;
//	int  angle = 0;
//	int opt = 0;   // 1�� ��ת�����ţ�0:  ������ת
//	double factor;
//	cvNamedWindow( "Source", 1 );
//	cvShowImage( "Source", image );
//
//	for(;;)
//	{
//		//��ת����map
//		// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
//		// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
//		float map[6];
//		
//		CvMat M = cvMat( 2, 3, CV_32F, map );
//		int width = image->width;
//		int height = image->height;
//		if(opt) // ��ת������
//			factor = (cos(angle*CV_PI/180.) + 1.05)*2;
//		else //  ������ת
//			factor = 1;
//		map[0] = (float)(factor*cos(-angle*2*CV_PI/180.));
//		map[1] = (float)(factor*sin(-angle*2*CV_PI/180.));
//		map[3] = -map[1];
//		map[4] = map[0];
//		// ����ת��������ͼ���м�
//		map[2] = width*0.5f;  
//		map[5] = height*0.5f;  
//		//  dst(x,y) = A * src(x,y) + b
//		cvGetQuadrangleSubPix( image, rotateimg, &M);
//		cvNamedWindow( "Rotate", 1 );
//		cvShowImage( "Rotate", rotateimg );
//		if( cvWaitKey(5) == 27 )
//			break;
//		angle =(int) (angle + delta) % 360;
//	} 
//
//	
//	return 0;
//}

//��תͼ�����ݲ��䣬�ߴ���Ӧ���
IplImage* rotateImage1(IplImage* img,int degree){
	double angle = degree  * CV_PI / 180.; // ����  
	double a = sin(angle), b = cos(angle); 
	int width = img->width;  
	int height = img->height;  
	int width_rotate= int(height * fabs(a) + width * fabs(b));  
	int height_rotate=int(width * fabs(a) + height * fabs(b));  
	//��ת����map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float map[6];
	CvMat map_matrix = cvMat(2, 3, CV_32F, map);  
	// ��ת����
	CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);  
	cv2DRotationMatrix(center, degree, 1.0, &map_matrix);  
	map[2] += (width_rotate - width) / 2;  
	map[5] += (height_rotate - height) / 2;  
	IplImage* img_rotate = cvCreateImage(cvSize(width_rotate, height_rotate), 8, 3); 
	//��ͼ��������任
	//CV_WARP_FILL_OUTLIERS - ����������ͼ������ء�
	//�������������������ͼ��ı߽��⣬��ô���ǵ�ֵ�趨Ϊ fillval.
	//CV_WARP_INVERSE_MAP - ָ�� map_matrix �����ͼ������ͼ��ķ��任��
	cvWarpAffine( img,img_rotate, &map_matrix, CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS, cvScalarAll(0));  
	return img_rotate;
}

//��תͼ�����ݲ��䣬�ߴ���Ӧ���
IplImage* rotateImage2(IplImage* img, int degree)  
{  
	double angle = degree  * CV_PI / 180.; 
	double a = sin(angle), b = cos(angle); 
	int width=img->width, height=img->height;
	//��ת�����ͼ�ߴ�
	int width_rotate= int(height * fabs(a) + width * fabs(b));  
	int height_rotate=int(width * fabs(a) + height * fabs(b));  
	IplImage* img_rotate = cvCreateImage(cvSize(width_rotate, height_rotate), img->depth, img->nChannels);  
	cvZero(img_rotate);  
	//��֤ԭͼ��������Ƕ���ת����С�ߴ�
	int tempLength = sqrt((double)width * width + (double)height *height) + 10;  
	int tempX = (tempLength + 1) / 2 - width / 2;  
	int tempY = (tempLength + 1) / 2 - height / 2;  
	IplImage* temp = cvCreateImage(cvSize(tempLength, tempLength), img->depth, img->nChannels);  
	cvZero(temp);  
	//��ԭͼ���Ƶ���ʱͼ��tmp����
	cvSetImageROI(temp, cvRect(tempX, tempY, width, height));  
	cvCopy(img, temp, NULL);  
	cvResetImageROI(temp);  
	//��ת����map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float m[6];  
	int w = temp->width;  
	int h = temp->height;  
	m[0] = b;  
	m[1] = a;  
	m[3] = -m[1];  
	m[4] = m[0];  
	// ����ת��������ͼ���м�  
	m[2] = w * 0.5f;  
	m[5] = h * 0.5f;  
	CvMat M = cvMat(2, 3, CV_32F, m);  
	cvGetQuadrangleSubPix(temp, img_rotate, &M);  
	cvReleaseImage(&temp);  
	return img_rotate;
}  

//��ʱ����תͼ��degree�Ƕȣ�ԭ�ߴ磩
void rotateImage(IplImage* img, IplImage *img_rotate,int degree)
{
	//��ת����Ϊͼ������
	CvPoint2D32f center;  
	center.x=float (img->width/2.0+0.5);
	center.y=float (img->height/2.0+0.5);
	//�����ά��ת�ķ���任����
	float m[6];            
	CvMat M = cvMat( 2, 3, CV_32F, m );
	cv2DRotationMatrix( center, degree,1, &M);
	//�任ͼ�񣬲��ú�ɫ�������ֵ
	cvWarpAffine(img,img_rotate, &M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) );
}

void rotateAndZoom(IplImage* img, IplImage *img_rotate,int degree)
{
		int width = img->width;
		int height = img->height;
		double angle = degree  * CV_PI / 180.; 
		double a = sin(angle), b = cos(angle); 

		double factor=(b+ 1.05)*2;
		float map[6];
		CvMat M = cvMat( 2, 3, CV_32F, map );
		
		map[0] = (float)(factor*b);
		map[1] = (float)(factor*a);
		map[3] = -map[1];
		map[4] = map[0];
		// ����ת��������ͼ���м�
		map[2] = width*0.5f;  
		map[5] = height*0.5f;  
		cvGetQuadrangleSubPix( img, img_rotate, &M);

	//	cvWarpAffine(img,img_rotate, &M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) );
}

void rotateAndZoom2(IplImage* img, IplImage *img_rotate,int degree)
{
	double angle = degree  * CV_PI / 180.; // ����  
	double a = sin(angle), b = cos(angle); 
	int width=img->width;
	int height=img->height;

	float map[6];            
	CvMat M = cvMat( 2, 3, CV_32F, map );
	CvPoint2D32f center;
	center.x=float (img->width/2.0+0.5);
	center.y=float (img->height/2.0+0.5);
	cv2DRotationMatrix( center, degree,1, &M);
	
	double factor=0.5;
	map[0]=map[0]*factor;
	map[1]=map[1]*factor;
	map[3]=map[3]*factor;
	map[4]=map[4]*factor;

	//map[2] =(float)(height * fabs(a) + width * fabs(b)-img->width*factor/2);  
	//map[5] = (float)(width * fabs(a) + height * fabs(b)- img->height*factor/2);  

	cvWarpAffine(img,img_rotate, &M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) );
}


//int main()  
//{  
//	IplImage *img = cvLoadImage("baboon.jpg");  
//	cvNamedWindow ("Source", 1);  
//	cvShowImage ("Source", img);  
//
//	//IplImage *img_rotate=cvCloneImage(img);
//	//rotateImage(img,img_rotate,30);
//	//cvNamedWindow( "Rotate", 1 );  
//	//cvShowImage( "Rotate", img_rotate);
//
//	//for(int degree=0;degree<361;degree++){
//	//	//rotateAndZoom(img,img_rotate,degree);
//	//	rotateImage(img,img_rotate,degree);
//
//	//	cvShowImage( "Rotate", img_rotate); 
//	//	if( cvWaitKey(5) == 27 )
//	//			break;
//	//}
//
//	IplImage *img_rotate=rotateImage1(img,30);
//	cvShowImage( "Rotate", img_rotate); 
//
//	cvWaitKey(0);  
//	cvReleaseImage(&img);  
//	cvReleaseImage(&img_rotate);  
//	return 0;  
//}  


int main( )
{
	Point2f srcTri[3];
	Point2f dstTri[3];
	Mat rot_mat( 2, 3, CV_32FC1 );
	Mat warp_mat( 2, 3, CV_32FC1 );
	Mat src, warp_dst, warp_rotate_dst;
	//����ͼ��
	src = imread( "baboon.jpg", 1 );
	warp_dst = Mat::zeros( src.rows, src.cols, src.type() );
	// ��3����ȷ��A����任
	srcTri[0] = Point2f( 0,0 );
	srcTri[1] = Point2f( src.cols - 1, 0 );
	srcTri[2] = Point2f( 0, src.rows - 1 );
	dstTri[0] = Point2f( src.cols*0.0, src.rows*0.33 );
	dstTri[1] = Point2f( src.cols*0.85, src.rows*0.25 );
	dstTri[2] = Point2f( src.cols*0.15, src.rows*0.7 );
	warp_mat = getAffineTransform( srcTri, dstTri );
	warpAffine( src, warp_dst, warp_mat, warp_dst.size() );
	/// ��ת����
	Point center = Point( warp_dst.cols/2, warp_dst.rows/2 );
	double angle = -50.0;
	double scale = 0.6;
	rot_mat = getRotationMatrix2D( center, angle, scale );
	warpAffine( warp_dst, warp_rotate_dst, rot_mat, warp_dst.size() );
	////OpenCV 1.0����ʽ
	//IplImage * img=cvLoadImage("baboon.jpg");
	//IplImage *img_rotate=cvCloneImage(img);
	//CvMat M =warp_mat;
	//cvWarpAffine(img,img_rotate, &M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) );
	//cvShowImage("Wrap2",img_rotate);

	namedWindow( "Source", CV_WINDOW_AUTOSIZE );
	imshow( "Source", src );
	namedWindow( "Wrap", CV_WINDOW_AUTOSIZE );
	imshow( "Wrap", warp_dst );
	namedWindow("Wrap+Rotate", CV_WINDOW_AUTOSIZE );
	imshow( "Wrap+Rotate", warp_rotate_dst );
	waitKey(0);
	return 0;
}