import cv2
import numpy as np
import dlib

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

while(True):
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    counter =0
    """
    for (x, y, w, h) in faces:
        print ('Number of Faces :',  len(faces) ) 
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w]
        counter+=1
        img_item ='face'+str(counter)+'.png'
        cv2.imwrite(img_item, roi_color)

        color = (0, 255, 0)
        stroke = 2
        end_x =  x+w
        end_y = y+h
        cv2.rectangle(frame, (x,y), (end_x, end_y), color, stroke)
    """
    faces_d = detector(gray, 1) # result
    #to draw faces on image
    for result in faces_d:
        x = result.left()
        y = result.top()
        x1 = result.right()
        y1 = result.bottom()
        roi_color = frame[y:y1, x:x1]
        roi_gray = gray[y:y1, x:x1] 
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
        counter+=1
        img_item ='face'+str(counter)+'.png'
        cv2.imwrite(img_item, roi_color)

    cv2.imshow('frame', frame)
    if (cv2.waitKey(20) & 0xFF  == ord('q') ):
        break

cv2.release()
cv2.destroyAllWindows()

"""

#img = cv.imread('Resources/Photos/group.jpg')
#cv.imshow('Group of 5 people', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray People', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)


print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)



cv.waitKey(0)

"""
