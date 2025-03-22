# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import cv2
from pathlib import Path

from models.experimental.blazepose.demo.blazebase import resize_pad, denormalize_detections
from models.experimental.blazepose.demo.blazepose import BlazePose
from models.experimental.blazepose.demo.blazepose_landmark import BlazePoseLandmark

from models.experimental.blazepose.demo.visualization import draw_detections, draw_landmarks, draw_roi, POSE_CONNECTIONS

"""
BlazePose is a two stage pipeline. (1, 896, 12) - is a regressor output for the first detector stage.
It contains 896 anchors, each anchor(or prior) contains:
SSD standard:

dx, dy for face bounding box center
width, height for face bounding box
Additional keypoints:
dx, dy for upper-body center
dx, dy for upper-body rotation point
dx, dy for full-body center
dx, dy for full-body rotation point
Then depending on the pipeline (upper-body vs full-body) we estimate appropriate ROI
on the basis of center and rotation point, and pass it to the tracker model,
which produce either 25 kp (for upper-body model, which is released in MediaPipe) or 33 kp for Full-Body.

This implementations is only for upper-body model.
"""


def model_location_generator(rel_path):
    internal_weka_path = Path("/mnt/MLPerf")
    has_internal_weka = (internal_weka_path / "bit_error_tests").exists()

    if has_internal_weka:
        return Path("/mnt/MLPerf") / rel_path
    else:
        return Path("/opt/tt-metal-models") / rel_path


model_path = model_location_generator("tt_dnn-models/Blazepose/models/")
DETECTOR_MODEL = str(model_path / "blazepose.pth")
LANDMARK_MODEL = str(model_path / "blazepose_landmark.pth")
ANCHORS = str(model_path / "anchors_pose.npy")

data_path = model_location_generator("tt_dnn-models/Blazepose/data/")
IMAGE_FILE = str(data_path / "yoga.jpg")
OUTPUT_FILE = "yoga_output.jpg"

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

pose_detector = BlazePose()

pose_detector.load_weights(DETECTOR_MODEL)
pose_detector.load_anchors(ANCHORS)
# pose_detector.state_dict()

pose_regressor = BlazePoseLandmark()
pose_regressor.load_weights(LANDMARK_MODEL)

image = cv2.imread(IMAGE_FILE)
image_height, image_width, _ = image.shape

frame = np.ascontiguousarray(image[:, ::-1, ::-1])

img1, img2, scale, pad = resize_pad(frame)

normalized_pose_detections = pose_detector.predict_on_image(img2)
print("Detector network returned ******************")
print(normalized_pose_detections)
print(normalized_pose_detections.size())

pose_detections = denormalize_detections(normalized_pose_detections, scale, pad)
print("Denormalized pose_detections ***************")
print(pose_detections)
print(pose_detections.size())

xc, yc, scale, theta = pose_detector.detection2roi(pose_detections)
img, affine, box = pose_regressor.extract_roi(frame, xc, yc, theta, scale)
flags, normalized_landmarks, mask = pose_regressor(img.to(gpu))
print("Landmark network returned *******************")
print(flags)
print(normalized_landmarks)
print(mask)

landmarks = pose_regressor.denormalize_landmarks(normalized_landmarks, affine)
print("Denormalized landmarks **********************")
print(landmarks)

# Originaly landmarks should be (x,y,z,visibility,presence) for every 33 keypoints .
# But this is only upper body detection
print(landmarks.size())

draw_detections(frame, pose_detections)

draw_roi(frame, box)

for i in range(len(flags)):
    landmark, flag = landmarks[i], flags[i]
    if flag > 0.5:
        draw_landmarks(frame, landmark, POSE_CONNECTIONS, size=2)

# Save image:
# cv2.imwrite(OUTPUT_FILE,frame)
