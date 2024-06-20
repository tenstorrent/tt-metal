# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import cv2
import os
import sys
import pytest
from pathlib import Path

from models.experimental.blazepose.demo.blazebase import resize_pad, denormalize_detections

from models.experimental.functional_blazepose.reference.torch_blazepose import (
    detection2roi,
    predict_on_image,
    denormalize_detections_ref,
)
from models.experimental.functional_blazepose.reference.torch_blazepose_landmark import (
    extract_roi,
    denormalize_landmarks,
    basepose_land_mark,
)
from models.experimental.blazepose.visualization import draw_detections, draw_landmarks, draw_roi, POSE_CONNECTIONS

from models.experimental.blazepose.demo.blazepose import BlazePose
from models.experimental.blazepose.demo.blazepose_landmark import BlazePoseLandmark

from models.experimental.blazepose.visualization import draw_detections, draw_landmarks, draw_roi, POSE_CONNECTIONS
from models.utility_functions import torch_random, skip_for_wormhole_b0
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc


def model_location_generator(rel_path):
    internal_weka_path = Path("/mnt/MLPerf")
    has_internal_weka = (internal_weka_path / "bit_error_tests").exists()

    if has_internal_weka:
        return Path("/mnt/MLPerf") / rel_path
    else:
        return Path("/opt/tt-metal-models") / rel_path


@skip_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1])
def test_blazepsoe(batch_size, reset_seeds):
    model_path = model_location_generator("tt_dnn-models/Blazepose/models/")
    DETECTOR_MODEL = str(model_path / "blazepose.pth")
    LANDMARK_MODEL = str(model_path / "blazepose_landmark.pth")
    ANCHORS = str(model_path / "anchors_pose.npy")

    pose_detector = BlazePose()
    pose_detector.load_weights(DETECTOR_MODEL)
    pose_detector.load_anchors(ANCHORS)
    pose_detector.state_dict()

    data_path = model_location_generator("tt_dnn-models/Blazepose/data/")
    IMAGE_FILE = str(data_path / "yoga.jpg")
    OUTPUT_FILE = "yoga_output.jpg"
    image = cv2.imread(IMAGE_FILE)
    image_height, image_width, _ = image.shape
    frame = np.ascontiguousarray(image[:, ::-1, ::-1])

    img1, img2, scale, pad = resize_pad(frame)

    normalized_pose_detections = pose_detector.predict_on_image(img2)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: pose_detector,
        convert_to_ttnn=lambda *_: False,
    )

    anchors = torch.tensor(np.load(ANCHORS), dtype=torch.float32)

    output = predict_on_image(img2, parameters, anchors)
    assert_with_pcc(normalized_pose_detections, output, 0.9999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1])
def test_blazepsoe_model(batch_size):
    model_path = model_location_generator("tt_dnn-models/Blazepose/models/")
    DETECTOR_MODEL = str(model_path / "blazepose.pth")
    LANDMARK_MODEL = str(model_path / "blazepose_landmark.pth")
    ANCHORS = str(model_path / "anchors_pose.npy")

    pose_detector = BlazePose()
    pose_detector.load_weights(DETECTOR_MODEL)
    pose_detector.load_anchors(ANCHORS)
    pose_detector.state_dict()

    data_path = model_location_generator("tt_dnn-models/Blazepose/data/")
    IMAGE_FILE = str(data_path / "yoga.jpg")
    OUTPUT_FILE = "yoga_output_ref.jpg"
    image = cv2.imread(IMAGE_FILE)
    image_height, image_width, _ = image.shape
    frame = np.ascontiguousarray(image[:, ::-1, ::-1])

    img1, img2, scale_resize, pad = resize_pad(frame)

    normalized_pose_detections = pose_detector.predict_on_image(img2)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: pose_detector,
        convert_to_ttnn=lambda *_: False,
    )

    anchors = torch.tensor(np.load(ANCHORS), dtype=torch.float32)

    normalized_pose_detections_ref = predict_on_image(img2, parameters, anchors)
    assert_with_pcc(normalized_pose_detections, normalized_pose_detections_ref, 0.9999)

    pose_regressor = BlazePoseLandmark()

    pose_regressor.load_weights(LANDMARK_MODEL)
    pose_detections = denormalize_detections(normalized_pose_detections, scale_resize, pad)
    xc, yc, scale, theta = pose_detector.detection2roi(pose_detections)

    img, affine, box = pose_regressor.extract_roi(frame, xc, yc, theta, scale)
    flags, normalized_landmarks, mask = pose_regressor(img)
    landmarks = pose_regressor.denormalize_landmarks(normalized_landmarks, affine)

    pose_detections_ref = denormalize_detections_ref(normalized_pose_detections_ref, scale_resize, pad)
    xc_ref, yc_ref, scale_ref, theta_ref = detection2roi(pose_detections_ref)
    img_ref, affine_ref, box_ref = extract_roi(frame, xc_ref, yc_ref, theta_ref, scale_ref)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: pose_regressor,
        convert_to_ttnn=lambda *_: False,
    )

    flags_ref, normalized_landmarks_ref, mask_ref = basepose_land_mark(img_ref, parameters)
    landmarks_ref = denormalize_landmarks(normalized_landmarks_ref, affine_ref)

    draw_detections(frame, pose_detections_ref)

    draw_roi(frame, box_ref)

    for i in range(len(flags)):
        landmark, flag = landmarks_ref[i], flags_ref[i]
        if flag > 0.5:
            draw_landmarks(frame, landmark, POSE_CONNECTIONS, size=2)

    # Save image:
    # cv2.imwrite(OUTPUT_FILE,frame)
    assert_with_pcc(flags, flags_ref, 0.9999)
    assert_with_pcc(normalized_landmarks, normalized_landmarks_ref, 0.9999)
    assert_with_pcc(mask, mask_ref, 0.9999)
