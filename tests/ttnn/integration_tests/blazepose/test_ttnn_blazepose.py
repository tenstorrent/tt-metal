# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import cv2
import os
import sys
import pytest
from pathlib import Path
import ttnn
from torch import nn
from models.experimental.blazepose.demo.blazebase import resize_pad, denormalize_detections

from models.experimental.functional_blazepose.tt.blazepose_utils import (
    # detection2roi,
    predict_on_image,
    # denormalize_detections_ref,
)

# from models.experimental.functional_blazepose.reference.torch_blazepose_landmark import (
#    extract_roi,
#    denormalize_landmarks,
#    basepose_land_mark,
# )
# from models.experimental.blazepose.visualization import draw_detections, draw_landmarks, draw_roi, POSE_CONNECTIONS

from models.experimental.blazepose.demo.blazepose import BlazePose
from models.experimental.blazepose.demo.blazepose_landmark import BlazePoseLandmark

from models.experimental.blazepose.visualization import draw_detections, draw_landmarks, draw_roi, POSE_CONNECTIONS

# from models.utility_functions import torch_random, skip_for_wormhole_b0
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc


def model_location_generator(rel_path):
    internal_weka_path = Path("/mnt/MLPerf")
    has_internal_weka = (internal_weka_path / "bit_error_tests").exists()

    if has_internal_weka:
        return Path("/mnt/MLPerf") / rel_path
    else:
        return Path("/opt/tt-metal-models") / rel_path


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)  # ,layout = ttnn.TILE_LAYOUT)
    return parameter


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        # weight = torch.permute(model.weight, (2, 3, 0, 1))
        # print("Shape of weight :", model.weight.shape)
        weight = model.weight
        bias = model.bias
        while weight.dim() < 4:
            weight = weight.unsqueeze(0)
        while bias.dim() < 4:
            bias = bias.unsqueeze(0)
        parameters["weight"] = preprocess_conv_parameter(weight, dtype=ttnn.bfloat16)
        parameters["bias"] = preprocess_conv_parameter(bias, dtype=ttnn.bfloat16)
    return parameters


# @skip_for_wormhole_b0()
# @pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_blazepsoe(reset_seeds, device):
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

    # parameters = preprocess_model_parameters(
    #    initialize_model=lambda: pose_detector, convert_to_ttnn=lambda *_: True, custom_preprocessor=custom_preprocessor
    # )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: pose_detector,
        convert_to_ttnn=lambda *_: False,
    )

    anchors = torch.tensor(np.load(ANCHORS), dtype=torch.float32)

    output = predict_on_image(img2, parameters, anchors, device)
    out1 = ttnn.to_torch(output[0])
    out2 = ttnn.to_torch(output[1])
    print("Shapes :", out1.shape, " ", normalized_pose_detections[0].shape)
    print("Shapes :", out2.shape, " ", normalized_pose_detections[1].shape)
    assert_with_pcc(normalized_pose_detections[0], out1, 0.9)
    assert_with_pcc(normalized_pose_detections[1], out2, 0.7)
