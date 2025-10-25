# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from ultralytics import YOLO

from models.demos.yolov8x.reference import yolov8x

YOLOV8X_L1_SMALL_SIZE = 24576


def load_torch_model(model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        model = YOLO("yolov8x.pt")
        model = model.model
        model.eval()
        state_dict = model.state_dict()

    else:
        weights_path = (
            model_location_generator("vision-models/yolov8x", model_subdir="", download_if_ci_v2=True) / "yolov8x.pth"
        )
        state_dict = torch.load(weights_path)
    torch_model = yolov8x.DetectionModel()

    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    return torch_model
