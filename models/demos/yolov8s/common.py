# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from ultralytics import YOLO


def load_torch_model(model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        torch_model = YOLO("yolov8s.pt")
        torch_model = torch_model.model
        torch_model.eval()
        state_dict = torch_model.state_dict()

    else:
        weights_path = (
            model_location_generator("vision-models/yolov8s", model_subdir="", download_if_ci_v2=True) / "yolov8s.pth"
        )
        state_dict = torch.load(weights_path)
    # torch_model = yolov8s.DetectionModel()

    # torch_model.load_state_dict(state_dict)
    # torch_model.eval()

    return torch_model
