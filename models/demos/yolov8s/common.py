# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from ultralytics import YOLO

YOLOV8S_L1_SMALL_SIZE = 24576


def load_torch_model(model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        torch_model = YOLO("yolov8s.pt").eval()
        return torch_model.model
    else:
        weights_path = (
            model_location_generator("vision-models/yolov8s", model_subdir="", download_if_ci_v2=True) / "yolov8s.pth"
        )
        state_dict = torch.load(weights_path)
        torch_model = YOLO("yolov8s.yaml")

        if "model.model" in list(state_dict.keys())[0]:
            torch_model.load_state_dict(state_dict)
            torch_model.eval()
            return torch_model.model
        else:
            torch_model = torch_model.model
            torch_model.load_state_dict(state_dict)
            torch_model.eval()
            return torch_model
