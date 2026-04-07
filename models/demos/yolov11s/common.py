# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from ultralytics import YOLO

from models.demos.yolov11s.reference import yolov11s

YOLOV11_L1_SMALL_SIZE = 32768


def load_torch_model(model_location_generator=None):
    torch_model = yolov11s.YoloV11()
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        model = YOLO("yolo11s.pt")
        state_dict = {k.replace("model.", "", 1): v for k, v in model.state_dict().items()}
    else:
        weights_path = (
            model_location_generator("vision-models/yolov11s", model_subdir="", download_if_ci_v2=True) / "yolov11s.pt"
        )
        yolov11_ckpt = torch.load(weights_path)
        state_dict = yolov11_ckpt["model"].float().state_dict()

    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    return torch_model
