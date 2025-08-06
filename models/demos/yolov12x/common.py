# # SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# # SPDX-License-Identifier: Apache-2.0

import os

import torch
from ultralytics import YOLO

from models.demos.yolov12x.reference import yolov12x

YOLOV12_L1_SMALL_SIZE = 24576


def load_torch_model(model_location_generator=None):
    if model_location_generator is None or "TT_GH_CI_INFRA" not in os.environ:
        torch_model = YOLO("yolo12x.pt")
        ckpt = torch_model.model
    else:
        weights_path = (
            model_location_generator("vision-models/yolov12x", model_subdir="", download_if_ci_v2=True) / "yolov12x.pt"
        )
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]
        state_dict = model.state_dict() if hasattr(model, "state_dict") else ckpt["model"]
    elif isinstance(ckpt, torch.nn.Module):
        state_dict = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise TypeError(f"Unexpected checkpoint format: {type(ckpt)}")

    torch_model = yolov12x.YoloV12x()
    torch_model.load_state_dict(state_dict, strict=False)
    torch_model.eval()

    return torch_model
