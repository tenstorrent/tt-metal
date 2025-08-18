# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from ultralytics import YOLO

from models.demos.yolov9c.reference.yolov9c import YoloV9

YOLOV9C_L1_SMALL_SIZE = 24576


def load_torch_model(model_task="segment", model_location_generator=None):
    weights = "yolov9c-seg.pt" if model_task == "segment" else "yolov9c.pt"
    enable_segment = model_task == "segment"

    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        torch_model = YOLO(weights)
        state_dict = torch_model.state_dict()
    else:
        if model_task == "segment":
            weights_path = (
                model_location_generator("vision-models/yolov9c", model_subdir="", download_if_ci_v2=True)
                / "yolov9c-seg.pth"
            )
        else:
            weights_path = (
                model_location_generator("vision-models/yolov9c", model_subdir="", download_if_ci_v2=True)
                / "yolov9c.pth"
            )
        state_dict = torch.load(weights_path)

    torch_model = YoloV9(enable_segment=enable_segment)
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    return torch_model
