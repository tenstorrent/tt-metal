# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from ultralytics import YOLO

from models.experimental.yolov12x.reference import yolov12x


def load_torch_model(model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        torch_model = YOLO("yolo12x.pt")
        state_dict = torch_model.state_dict()

    else:
        weights_path = (
            model_location_generator("vision-models/yolov12x", model_subdir="", download_if_ci_v2=True) / "yolov12x.pth"
        )
        state_dict = torch.load(weights_path)
    torch_model = yolov12x.YoloV12x()

    state_dict = torch_model.state_dict() if state_dict is None else state_dict

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)

    torch_model.eval()

    return torch_model
