# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch

from models.demos.utils.common_demo_utils import attempt_load
from models.demos.yolov8s_world.reference import yolov8s_world

YOLOV8SWORLD_L1_SMALL_SIZE = 24576


def load_torch_model(model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        weights_torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
        torch_model = yolov8s_world.YOLOWorld(model_torch=weights_torch_model)
        state_dict = weights_torch_model.state_dict()

    else:
        weights_path = (
            model_location_generator("vision-models/yolov8s_world", model_subdir="", download_if_ci_v2=True)
            / "yolov8s-world.pth"
        )
        state_dict = torch.load(weights_path)
        torch_model = yolov8s_world.YOLOWorld(weights_path=weights_path)

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        new_state_dict[name1] = parameter2

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    return torch_model
