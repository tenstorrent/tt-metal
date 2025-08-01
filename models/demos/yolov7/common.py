# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch

from models.demos.yolov7.reference.model import Yolov7_model

YOLOV7_L1_SMALL_SIZE = 24576


def load_torch_model(model_location_generator=None):
    torch_model = Yolov7_model()
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        weights_path = "models/demos/yolov7/tests/pcc/yolov7.pt"
        if not os.path.exists(weights_path):
            torch.hub.download_url_to_file(
                "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt", weights_path
            )
    else:
        weights_path = (
            model_location_generator("vision-models/yolov7", model_subdir="", download_if_ci_v2=True) / "yolov7.pt"
        )

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"].float().state_dict()
    torch_model.load_state_dict(state_dict, strict=False)
    torch_model.eval()

    return torch_model
