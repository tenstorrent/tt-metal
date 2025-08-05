# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch

from models.demos.yolov6l.reference.yolov6l_utils import fuse_model

YOLOV6L_L1_SMALL_SIZE = 24576


def load_torch_model(model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        weights = "models/demos/yolov6l/tests/pcc/yolov6l.pt"
        if not os.path.exists(weights):
            os.system("bash models/demos/yolov6l/weights_download.sh")
    else:
        weights = (
            model_location_generator("vision-models/yolov6l", model_subdir="", download_if_ci_v2=True) / "yolov6l.pt"
        )

    ckpt = torch.load(weights, map_location=torch.device("cpu"), weights_only=False)
    model = ckpt["ema" if ckpt.get("ema") else "model"].float()
    model = fuse_model(model).eval()

    return model
