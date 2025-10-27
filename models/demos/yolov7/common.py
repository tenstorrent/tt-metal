# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import types

import torch

from models.demos.yolov7.reference.model import Yolov7_model

YOLOV7_L1_SMALL_SIZE = 24576


def setup_module_mapping():
    """Setup module mapping to handle YOLOv7 checkpoint loading."""
    if "models.yolo" not in sys.modules:
        yolo_module = types.ModuleType("yolo")
        sys.modules["models.yolo"] = yolo_module

        # Import all the classes we need from our reference model
        from models.demos.yolov7.reference.model import MP, SPPCSPC, Concat, Conv, Detect, RepConv, Yolov7_model

        yolo_module.Model = Yolov7_model
        yolo_module.Conv = Conv
        yolo_module.SPPCSPC = SPPCSPC
        yolo_module.RepConv = RepConv
        yolo_module.Detect = Detect
        yolo_module.Concat = Concat
        yolo_module.MP = MP

        # Always override models.common to ensure we have the classes needed for checkpoint loading
        common_module = types.ModuleType("common")
        sys.modules["models.common"] = common_module

        common_module.Conv = Conv
        common_module.SPPCSPC = SPPCSPC
        common_module.RepConv = RepConv
        common_module.Detect = Detect
        common_module.Concat = Concat
        common_module.MP = MP


def load_torch_model(model_location_generator=None):
    # Setup module mapping before loading checkpoint
    setup_module_mapping()

    torch_model = Yolov7_model()
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        weights_path = "models/demos/yolov7/yolov7.pt"
        if not os.path.exists(weights_path):
            torch.hub.download_url_to_file(
                "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt", weights_path
            )
    else:
        weights_path = (
            model_location_generator("vision-models/yolov7", model_subdir="", download_if_ci_v2=True) / "yolov7.pt"
        )

    # Load the actual checkpoint with module mapping in place
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"].float().state_dict()
    torch_model.load_state_dict(state_dict, strict=False)
    torch_model.eval()

    return torch_model
