# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from ultralytics import YOLO

from models.demos.yolov11l.reference import yolov11

YOLOV11_SUPPORTED_INPUT_RESOLUTIONS = (640, 1280)
YOLOV11_DEFAULT_INPUT_H = 640
YOLOV11_DEFAULT_INPUT_W = 640

_YOLOV11_L1_SMALL_BASE_640 = 16000
_YOLOV11_TRACE_REGION_BASE_E2E_640 = 6434816


def yolov11_l1_small_size_for_res(inp_h: int, inp_w: int) -> int:
    return int(_YOLOV11_L1_SMALL_BASE_640 * inp_h * inp_w // (640 * 640))


def yolov11_trace_region_size_e2e_for_res(inp_h: int, inp_w: int) -> int:
    return int(_YOLOV11_TRACE_REGION_BASE_E2E_640 * inp_h * inp_w // (640 * 640))


YOLOV11_L1_SMALL_SIZE = yolov11_l1_small_size_for_res(YOLOV11_DEFAULT_INPUT_H, YOLOV11_DEFAULT_INPUT_W)


def load_torch_model(model_location_generator=None):
    torch_model = yolov11.YoloV11()
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        model = YOLO("yolo11l.pt")
        state_dict = {k.replace("model.", "", 1): v for k, v in model.state_dict().items()}
    else:
        weights_path = (
            model_location_generator("vision-models/yolov11l", model_subdir="", download_if_ci_v2=True) / "yolov11l.pt"
        )
        yolov11_ckpt = torch.load(weights_path)
        state_dict = yolov11_ckpt["model"].float().state_dict()

    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    return torch_model
