# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from ultralytics import YOLO

# Baseline L1 small buffer size tuned at 640x640; scale for other resolutions via yolov8l_l1_small_size_for_res.
_YOLOV8L_L1_SMALL_BASE_640 = 24576

YOLOV8L_SUPPORTED_INPUT_RESOLUTIONS = (640, 1280)
YOLOV8L_DEFAULT_INPUT_H = 640
YOLOV8L_DEFAULT_INPUT_W = 640

# Backward-compatible defaults used by existing call sites.
YOLOV8L_INPUT_H = YOLOV8L_DEFAULT_INPUT_H
YOLOV8L_INPUT_W = YOLOV8L_DEFAULT_INPUT_W


def yolov8l_l1_small_size_for_res(inp_h: int, inp_w: int) -> int:
    return int(_YOLOV8L_L1_SMALL_BASE_640 * inp_h * inp_w // (640 * 640))


YOLOV8L_L1_SMALL_SIZE = yolov8l_l1_small_size_for_res(YOLOV8L_INPUT_H, YOLOV8L_INPUT_W)

# Trace capture region (bytes), scaled from 640² graph tuning.
_YOLOV8L_TRACE_REGION_BASE_1CQ_640 = 3686400
_YOLOV8L_TRACE_REGION_BASE_E2E_640 = 6434816


def yolov8l_trace_region_size_1cq_for_res(inp_h: int, inp_w: int) -> int:
    return int(_YOLOV8L_TRACE_REGION_BASE_1CQ_640 * inp_h * inp_w // (640 * 640))


def yolov8l_trace_region_size_e2e_for_res(inp_h: int, inp_w: int) -> int:
    return int(_YOLOV8L_TRACE_REGION_BASE_E2E_640 * inp_h * inp_w // (640 * 640))


YOLOV8L_TRACE_REGION_SIZE_1CQ = yolov8l_trace_region_size_1cq_for_res(YOLOV8L_INPUT_H, YOLOV8L_INPUT_W)
YOLOV8L_TRACE_REGION_SIZE_E2E = yolov8l_trace_region_size_e2e_for_res(YOLOV8L_INPUT_H, YOLOV8L_INPUT_W)


def load_torch_model(model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        torch_model = YOLO("yolov8l.pt").eval()
        return torch_model.model
    else:
        weights_path = (
            model_location_generator("vision-models/yolov8l", model_subdir="", download_if_ci_v2=True) / "yolov8l.pth"
        )
        state_dict = torch.load(weights_path)
        torch_model = YOLO("yolov8l.yaml")

        if "model.model" in list(state_dict.keys())[0]:
            torch_model.load_state_dict(state_dict)
            torch_model.eval()
            return torch_model.model
        else:
            torch_model = torch_model.model
            torch_model.load_state_dict(state_dict)
            torch_model.eval()
            return torch_model
