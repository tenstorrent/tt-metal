# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import ttnn
import torch
import pytest
import torch.nn as nn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.functional_yolov8m.tt.ttnn_optimized_yolov8m import YOLOv8m
from models.experimental.functional_yolov8m.reference import yolov8m_utils

from models.experimental.functional_yolov8m.tt.ttnn_yolov8m_utils import (
    ttnn_decode_bboxes,
    custom_preprocessor,
)

try:
    sys.modules["ultralytics"] = yolov8m_utils
    sys.modules["ultralytics.nn.tasks"] = yolov8m_utils
    sys.modules["ultralytics.nn.modules.conv"] = yolov8m_utils
    sys.modules["ultralytics.nn.modules.block"] = yolov8m_utils
    sys.modules["ultralytics.nn.modules.head"] = yolov8m_utils

except KeyError:
    print("models.experimental.functional_yolov8m.reference.yolov8m_utils not found.")


def decode_bboxes(distance, anchor_points, xywh=True, dim=1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset

        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        w = "models/experimental/functional_yolov8m/demo/yolov8m.pt"
        ckpt = torch.load(w, map_location=map_location)
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().eval())
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]
    else:
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [torch.rand((1, 3, 640, 640))], ids=["input_tensor1"])
def test_optimized_yolov8m(device, input_tensor, reset_seeds):
    disable_persistent_kernel_cache()

    torch_model = attempt_load("yolov8m.pt", map_location="cpu")

    state_dict = torch_model.state_dict()

    torch_model_output = torch_model(input_tensor)[0]

    parameters = custom_preprocessor(device, state_dict)
    ttnn_input = input_tensor.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(
        ttnn_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    ttnn_model_output = YOLOv8m(device, ttnn_input, parameters)[0]
    ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 1)
    logger.info(f"Passing: {passing}, PCC: {pcc}")
