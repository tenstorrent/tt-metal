# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import ttnn
import torch
import pytest
from pathlib import Path
import torch.nn as nn
from loguru import logger
from ultralytics import YOLO
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import disable_persistent_kernel_cache
from models.demos.yolov8x.tt.ttnn_yolov8x import TtYolov8xModel, TtConv, TtC2f, TtSppf, TtDFL
from models.demos.yolov8x.tt.ttnn_yolov8x_utils import (
    ttnn_decode_bboxes,
    custom_preprocessor,
)
from models.demos.yolov8x.reference import yolov8x


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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True, ids=["0"])
@pytest.mark.parametrize(
    "input_tensor",
    [torch.rand((1, 3, 640, 640))],
    ids=["input_tensor1"],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
def test_yolov8x_640(device, input_tensor, use_weights_from_ultralytics):
    disable_persistent_kernel_cache()

    inp_h, inp_w = input_tensor.shape[2], input_tensor.shape[3]
    if use_weights_from_ultralytics:
        torch_model = YOLO("yolov8x.pt")
        torch_model = torch_model.model
        torch_model.eval()
        state_dict = torch_model.state_dict()
    else:
        torch_model = yolov8x.DetectionModel()
        torch_model.eval()
        state_dict = torch_model.state_dict()
    parameters = custom_preprocessor(device, state_dict, inp_h, inp_w)
    ttnn_model = TtYolov8xModel(device=device, parameters=parameters)
    parameters = custom_preprocessor(device, state_dict, inp_h=inp_h, inp_w=inp_w)

    n, c, h, w = input_tensor.shape
    if c == 3:
        c = 16
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input = ttnn_input.to(device, input_mem_config)

    with torch.inference_mode():
        ttnn_model_output = ttnn_model(ttnn_input)[0]
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    with torch.inference_mode():
        torch_model_output = torch_model(input_tensor)[0]

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand((1, 3, 640, 640)))], ids=["input_tensor1"])
def test_Conv(device, input_tensor):
    disable_persistent_kernel_cache()

    torch_model = YOLO("yolov8x.pt")
    torch_model = torch_model.model
    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))

    state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict)

    with torch.inference_mode():
        conv_0 = TtConv(
            device,
            parameters,
            "model.0",
            input_params=[3, 2, 1, 80, 3],
            change_shard=False,
            deallocate_activation=True,
            act_block_h=True,
        )
        conv_0, out_h, out_w = conv_0(ttnn_input)
        ttnn_model_output = ttnn.to_torch(conv_0)
        ttnn_model_output = ttnn_model_output.reshape((1, out_h, out_w, ttnn_model_output.shape[-1]))
        ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))

    submodule = torch_model.get_submodule("model.0")

    with torch.inference_mode():
        torch_model_output = submodule(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand((1, 160, 160, 160)))], ids=["input_tensor1"])
def test_C2f(device, input_tensor, reset_seeds):
    disable_persistent_kernel_cache()

    torch_model = YOLO("yolov8x.pt")
    torch_model = torch_model.model
    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))

    ttnn_input = ttnn.from_device(ttnn_input)

    state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict)

    c2f_configs = {
        "model.2": {"input_params": ((1, 1, 0, 160, 160), (1, 1, 0, 160, 400), (3, 1, 1, 80, 80))},
    }

    with torch.inference_mode():
        c2f_2 = TtC2f(
            device,
            parameters,
            "model.2",
            n=3,
            shortcut=True,
            change_shard=False,
            input_params=c2f_configs["model.2"]["input_params"],
        )
        c2f_2, out_h, out_w = c2f_2(ttnn_input)
        ttnn_model_output = ttnn.to_torch(c2f_2)
        ttnn_model_output = ttnn_model_output.reshape((1, out_h, out_w, ttnn_model_output.shape[-1]))
        ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))

    submodule = torch_model.get_submodule("model.2")

    with torch.inference_mode():
        torch_model_output = submodule(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.97)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand((1, 640, 20, 20)))], ids=["input_tensor1"])
def test_SPPF(device, input_tensor, reset_seeds):
    disable_persistent_kernel_cache()

    torch_model = YOLO("yolov8x.pt")
    torch_model = torch_model.model
    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))

    ttnn_input = ttnn.from_device(ttnn_input)

    state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict)

    sppf_configs = {"input_params": ((1, 1, 0, 320, 640), (1, 1, 0, 640, 1280))}

    with torch.inference_mode():
        sppf = TtSppf(device, parameters, "model.9", input_params=sppf_configs["input_params"], batch_size=1)
        ttnn_model_output, out_h, out_w = sppf(ttnn_input)
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)
        ttnn_model_output = ttnn_model_output.reshape((1, out_h, out_w, ttnn_model_output.shape[-1]))
        ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))

    submodule = torch_model.get_submodule("model.9")

    with torch.inference_mode():
        torch_model_output = submodule(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand((1, 64, 8400)))], ids=["input_tensor1"])
def test_DFL(device, input_tensor, reset_seeds):
    disable_persistent_kernel_cache()

    torch_model = YOLO("yolov8x.pt")
    torch_model = torch_model.model
    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict)

    with torch.inference_mode():
        dfl = TtDFL(device, parameters, "model.22.dfl", input_params=[1, 1, 0, 1, 16])
        ttnn_model_output = dfl(ttnn_input)
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    submodule = torch_model.get_submodule("model.22.dfl")

    with torch.inference_mode():
        torch_model_output = submodule(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "distance, anchors", [(torch.rand((1, 4, 8400)), torch.rand((1, 2, 8400)))], ids=["input_tensor"]
)
def test_dist2bbox(device, distance, anchors):
    disable_persistent_kernel_cache()

    ttnn_distance = ttnn.from_torch(distance, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_anchors = ttnn.from_torch(anchors, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_model_output = ttnn_decode_bboxes(device, ttnn_distance, ttnn_anchors)
    ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    torch_model_output = decode_bboxes(distance, anchors)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")
