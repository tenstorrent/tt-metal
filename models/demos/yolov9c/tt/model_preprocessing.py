# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, infer_ttnn_module_args, preprocess_model_parameters

import ttnn
from models.demos.yolov9c.reference.yolov9c import Conv, YoloV9


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    if isinstance(model, Conv):
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv, model.bn)
        parameters["conv"] = {}
        parameters["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    if isinstance(model, nn.ConvTranspose2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    return parameters


def create_yolov9c_input_tensors(
    device, batch_size=1, input_channels=3, input_height=640, input_width=640, model=False
):
    torch_input_tensor = torch.randn(batch_size, input_channels, input_height, input_width)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    memory_config = None
    if model:
        ttnn_input_tensor = F.pad(ttnn_input_tensor, (0, 29))
        memory_config = ttnn.create_sharded_memory_config(
            [6400, 32],
            core_grid=device.core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    memory_config = memory_config if memory_config else ttnn.L1_MEMORY_CONFIG
    ttnn_input_tensor = ttnn.from_torch(
        ttnn_input_tensor, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config
    )
    return torch_input_tensor, ttnn_input_tensor


def make_anchors(device, feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    for i, stride in enumerate(strides):
        h, w = feats[i], feats[i]
        sx = torch.arange(end=w) + grid_cell_offset
        sy = torch.arange(end=h) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride))

    a = torch.cat(anchor_points).transpose(0, 1).unsqueeze(0)
    b = torch.cat(stride_tensor).transpose(0, 1)

    return (
        ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )


def create_yolov9c_model_parameters(model: YoloV9, input_tensor: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    parameters["model_args"] = model

    feats = [
        input_tensor.shape[3] // 8,
        input_tensor.shape[3] // 16,
        input_tensor.shape[3] // 32,
    ]
    strides = [8.0, 16.0, 32.0]

    anchors, strides = make_anchors(device, feats, strides)
    if "model" in parameters:
        parameters.model[22]["anchors"] = anchors
        parameters.model[22]["strides"] = strides

    return parameters


def create_yolov9c_model_parameters_detect(
    model: YoloV9, input_tensor_1: torch.Tensor, input_tensor_2: torch.Tensor, input_tensor_3: torch.Tensor, device
):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(input_tensor_1, input_tensor_2, input_tensor_3), device=None
    )
    parameters["model_args"] = model

    feats = [80, 40, 20]  # Values depends on input resolution. Current: 640x640
    strides = [8.0, 16.0, 32.0]

    anchors, strides = make_anchors(device, feats, strides)  # Optimization: Processing make anchors outside model run

    parameters["anchors"] = anchors
    parameters["strides"] = strides
    parameters["model"] = model

    return parameters
