# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from ttnn.model_preprocessing import preprocess_model_parameters, fold_batch_norm2d_into_conv2d, infer_ttnn_module_args
from models.experimental.yolov10.reference.yolov10 import YOLOv10, Conv


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
        ttnn.from_torch(
            a, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        ),
        ttnn.from_torch(
            b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        ),
    )


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

    return parameters


def create_yolov10x_model_parameters(model: YOLOv10, input_tensor: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=None,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    feats = [80, 40, 20]
    strides = [8.0, 16.0, 32.0]

    anchors, strides = make_anchors(device, feats, strides)
    if "model" in parameters:
        parameters.model[23]["anchors"] = anchors
        parameters.model[23]["strides"] = strides

    parameters["model_args"] = model

    return parameters


def create_yolov10_model_parameters_detect(model, input_tensor_1, input_tensor_2, input_tensor_3, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=None,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model([input_tensor_1, input_tensor_2, input_tensor_3]), device=None
    )

    feats = [80, 40, 20]
    strides = torch.tensor([8.0, 16.0, 32.0])

    anchors, strides = make_anchors(device, feats, strides)  # Optimization: Processing make anchors outside model run

    parameters["anchors"] = anchors
    parameters["strides"] = strides
    parameters["model_args"] = model

    parameters["model"] = model

    return parameters


def create_yolov10x_input_tensors(
    device, batch_size=1, input_channels=3, input_height=640, input_width=640, input_dtype=ttnn.bfloat8_b
):
    torch_input_tensor = torch.randn(batch_size, input_channels, input_height, input_width)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(
        ttnn_input_tensor,
        dtype=input_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    return torch_input_tensor, ttnn_input_tensor

    return parameters
