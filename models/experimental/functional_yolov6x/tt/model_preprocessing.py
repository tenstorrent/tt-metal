# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters, fold_batch_norm2d_into_conv2d, infer_ttnn_module_args
from models.experimental.functional_yolov6x.reference.yolov6x import Yolov6x_model, Conv


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
        weight = model.weight
        bias = model.bias
        parameters["conv_t"] = {}
        parameters["conv_t"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv_t"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    return parameters


def create_yolov6x_model_parameters_sppf(model: Yolov6x_model, torch_input: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(torch_input), device=device
    )
    parameters["model_args"] = model
    return parameters


def create_yolov6x_model_parameters_detect(
    model: Yolov6x_model,
    input_tensor_1: torch.Tensor,
    input_tensor_2: torch.Tensor,
    input_tensor_3: torch.Tensor,
    device,
):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model([input_tensor_1, input_tensor_2, input_tensor_3]), device=device
    )
    feats = [80, 40, 20]
    strides = torch.tensor([8.0, 16.0, 32.0])

    anchors, strides = make_anchors(device, feats, strides)
    parameters["anchors"] = anchors
    parameters["strides"] = strides
    parameters["model_args"] = model

    return parameters


def create_yolov6x_model_parameters(model: Yolov6x_model, input_tensor: torch.Tensor, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(input_tensor), device=device
    )
    feats = [80, 40, 20]
    strides = torch.tensor([8.0, 16.0, 32.0])

    anchors, strides = make_anchors(device, feats, strides)
    if "model" in parameters:
        parameters.model[28]["anchors"] = anchors
        parameters.model[28]["strides"] = strides

    parameters["model_args"] = model

    return parameters
