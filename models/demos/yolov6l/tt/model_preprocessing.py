# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import sys

import torch
import torch.nn as nn
from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters

import ttnn
from models.demos.yolov6l.reference.yolov6l import Model
from models.demos.yolov6l.reference.yolov6l_utils import fuse_model

sys.path.append("models/demos/yolov6l/reference/")


def generate_anchors(
    device,
    feats,
    fpn_strides,
    grid_cell_offset=0.5,
):
    anchor_points = []
    stride_tensor = []
    assert feats is not None

    for i, stride in enumerate(fpn_strides):
        h, w = feats[i]
        shift_x = torch.arange(end=w) + grid_cell_offset
        shift_y = torch.arange(end=h) + grid_cell_offset
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        anchor_point = torch.stack([shift_x, shift_y], axis=-1).to(torch.float)

        anchor_points.append(anchor_point.reshape([-1, 2]))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float))
    anchor_points = torch.cat(anchor_points)
    stride_tensor = torch.cat(stride_tensor)
    return (
        ttnn.from_torch(anchor_points, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(stride_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    if isinstance(model, nn.ConvTranspose2d):
        weight = model.weight
        bias = model.bias
        parameters["conv_t"] = {}
        parameters["conv_t"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv_t"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    return parameters


def create_yolov6l_model_parameters(model: Model, torch_input: torch.Tensor, device):
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

    feats = [(80, 80), (40, 40), (20, 20)]
    strides = torch.tensor([8.0, 16.0, 32.0])
    anchor_points, stride_tensor = generate_anchors(device, feats, strides)

    ones_tensor = torch.ones((1, 8400, 1), dtype=torch.float32)
    ones_tensor = ttnn.from_torch(ones_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    if "detect" in parameters:
        parameters.detect["anchors"] = anchor_points
        parameters.detect["strides"] = stride_tensor
        parameters.detect["ones_tensor"] = ones_tensor
    return parameters


def create_yolov6l_model_parameters_detect(model: Model, torch_input: torch.Tensor, device):
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

    feats = [(80, 80), (40, 40), (20, 20)]
    strides = torch.tensor([8.0, 16.0, 32.0])
    anchor_points, stride_tensor = generate_anchors(device, feats, strides)
    parameters["anchors"] = anchor_points
    parameters["strides"] = stride_tensor

    ones_tensor = torch.ones((1, 8400, 1), dtype=torch.float32)
    ones_tensor = ttnn.from_torch(ones_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    parameters["ones_tensor"] = ones_tensor
    return parameters


def create_yolov6l_model_parameters_bottlerep(model: Model, torch_input: torch.Tensor, device):
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


def create_yolov6l_model_parameters_repblock(model: Model, torch_input: torch.Tensor, device):
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


def create_yolov6l_model_parameters_bep_c3(model: Model, torch_input: torch.Tensor, device):
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


def create_yolov6l_model_parameters_sppf(model: Model, torch_input: torch.Tensor, device):
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


def load_torch_model_yolov6l():
    weights = "tests/ttnn/integration_tests/yolov6l/yolov6l.pt"
    if not os.path.exists(weights):
        os.system("bash models/demos/yolov6l/weights_download.sh")

    ckpt = torch.load(weights, map_location=torch.device("cpu"), weights_only=False)
    model = ckpt["ema" if ckpt.get("ema") else "model"].float()
    model = fuse_model(model).eval()
    return model
