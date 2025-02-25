# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from ttnn.model_preprocessing import preprocess_model_parameters, fold_batch_norm2d_into_conv2d, infer_ttnn_module_args

# from models.experimental.functional_yolov10.tt.common import Conv as ttnn_Conv
from models.experimental.functional_yolov10.reference.yolov10 import YOLOv10, Conv, C2f


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)

    if isinstance(model, Conv):
        print("inside  2 custom")
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
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    parameters["model_args"] = model

    return parameters


def create_yolov10x_input_tensors(device, batch_size=1, input_channels=3, input_height=640, input_width=640):
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
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return torch_input_tensor, ttnn_input_tensor

    # feats = [
    #     input_tensor.shape[3] // 8,
    #     input_tensor.shape[3] // 16,
    #     input_tensor.shape[3] // 32,
    # ]
    # strides = [8.0, 16.0, 32.0]

    # anchors, strides = make_anchors(device, feats, strides)
    # if "model" in parameters:
    #     parameters.model[16]["anchors"] = anchors
    #     parameters.model[16]["strides"] = strides

    return parameters
