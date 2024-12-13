# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn


def preprocess_conv_parameter_resnet(parameter, *, dtype):
    while len(parameter.shape) < 4:
        parameter = parameter.unsqueeze(0)
    parameter = ttnn.from_torch(parameter, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    return parameter


def custom_preprocessor_resnet(model, name):
    parameters = {}

    if isinstance(model, nn.Conv2d):
        weight = torch.permute(model.weight, (2, 3, 0, 1))
        parameters["weight"] = preprocess_conv_parameter_resnet(weight, dtype=ttnn.bfloat8_b)
        parameters["bias"] = preprocess_conv_parameter_resnet(model.bias, dtype=ttnn.bfloat8_b)


def preprocess_groupnorm_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)
    return parameter


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.GroupNorm):
        parameters["weight"] = preprocess_groupnorm_parameter(model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = preprocess_groupnorm_parameter(model.bias, dtype=ttnn.bfloat16)

    if isinstance(model, nn.Conv2d):
        parameters["weight"] = preprocess_conv_parameter(model.weight, dtype=ttnn.float32)
        bias = model.bias.reshape((1, 1, 1, -1))
        parameters["bias"] = preprocess_conv_parameter(bias, dtype=ttnn.float32)

    if isinstance(model, (nn.Linear, nn.LayerNorm)):
        weight = model.weight.T.contiguous()
        while len(weight.shape) < 4:
            weight = weight.unsqueeze(0)
        parameters["weight"] = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if model.bias is not None:
            bias = model.bias
            while len(bias.shape) < 4:
                bias = bias.unsqueeze(0)
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return parameters
