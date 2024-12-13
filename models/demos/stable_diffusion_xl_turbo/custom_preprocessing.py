# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn


def preprocess_conv_parameter(parameter, *, dtype):
    while len(parameter.shape) < 4:
        parameter = parameter.unsqueeze(0)
    parameter = ttnn.from_torch(parameter, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    return parameter


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, nn.Conv2d):
        weight = torch.permute(model.weight, (2, 3, 0, 1))
        parameters["weight"] = preprocess_conv_parameter(weight, dtype=ttnn.bfloat8_b)
        parameters["bias"] = preprocess_conv_parameter(model.bias, dtype=ttnn.bfloat8_b)


def preprocess_conv_parameter_2(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)
    return parameter


def custom_preprocessor_2(model, name):
    parameters = {}

    if isinstance(model, nn.Conv2d):
        parameters["weight"] = preprocess_conv_parameter_2(model.weight, dtype=ttnn.float32)
        bias = model.bias.reshape((1, 1, 1, -1))
        parameters["bias"] = preprocess_conv_parameter_2(bias, dtype=ttnn.float32)

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
