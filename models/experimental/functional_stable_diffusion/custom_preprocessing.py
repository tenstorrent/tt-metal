# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import ttnn


def preprocess_groupnorm_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.GroupNorm):
        parameters["weight"] = preprocess_groupnorm_parameter(model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = preprocess_groupnorm_parameter(model.bias, dtype=ttnn.bfloat16)

    if isinstance(model, nn.Conv2d):
        weight = torch.permute(model.weight, (2, 3, 0, 1))
        parameters["weight"] = preprocess_conv_parameter(weight, dtype=ttnn.bfloat16)
        parameters["bias"] = preprocess_conv_parameter(model.bias, dtype=ttnn.bfloat16)
    return parameters
