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


# ===================================================


def preprocess_conv_parameter_b(parameter, *, dtype, device):
    parameter = ttnn.from_torch(
        parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, device=device
    )
    return parameter


def custom_preprocessor_b(model, name):
    import tt_lib as ttl

    device = ttl.device.GetDefaultDevice()

    parameters = {}
    if isinstance(model, nn.GroupNorm):
        parameters["weight"] = preprocess_groupnorm_parameter(model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = preprocess_groupnorm_parameter(model.bias, dtype=ttnn.bfloat16)

    if isinstance(model, nn.Conv2d):
        weight = torch.permute(model.weight, (2, 3, 0, 1))
        parameters["weight"] = preprocess_conv_parameter_b(weight, dtype=ttnn.bfloat16, device=device)
        parameters["bias"] = preprocess_conv_parameter_b(model.bias, dtype=ttnn.bfloat16, device=device)
    return parameters
