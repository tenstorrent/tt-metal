# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
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
