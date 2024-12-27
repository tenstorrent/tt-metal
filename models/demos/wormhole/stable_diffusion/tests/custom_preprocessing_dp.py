# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torchvision
from torch import nn


def preprocess_groupnorm_parameter(parameter, *, dtype, mesh_mapper=None):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper)
    return parameter


def preprocess_conv_parameter(parameter, *, dtype, mesh_mapper=None):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper)
    return parameter


def custom_preprocessor(model, name, mesh_mapper=None):
    parameters = {}
    if isinstance(model, nn.GroupNorm):
        parameters["weight"] = preprocess_groupnorm_parameter(
            model.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters["bias"] = preprocess_groupnorm_parameter(model.bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

    if isinstance(model, nn.Conv2d):
        weight = torch.permute(model.weight, (2, 3, 0, 1))
        parameters["weight"] = preprocess_conv_parameter(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
        parameters["bias"] = preprocess_conv_parameter(model.bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

    if isinstance(model, (nn.Linear, nn.LayerNorm)):
        weight = model.weight.T.contiguous()
        while len(weight.shape) < 4:
            weight = weight.unsqueeze(0)
        parameters["weight"] = ttnn.from_torch(
            weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper
        )
        if model.bias is not None:
            bias = model.bias
            while len(bias.shape) < 4:
                bias = bias.unsqueeze(0)
            parameters["bias"] = ttnn.from_torch(
                bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper
            )
    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model,
            name,
            mesh_mapper=mesh_mapper,
        )

    return custom_mesh_preprocessor
