# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


def preprocess_linear_weight(weight, mesh_mapper, mesh_device):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(
        weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper, device=mesh_device
    )
    return weight


def preprocess_linear_bias(bias, mesh_mapper, mesh_device):
    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(
        bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper, device=mesh_device
    )
    return bias


def preprocess_layernorm_parameter(parameter, mesh_mapper, mesh_device):
    parameter = parameter.reshape((1, -1))
    parameter = ttnn.from_torch(
        parameter, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper, device=mesh_device
    )
    return parameter


def preprocess_embedding_weight(weight, mesh_mapper, mesh_device):
    weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper, device=mesh_device)
    return weight


def preprocess_attn_weight(query, key, value, mesh_mapper, mesh_device):
    qkv_weight = torch.cat(
        [query, key, value],
        dim=0,
    )
    weight = preprocess_linear_weight(qkv_weight, mesh_mapper, mesh_device)
    return weight


def preprocess_attn_bias(query, key, value, mesh_mapper, mesh_device):
    qkv_weight = torch.cat(
        [query, key, value],
        dim=0,
    )
    weight = preprocess_linear_bias(qkv_weight, mesh_mapper, mesh_device)
    return weight
