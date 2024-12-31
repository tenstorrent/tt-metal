# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
from typing import Tuple
import math
from models.demos.stable_diffusion.tt.resnetblock2d_utils import get_weights, get_mask_tensor


def preprocess_groupnorm_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)
    return parameter


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.GroupNorm):
        weight = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        bias = ttnn.from_torch(model.bias, dtype=ttnn.bfloat16)
        grid_size = ttnn.CoreGrid(y=4, x=8)
        parameters["weight"], parameters["bias"] = weight, bias
        parameters["tt_weight"], parameters["tt_bias"] = get_weights(weight, bias, model.num_channels, grid_size.y)
        parameters["input_mask_tensor"] = get_mask_tensor(model.num_channels, model.num_groups, grid_size.y)

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


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
    device=None,
):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    tensor = ttnn.arange(0, half_dim, dtype=ttnn.float32, device=device)
    tensor = ttnn.to_layout(tensor, layout=ttnn.TILE_LAYOUT)
    exponent = -math.log(max_period) * tensor
    exponent = exponent * (1 / (half_dim - downscale_freq_shift))

    emb = ttnn.exp(exponent)
    timesteps = ttnn.reshape(timesteps, (timesteps.shape[-1], 1))
    emb = ttnn.reshape(emb, (1, emb.shape[-1]))

    timesteps = ttnn.to_layout(timesteps, layout=ttnn.TILE_LAYOUT)
    emb = ttnn.to_layout(emb, layout=ttnn.TILE_LAYOUT)
    emb = timesteps * emb

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = ttnn.concat([ttnn.sin(emb), ttnn.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = ttnn.concat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def Timesteps(
    timesteps, num_channels: int, flip_sin_to_cos: bool, freq_shift: float, scale: int = 1, device=None, flag=False
) -> Tuple[int, int]:
    t_emb = get_timestep_embedding(timesteps, num_channels, flip_sin_to_cos, freq_shift, scale, device=device)
    return t_emb
