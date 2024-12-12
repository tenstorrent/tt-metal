# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
import torch


def get_timestep_embedding_tt(
    timesteps, embedding_dim, device, flip_sin_to_cos=False, downscale_freq_shift=1, scale=1, max_period=10000
):
    half_dim = embedding_dim // 2
    val1 = -math.log(max_period)
    val2 = ttnn.arange(start=0, end=half_dim, dtype=ttnn.float32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    val2 = ttnn.to_layout(val2, layout=ttnn.TILE_LAYOUT)
    expon = ttnn.multiply(val2, val1)
    exponent = ttnn.div(expon, (half_dim - downscale_freq_shift), round_mode=None)
    emb = ttnn.exp(exponent)
    emb = ttnn.squeeze(emb, dim=0)
    emb = ttnn.squeeze(emb, dim=0)
    timesteps_p = ttnn.permute(timesteps, (1, 0))
    emb = ttnn.matmul(timesteps_p, emb, memory_config=ttnn.L1_MEMORY_CONFIG)
    emb = ttnn.multiply(emb, scale)
    emb_sin = ttnn.sin(emb)
    emb_cos = ttnn.cos(emb)
    emb = ttnn.concat([emb_sin, emb_cos], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
    if flip_sin_to_cos:
        emb = ttnn.concat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

    return emb


class ttnn_Timesteps:
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()

        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timestamps, device):
        t_emb = get_timestep_embedding_tt(
            timestamps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
            device=device,
        )
        return t_emb
