# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

import ttnn


class ApplyRotaryPosEmb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x, cos, sin):
        # Broadcast cos/sin across heads (and tokens implicitly)
        num_heads = x.shape[-2]
        # GLM applies rotary only on the first half of head_dim; our math uses a,b that are half of that.
        # Ensure cos/sin width matches a,b width BEFORE broadcasting to avoid slicing a height-sharded tensor.
        half_rot = (x.shape[-1] // 2) // 2
        # If cos/sin come height-sharded (e.g., from RotarySetup for decode), slicing width
        # directly will fail due to shard width constraints. Convert to DRAM first when needed.
        cos_in = cos
        sin_in = sin
        if cos_in.shape[-1] != half_rot:
            cos_in = ttnn.to_memory_config(cos_in, ttnn.DRAM_MEMORY_CONFIG)
            sin_in = ttnn.to_memory_config(sin_in, ttnn.DRAM_MEMORY_CONFIG)
            cos_in = ttnn.slice(cos_in, (0, 0, 0, 0), (cos_in.shape[0], cos_in.shape[1], cos_in.shape[2], half_rot))
            sin_in = ttnn.slice(sin_in, (0, 0, 0, 0), (sin_in.shape[0], sin_in.shape[1], sin_in.shape[2], half_rot))

        cos_bcast = ttnn.repeat(cos_in, (1, 1, num_heads, 1))
        sin_bcast = ttnn.repeat(sin_in, (1, 1, num_heads, 1))

        # GLM uses partial rotary (e.g., partial_rotary_factor=0.5).
        # Rotate only the first half of the head dimension; keep the second half unchanged.
        x_rot, x_pass = ttnn.chunk(x, 2, -1)  # split head_dim into [rotary_dim, remain]
        a, b = ttnn.chunk(x_rot, 2, -1)  # split rotary_dim into halves
        a_rot = a * cos_bcast - b * sin_bcast
        b_rot = b * cos_bcast + a * sin_bcast
        x_rotated = ttnn.concat((a_rot, b_rot), dim=-1)
        return ttnn.concat((x_rotated, x_pass), dim=-1)
