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
        cos_bcast = ttnn.repeat(cos, (1, 1, num_heads, 1))
        sin_bcast = ttnn.repeat(sin, (1, 1, num_heads, 1))
        # GLM applies rotary only on the first half of head_dim; our math uses a,b that are half of that.
        # Ensure cos/sin width matches a,b width to avoid invalid subtile broadcasts in binary_ng.
        half_rot = (x.shape[-1] // 2) // 2
        if cos_bcast.shape[-1] != half_rot:
            cos_bcast = ttnn.slice(
                cos_bcast,
                (0, 0, 0, 0),
                (cos_bcast.shape[0], cos_bcast.shape[1], cos_bcast.shape[2], half_rot),
            )
            sin_bcast = ttnn.slice(
                sin_bcast,
                (0, 0, 0, 0),
                (sin_bcast.shape[0], sin_bcast.shape[1], sin_bcast.shape[2], half_rot),
            )

        # GLM uses partial rotary (e.g., partial_rotary_factor=0.5).
        # Rotate only the first half of the head dimension; keep the second half unchanged.
        x_rot, x_pass = ttnn.chunk(x, 2, -1)  # split head_dim into [rotary_dim, remain]
        a, b = ttnn.chunk(x_rot, 2, -1)  # split rotary_dim into halves
        a_rot = a * cos_bcast - b * sin_bcast
        b_rot = b * cos_bcast + a * sin_bcast
        x_rotated = ttnn.concat((a_rot, b_rot), dim=-1)
        return ttnn.concat((x_rotated, x_pass), dim=-1)
