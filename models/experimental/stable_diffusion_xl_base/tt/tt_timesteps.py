# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from math import log
import torch
import ttnn


class TtTimesteps:
    def __init__(self, device, num_channels, flip_sin_to_cos, downscale_freq_shift, scale):
        super().__init__()

        self.device = device
        self.num_channels = num_channels  # embedding_dim
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = 10000

        self.half_dim = self.num_channels // 2
        exponent = -log(self.max_period) * torch.arange(start=0, end=self.half_dim, dtype=torch.float32)
        exponent = exponent / (self.half_dim - downscale_freq_shift)

        # Setting emb to bfloat16 increases image quality
        self.emb = ttnn.from_torch(
            torch.exp(exponent),
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, timesteps):
        emb = ttnn.multiply(ttnn.unsqueeze(timesteps, -1), ttnn.unsqueeze(self.emb, 0))
        emb = ttnn.multiply(emb, self.scale)

        emb = ttnn.concat([ttnn.sin(emb), ttnn.cos(emb)], dim=-1)

        if self.flip_sin_to_cos:
            emb = ttnn.concat([emb[:, self.half_dim :], emb[:, : self.half_dim]], dim=-1)
        return emb
