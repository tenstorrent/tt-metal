# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SD3.5 Medium TimeStepEmbedder Implementation

This module implements TimeStepEmbedder for
SD3.5 Medium, matching the reference implementation.
"""

import math
import torch
import ttnn
import torch.nn as nn

from ...layers.linear import Linear


class TimestepEmbedder(nn.Module):
    """
    TTNN implementation of timestep embedding
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, mesh_device=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mesh_device = mesh_device

        # Two Linear layers same as reference model
        self.linear1 = Linear(
            in_features=frequency_embedding_size,
            out_features=hidden_size,
            bias=True,
            mesh_device=mesh_device,
        )

        self.linear2 = Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
            mesh_device=mesh_device,
        )

    def forward(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """
        t: TTNN Tensor of shape [B] (bf16)
        """
        # Convert to torch for sinusoidal embedding
        t_torch = ttnn.to_torch(t)
        B = t_torch.shape[0]
        half = self.frequency_embedding_size // 2

        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half).to(t_torch.device)

        args = t_torch[:, None].float() * freqs[None]
        embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        embed = embed.to(torch.bfloat16)

        # Convert back into TTNN with TILE layout
        x = ttnn.from_torch(
            embed,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )

        # Ensure padded dims exist for matmul => [B, Freq] → [B, 1, Freq]
        if len(x.shape) == 2:
            x = ttnn.unsqueeze(x, 1)

        # Linear1 → SiLU → Linear2 (all TILE)
        x = self.linear1(x)
        x = ttnn.silu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = self.linear2(x)

        return x
