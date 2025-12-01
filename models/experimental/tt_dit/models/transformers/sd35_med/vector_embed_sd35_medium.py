# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn
import torch
from models.experimental.tt_dit.layers.linear import Linear


class VectorEmbedder(nn.Module):
    """TTNN implementation of VectorEmbedder"""

    def __init__(self, input_dim: int, hidden_size: int, mesh_device=None, dtype=torch.bfloat16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.mesh_device = mesh_device

        self.linear1 = Linear(
            in_features=input_dim,
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

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Ensure input is 3D for TTNN matmul: [B, 1, F]
        if len(x.shape) == 2:
            x = ttnn.unsqueeze(x, 1)

        # Linear → SiLU → Linear
        x = self.linear1(x)
        x = ttnn.silu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = self.linear2(x)

        return x
