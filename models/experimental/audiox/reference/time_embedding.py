# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, out_features: int, std: float = 1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn(out_features // 2, in_features) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = 2 * math.pi * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class TimestepEmbedding(nn.Module):
    """Timestep -> Fourier features -> Linear -> SiLU -> Linear."""

    def __init__(self, embed_dim: int, fourier_dim: int = 256):
        super().__init__()
        self.fourier = FourierFeatures(1, fourier_dim)
        self.linear1 = nn.Linear(fourier_dim, embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t expected as [B] or [B, 1]; ensure trailing feature dim of 1.
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.linear2(self.act(self.linear1(self.fourier(t))))
