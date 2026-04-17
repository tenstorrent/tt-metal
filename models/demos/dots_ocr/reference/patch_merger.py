# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


class PatchMergerRef(nn.Module):
    """
    Reference patch merger:
      - RMSNorm over hidden_size
      - reshape merge spatial_merge_size^2 into channel dim
      - Linear -> GELU -> Linear to out_hidden_size
    """

    def __init__(self, *, hidden_size: int, out_hidden_size: int, spatial_merge_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_hidden_size = out_hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.mlp_size = hidden_size * (spatial_merge_size**2)

        self.norm = RMSNorm(hidden_size, eps=eps)
        self.fc1 = nn.Linear(self.mlp_size, self.mlp_size, bias=False)
        self.fc2 = nn.Linear(self.mlp_size, out_hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, S_patch, H]
        x = self.norm(x)
        B, one, S_patch, H = x.shape
        assert H == self.hidden_size
        assert S_patch % (self.spatial_merge_size**2) == 0
        S_img = S_patch // (self.spatial_merge_size**2)
        x = x.reshape(B, one, S_img, self.mlp_size)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x
