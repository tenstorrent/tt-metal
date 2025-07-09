# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


class AdaLayerNormDummy(torch.nn.Module):
    def __init__(self, embedding_dim: int, inner_dim: int) -> None:
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(embedding_dim, inner_dim)
        self.norm = torch.nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/normalization.py
class RmsNorm(torch.nn.Module):
    def __init__(self, *, dim: int, eps: float) -> None:
        super().__init__()

        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(self.weight.dtype) * self.weight
