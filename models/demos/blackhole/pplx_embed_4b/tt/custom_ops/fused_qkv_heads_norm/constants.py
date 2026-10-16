# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-prepared constant tiles for the fused head-split + RMSNorm op.

The compute kernel avoids in-kernel scalar generation: gamma is delivered as
row-replicated 32x32 tiles (so a plain elementwise ``mul_tiles`` applies it),
the reduce scaler is a tile filled with ``1/head_dim`` (turning the row-sum of
squares into the mean-square) and eps is a tile filled with the norm epsilon.
Built once per layer at attention construction; a few KB per layer.
"""
import torch

import ttnn

TILE = 32


def make_norm_constants(gamma_q: torch.Tensor, gamma_k: torch.Tensor, eps: float, device, memory_config=None):
    """gamma_*: ``[head_dim]`` torch tensors. Returns (gamma_q_tiles, gamma_k_tiles, scaler, eps) device tensors."""
    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    head_dim = int(gamma_q.numel())
    assert head_dim % TILE == 0 and gamma_k.numel() == head_dim

    def tiles(v):
        rep = v.to(torch.float32).reshape(1, 1, 1, head_dim).expand(1, 1, TILE, head_dim).contiguous()
        return ttnn.from_torch(
            rep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )

    def const(val):
        t = torch.full((1, 1, TILE, TILE), float(val), dtype=torch.float32)
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )

    return tiles(gamma_q), tiles(gamma_k), const(1.0 / head_dim), const(eps)
