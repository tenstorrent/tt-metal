# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Constant tiles for the fused residual-add + RMSNorm op (see op.py).

gamma is delivered as row-replicated 32x32 tiles over the full width (so a plain
``mul_tiles`` applies it), the reduce scaler is a tile filled with ``1/W`` and eps a
tile filled with the norm epsilon. Built once per norm; ~170 KB for W=2560.
"""
import torch

import ttnn

TILE = 32


def make_add_norm_constants(gamma: torch.Tensor, eps: float, device, memory_config=None):
    """gamma: ``[W]`` torch tensor. Returns (gamma_tiles ``[1,1,32,W]`` bf16, scaler, eps) device tensors."""
    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    width = int(gamma.numel())
    assert width % TILE == 0
    rep = gamma.to(torch.float32).reshape(1, 1, 1, width).expand(1, 1, TILE, width).contiguous()
    gamma_tiles = ttnn.from_torch(
        rep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )

    def const(val):
        t = torch.full((1, 1, TILE, TILE), float(val), dtype=torch.float32)
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )

    return gamma_tiles, const(1.0 / width), const(eps)
