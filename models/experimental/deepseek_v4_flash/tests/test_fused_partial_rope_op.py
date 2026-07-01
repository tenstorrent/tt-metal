# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit test for ``ttnn.experimental.fused_partial_rope``.

The op fuses the deepseek_v4_flash ``_apply_rope`` calc (attention.py lines 151-165) into one
Blackhole device op on height-sharded L1 tensors: interleaved RoPE on the trailing ``rope_dim``
channels via a ``rotate_half`` matmul, with the leading "nope" channels passed through untouched.

This test is self-contained: it builds a random input + cos/sin tables + rotate matrix, runs the
op on device, and compares against a torch reference of exactly that math.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc


PCC_THRESHOLD = 0.999
TILE = 32


def _interleaved_rotate_matrix(rope_dim: int) -> torch.Tensor:
    """[rope_dim, rope_dim] interleaved ``rotate_half`` matrix (matches attention.py)."""
    r = torch.zeros(rope_dim, rope_dim, dtype=torch.float32)
    for p in range(rope_dim // 2):
        r[2 * p, 2 * p + 1] = 1.0
        r[2 * p + 1, 2 * p] = -1.0
    return r


def _torch_reference(x, cos, sin, rot, rope_dim):
    """Mirror of ``_apply_rope`` in torch float32."""
    d = x.shape[-1]
    if d == rope_dim:
        nope, rope = None, x
    else:
        nope = x[..., : d - rope_dim]
        rope = x[..., d - rope_dim :]
    rotated = rope * cos + (rope @ rot) * sin
    if nope is None:
        return rotated
    return torch.cat([nope, rotated], dim=-1)


def _height_sharded_cfg(width: int, num_cores: int) -> ttnn.MemoryConfig:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard = ttnn.ShardSpec(grid, [TILE, width], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)


# (head_dim D, rope_dim Rd, rows)
@pytest.mark.parametrize(
    "D, Rd, rows",
    (
        (512, 64, 64),  # deepseek q shape: H=64 -> 2 cores
        (512, 64, 32),  # kv/compressor single tile-row -> 1 core
        (512, 64, 96),  # 3 tile-rows -> 3 cores
        (64, 64, 32),  # D == Rd edge case (no nope)
    ),
)
def test_fused_partial_rope_op(device, reset_seeds, D, Rd, rows):
    num_cores = rows // TILE

    x = torch.randn(1, 1, rows, D, dtype=torch.float32)
    cos = torch.randn(1, 1, rows, Rd, dtype=torch.float32)
    sin = torch.randn(1, 1, rows, Rd, dtype=torch.float32)
    rot = _interleaved_rotate_matrix(Rd)

    ref = _torch_reference(x, cos, sin, rot, Rd)

    # X is height-sharded L1 (one tile-row per core); cos/sin/trans_mat are DRAM-interleaved.
    x_tt = ttnn.to_memory_config(
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        _height_sharded_cfg(D, num_cores),
    )
    cos_tt = ttnn.from_torch(
        cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    sin_tt = ttnn.from_torch(
        sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    # trans_mat is a single [32, 32] rotate_half tile (block of the full [Rd, Rd] matrix),
    # applied per rope tile by the kernel matmul. DRAM-interleaved (replicated) device tensor.
    trans_mat = _interleaved_rotate_matrix(TILE).reshape(1, 1, TILE, TILE)
    trans_mat_tt = ttnn.from_torch(
        trans_mat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out_tt = ttnn.experimental.fused_partial_rope(x_tt, cos_tt, sin_tt, trans_mat_tt, Rd)
    got = ttnn.to_torch(out_tt).reshape(ref.shape).float()

    passing, pcc_message = comp_pcc(ref, got, pcc=PCC_THRESHOLD)
    logger.info(f"[fused_partial_rope D={D} Rd={Rd} rows={rows}] {comp_allclose(ref, got)}")
    logger.info(f"[fused_partial_rope D={D} Rd={Rd} rows={rows}] PCC: {pcc_message}")
    assert passing, f"fused_partial_rope PCC < {PCC_THRESHOLD} (D={D}, Rd={Rd}, rows={rows}): {pcc_message}"
