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


def _torch_reference(x, cos, sin, rot, rope_dim, head_dim=None):
    """Mirror of ``_apply_rope`` in torch float32, including packed ``head_dim`` blocks."""
    d = x.shape[-1]
    hd = d if head_dim is None else head_dim
    blocks = []
    for h in range(d // hd):
        block = x[..., h * hd : (h + 1) * hd]
        if hd == rope_dim:
            rotated = block * cos + (block @ rot) * sin
            blocks.append(rotated)
        else:
            nope = block[..., : hd - rope_dim]
            rope = block[..., hd - rope_dim :]
            rotated = rope * cos + (rope @ rot) * sin
            blocks.append(torch.cat([nope, rotated], dim=-1))
    return torch.cat(blocks, dim=-1)


def _height_sharded_cfg(width: int, num_cores: int, shard_height: int = TILE) -> ttnn.MemoryConfig:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard = ttnn.ShardSpec(grid, [shard_height, width], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)


def _width_sharded_cfg(rows: int, width: int, num_cores: int) -> ttnn.MemoryConfig:
    """Width-sharded L1: every core holds all ``rows`` but a ``width // num_cores`` column slice."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard = ttnn.ShardSpec(grid, [rows, width // num_cores], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard)


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


# (head_dim D, rope_dim Rd, rows, num_cores, cos broadcast). The core count picks how the D
# columns split against the nope/rope boundary at D - Rd:
#   512/64 over 8 cores -> 2 tiles per core, boundary at tile 14: cores 0-6 nope-only, core 7 rope-only
#   512/64 over 4 cores -> 4 tiles per core, boundary inside core 3's slice (straddle)
#   D == Rd             -> every core is rope-only
@pytest.mark.parametrize(
    "D, Rd, rows, num_cores, cos_bcast",
    (
        (512, 64, 64, 8, False),  # aligned split: nope-only + rope-only cores
        (512, 64, 64, 4, False),  # boundary straddles a shard
        (512, 128, 96, 8, False),  # 3 row-tiles per core, wider rope region
        (64, 64, 32, 2, False),  # D == Rd edge case (no nope)
        (512, 64, 64, 4, True),  # single cos/sin row broadcast over all row-tiles
    ),
)
def test_fused_partial_rope_op_width_sharded(device, reset_seeds, D, Rd, rows, num_cores, cos_bcast):
    cos_rows = 1 if cos_bcast else rows

    x = torch.randn(1, 1, rows, D, dtype=torch.float32)
    cos = torch.randn(1, 1, cos_rows, Rd, dtype=torch.float32)
    sin = torch.randn(1, 1, cos_rows, Rd, dtype=torch.float32)
    rot = _interleaved_rotate_matrix(Rd)

    ref = _torch_reference(x, cos, sin, rot, Rd)

    # X is width-sharded L1 (a column slice per core); cos/sin/trans_mat are DRAM-interleaved.
    x_tt = ttnn.to_memory_config(
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        _width_sharded_cfg(rows, D, num_cores),
    )
    cos_tt = ttnn.from_torch(
        cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    sin_tt = ttnn.from_torch(
        sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    trans_mat = _interleaved_rotate_matrix(TILE).reshape(1, 1, TILE, TILE)
    trans_mat_tt = ttnn.from_torch(
        trans_mat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out_tt = ttnn.experimental.fused_partial_rope(x_tt, cos_tt, sin_tt, trans_mat_tt, Rd)
    got = ttnn.to_torch(out_tt).reshape(ref.shape).float()

    tag = f"D={D} Rd={Rd} rows={rows} cores={num_cores} bcast={cos_bcast}"
    passing, pcc_message = comp_pcc(ref, got, pcc=PCC_THRESHOLD)
    logger.info(f"[fused_partial_rope width-sharded {tag}] {comp_allclose(ref, got)}")
    logger.info(f"[fused_partial_rope width-sharded {tag}] PCC: {pcc_message}")
    assert passing, f"fused_partial_rope width-sharded PCC < {PCC_THRESHOLD} ({tag}): {pcc_message}"


def _trans_mat_tt(device):
    trans_mat = _interleaved_rotate_matrix(TILE).reshape(1, 1, TILE, TILE)
    return ttnn.from_torch(
        trans_mat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _cos_sin_tt(cos, sin, device):
    kwargs = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.from_torch(cos, **kwargs), ttnn.from_torch(sin, **kwargs)


# ROW_MAJOR X is consumed as 1x32 faces (one row of 32 BF16s per tile). cos/sin stay TILE and
# must be a single broadcast row (decode). Output layout matches X.
@pytest.mark.parametrize(
    "D, Rd, rows",
    (
        (512, 64, 1),  # single decode row
        (512, 64, 64),  # 64 heads, one row each as 1x32 faces
        (64, 64, 1),  # D == Rd, one row
    ),
)
def test_fused_partial_rope_op_row_major_height_sharded(device, reset_seeds, D, Rd, rows):
    x = torch.randn(1, 1, rows, D, dtype=torch.float32)
    cos = torch.randn(1, 1, 1, Rd, dtype=torch.float32)
    sin = torch.randn(1, 1, 1, Rd, dtype=torch.float32)
    rot = _interleaved_rotate_matrix(Rd)
    ref = _torch_reference(x, cos, sin, rot, Rd)

    x_tt = ttnn.to_memory_config(
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
        _height_sharded_cfg(D, num_cores=1, shard_height=rows),
    )
    cos_tt, sin_tt = _cos_sin_tt(cos, sin, device)
    out_tt = ttnn.experimental.fused_partial_rope(x_tt, cos_tt, sin_tt, _trans_mat_tt(device), Rd)

    assert out_tt.layout == ttnn.ROW_MAJOR_LAYOUT
    got = ttnn.to_torch(out_tt).reshape(ref.shape).float()
    passing, pcc_message = comp_pcc(ref, got, pcc=PCC_THRESHOLD)
    tag = f"D={D} Rd={Rd} rows={rows}"
    logger.info(f"[fused_partial_rope rm-hs {tag}] {comp_allclose(ref, got)}")
    logger.info(f"[fused_partial_rope rm-hs {tag}] PCC: {pcc_message}")
    assert passing, f"fused_partial_rope rm-hs PCC < {PCC_THRESHOLD} ({tag}): {pcc_message}"


@pytest.mark.parametrize(
    "D, Rd, rows, num_cores",
    (
        (512, 64, 1, 8),  # 1x32 faces, aligned nope/rope split
        (512, 64, 64, 4),  # 64 heads, boundary straddles a shard
        (64, 64, 1, 2),  # D == Rd
    ),
)
def test_fused_partial_rope_op_row_major_width_sharded(device, reset_seeds, D, Rd, rows, num_cores):
    x = torch.randn(1, 1, rows, D, dtype=torch.float32)
    cos = torch.randn(1, 1, 1, Rd, dtype=torch.float32)
    sin = torch.randn(1, 1, 1, Rd, dtype=torch.float32)
    rot = _interleaved_rotate_matrix(Rd)
    ref = _torch_reference(x, cos, sin, rot, Rd)

    x_tt = ttnn.to_memory_config(
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
        _width_sharded_cfg(rows, D, num_cores),
    )
    cos_tt, sin_tt = _cos_sin_tt(cos, sin, device)
    out_tt = ttnn.experimental.fused_partial_rope(x_tt, cos_tt, sin_tt, _trans_mat_tt(device), Rd)

    assert out_tt.layout == ttnn.ROW_MAJOR_LAYOUT
    got = ttnn.to_torch(out_tt).reshape(ref.shape).float()
    passing, pcc_message = comp_pcc(ref, got, pcc=PCC_THRESHOLD)
    tag = f"D={D} Rd={Rd} rows={rows} cores={num_cores}"
    logger.info(f"[fused_partial_rope rm-ws {tag}] {comp_allclose(ref, got)}")
    logger.info(f"[fused_partial_rope rm-ws {tag}] PCC: {pcc_message}")
    assert passing, f"fused_partial_rope rm-ws PCC < {PCC_THRESHOLD} ({tag}): {pcc_message}"


def _run_fused_partial_rope(device, x, cos, sin, rot, Rd, mem_cfg, layout, head_dim=None):
    kwargs = dict(dtype=ttnn.bfloat16, device=device)
    x_tt = ttnn.to_memory_config(ttnn.from_torch(x, layout=layout, **kwargs), mem_cfg)
    cos_tt, sin_tt = _cos_sin_tt(cos, sin, device)
    out_tt = ttnn.experimental.fused_partial_rope(
        x_tt, cos_tt, sin_tt, _trans_mat_tt(device), Rd, head_dim=0 if head_dim is None else head_dim
    )
    assert out_tt.layout == layout
    ref = _torch_reference(x, cos, sin, rot, Rd, head_dim=head_dim)
    got = ttnn.to_torch(out_tt).reshape(ref.shape).float()
    return ref, got


# Packed last dim: D = n_heads * head_dim (512), same cos/sin applied to every block.
@pytest.mark.parametrize(
    "head_dim, Rd, n_heads, rows, num_cores, layout, sharded",
    (
        (512, 64, 2, 32, 8, ttnn.TILE_LAYOUT, "width"),  # two packed heads, TILE width-sharded
        (512, 64, 2, 64, 4, ttnn.TILE_LAYOUT, "width"),  # shards straddle head boundaries
        (512, 64, 2, 32, 1, ttnn.TILE_LAYOUT, "height"),  # height-sharded, full D per core
        (512, 64, 2, 2, 8, ttnn.ROW_MAJOR_LAYOUT, "width"),  # RM 1x32 faces, packed heads
        (512, 64, 2, 2, 1, ttnn.ROW_MAJOR_LAYOUT, "height"),  # RM 1x32, full D on one core
    ),
)
def test_fused_partial_rope_packed_heads(device, reset_seeds, head_dim, Rd, n_heads, rows, num_cores, layout, sharded):
    D = n_heads * head_dim
    cos_rows = 1 if layout == ttnn.ROW_MAJOR_LAYOUT else rows
    x = torch.randn(1, 1, rows, D, dtype=torch.float32)
    cos = torch.randn(1, 1, cos_rows, Rd, dtype=torch.float32)
    sin = torch.randn(1, 1, cos_rows, Rd, dtype=torch.float32)
    rot = _interleaved_rotate_matrix(Rd)

    if sharded == "width":
        mem_cfg = _width_sharded_cfg(rows, D, num_cores)
    else:
        shard_h = rows if layout == ttnn.ROW_MAJOR_LAYOUT else TILE
        mem_cfg = _height_sharded_cfg(D, num_cores, shard_height=shard_h)

    ref, got = _run_fused_partial_rope(device, x, cos, sin, rot, Rd, mem_cfg, layout, head_dim=head_dim)
    passing, pcc_message = comp_pcc(ref, got, pcc=PCC_THRESHOLD)
    tag = f"Hd={head_dim} Rd={Rd} heads={n_heads} rows={rows} cores={num_cores} {layout} {sharded}"
    logger.info(f"[fused_partial_rope packed {tag}] {comp_allclose(ref, got)}")
    logger.info(f"[fused_partial_rope packed {tag}] PCC: {pcc_message}")
    assert passing, f"fused_partial_rope packed-heads PCC < {PCC_THRESHOLD} ({tag}): {pcc_message}"
