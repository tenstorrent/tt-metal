# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 — Multi-core row distribution + HEIGHT_SHARDED.

Two things land together, both the Row-axis knob-turn (each core owns a
contiguous span of tile-rows and reduces LOCALLY over W — no cross-core comms):

  * Interleaved multi-core: the tile-row range is spread over the grid via
    split_work_to_cores. Exercised implicitly by the multi-tile-row shapes here
    (and by test_rms_norm.py's multi-batch shapes) — total_tile_rows > 1 now
    dispatches to multiple cores.
  * HEIGHT_SHARDED: the SAME split with the row->core assignment pinned by the
    shard spec. The resident L1 shard is streamed through the SAME bounded
    scratch CBs via TensorAccessor (a local L1->L1 read); output inherits the
    input's shard spec.

Kernels are unchanged from Phase 0/R3 — they already key off per-core
(start_tile_row, num_tile_rows) RT args. This file guards the host-side work
distribution against regression. Sharded configs are built with the golden
`eval.sharding.auto_shard_config` (the same legal-shard synthesizer the golden
suite uses) so the unit + golden nets agree.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}
_TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def pytorch_rms_norm(x, gamma=None, epsilon=1e-6):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + epsilon)
    out = xf / rms
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out.to(x.dtype)


def _cfg():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


# tile-aligned (single/multi core), non-tile-aligned H and W, multi-image, 3D, 2D.
SHARDED_SHAPES = [
    (1, 1, 256, 512),  # 8 tile-rows -> 8 cores, full W=512 each (design loose case)
    (2, 4, 128, 512),  # multi-image, 32 tile-rows
    (1, 1, 64, 17),  # W non-aligned (masked reduce) + sharded
    (1, 1, 50, 128),  # H non-aligned (padding tile-rows) + sharded
    (4, 128, 512),  # 3D
    (1024, 1024),  # 2D, 32 tile-rows
]


@pytest.mark.parametrize("shape", SHARDED_SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("with_gamma", [False, True])
def test_rms_norm_height_sharded(device, shape, dtype, layout, with_gamma):
    torch.manual_seed(42)
    torch_dtype = _TORCH_DTYPE[dtype]
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch_dtype)
    mem_cfg = auto_shard_config(list(shape), HEIGHT, layout=layout, dtype=dtype, device=device)
    ttnn_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=mem_cfg)

    if with_gamma:
        torch_gamma = torch.randn(W, dtype=torch_dtype)
        ttnn_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=dtype,
            layout=layout,  # gamma format follows input layout here (both legs exercised)
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        torch_gamma = None
        ttnn_gamma = None

    expected = pytorch_rms_norm(torch_input, gamma=torch_gamma, epsilon=1e-6)

    # Output inherits the input's shard spec (the norm contract the golden
    # harness enforces for sharded input).
    ttnn_output = rms_norm(
        ttnn_input,
        gamma=ttnn_gamma,
        epsilon=1e-6,
        compute_kernel_config=_cfg(),
        memory_config=ttnn_input.memory_config(),
    )

    assert ttnn_output.layout == layout, "output layout must match input layout"
    assert ttnn_output.memory_config().memory_layout == HEIGHT, "output must stay HEIGHT_SHARDED"

    actual = ttnn.to_torch(ttnn_output).reshape(expected.shape)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), PCC[dtype])


# Refinement 5 — WIDTH_SHARDED + BLOCK_SHARDED cross-core reduction. The hidden W
# is split across a reduction GROUP of cores; each core reduces its LOCAL W-slice to
# a partial Σx²/W_global, the group root folds the partials + rsqrt-finalizes, and
# broadcasts 1/rms back via the mcast_pipe. Rectangular groups only (BLOCK = grid-row
# lines; WIDTH = the shard-grid bbox); TILE input, tile-aligned W.
WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED

# (shape, memory_layout): each is a clean rectangular group fit.
CROSS_CORE_SHAPES = [
    ((1, 1, 32, 2048), WIDTH),  # design loose case: 64-core group, 1 tile-row
    ((1, 1, 256, 512), BLOCK),  # design loose case: 8x8 grid, 8 line-groups
    ((4, 8, 32, 256), WIDTH),  # multirow: 8-core group x 32 tile-rows (multi-round)
    ((2, 4, 128, 512), BLOCK),  # multirow: per-core 4 tile-rows (multi-round)
    ((1, 1, 50, 512), WIDTH),  # H non-aligned (padding tile-rows dropped)
]


@pytest.mark.parametrize("shape,memory_layout", CROSS_CORE_SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("with_gamma", [False, True])
def test_rms_norm_cross_core_sharded(device, shape, memory_layout, dtype, with_gamma):
    """WIDTH/BLOCK cross-core reduce-root combine: output matches the reference and
    keeps the input's shard spec."""
    torch.manual_seed(0)
    tdt = _TORCH_DTYPE[dtype]
    torch_input = torch.randn(shape, dtype=tdt)
    gamma_t = torch.randn(shape[-1], dtype=tdt) if with_gamma else None
    mem_cfg = auto_shard_config(list(shape), memory_layout, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_cfg
    )
    ttnn_gamma = (
        ttnn.from_torch(gamma_t.reshape(1, 1, 1, shape[-1]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        if with_gamma
        else None
    )
    out = rms_norm(ttnn_input, gamma=ttnn_gamma, compute_kernel_config=_cfg(), memory_config=ttnn_input.memory_config())
    assert out.memory_config().memory_layout == memory_layout
    actual = ttnn.to_torch(out)
    expected = pytorch_rms_norm(torch_input, gamma=gamma_t)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), PCC[dtype])


# Refinement 5a / 5b — ROW_MAJOR + WIDTH/BLOCK_SHARDED cross-core reduction. RM width-sharding
# splits each logical row's W across cores at sub-tile (stick) granularity, so each core
# reads its OWN resident [Hs, Ws] shard from local L1, zero-pads the sub-tile W (and H) tail
# to whole tiles, tilizes, and the SAME R5 reduce-root gather + broadcast combine runs
# unchanged. R5b lifts the WIDTH w_non exclusion: auto_shard_config pads a non-tile-aligned W
# into ceil(W/w_gran) cores, which overflows into a RAGGED (non-rectangular) grid the mcast
# can't address, so the root broadcasts 1/rms by UNICAST to each member (rectangular WIDTH/BLOCK
# groups keep the mcast fast path). Supported: no_gamma + RM gamma, all alignments (TILE gamma
# still excluded — sub-tile gamma column offset, 5c).
RM_CROSS_CORE_SHAPES = [
    ((1, 1, 32, 64), WIDTH),  # sub-tile Ws=8, 1 tile-row
    ((2, 4, 128, 512), WIDTH),  # multi-tile-row (32), multi-image
    ((1, 1, 17, 64), WIDTH),  # h_non
    ((1, 1, 32, 4096), WIDTH),  # wide, tile-aligned Ws
    ((1, 1, 32, 50), WIDTH),  # R5b: w_non — bf16 rectangular (7 cores), fp32 ragged (13 in 8x2)
    ((2, 1, 128, 100), WIDTH),  # R5b: w_non ragged (bf16 13/8x2, fp32 25/8x4), multi-round (8)
    ((4, 8, 32, 47), WIDTH),  # R5b: w_non, multi-round (32 tile-rows) — fp32 ragged (12/8x2)
    ((1, 1, 32, 64), BLOCK),  # sub-tile H=4 AND W=8
    ((1, 1, 256, 512), BLOCK),  # design loose case
    ((1, 1, 64, 17), BLOCK),  # w_non (BLOCK grids stay rectangular)
    ((1024, 1024), BLOCK),  # 2D, tile-aligned per-core
]


@pytest.mark.parametrize("shape,memory_layout", RM_CROSS_CORE_SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("with_gamma", [False, True])
def test_rms_norm_row_major_cross_core(device, shape, memory_layout, dtype, with_gamma):
    """R5a: ROW_MAJOR WIDTH/BLOCK cross-core (RM gamma / no_gamma) matches the reference
    and keeps the input's shard spec."""
    torch.manual_seed(0)
    tdt = _TORCH_DTYPE[dtype]
    torch_input = torch.randn(shape, dtype=tdt)
    gamma_t = torch.randn(shape[-1], dtype=tdt) if with_gamma else None
    mem_cfg = auto_shard_config(list(shape), memory_layout, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=device)
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_cfg
    )
    ttnn_gamma = (
        ttnn.from_torch(gamma_t.reshape(1, 1, 1, shape[-1]), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        if with_gamma
        else None
    )
    out = rms_norm(ttnn_input, gamma=ttnn_gamma, compute_kernel_config=_cfg(), memory_config=ttnn_input.memory_config())
    assert out.memory_config().memory_layout == memory_layout
    actual = ttnn.to_torch(out).reshape(shape)
    expected = pytorch_rms_norm(torch_input, gamma=gamma_t)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), PCC[dtype])


# Refinement 5b — RM + WIDTH_SHARDED with non-tile-aligned W. auto_shard_config pads W into
# ceil(W/w_gran) cores; when that overflows a full grid row into a partial one the shard grid
# is RAGGED (ncores != nx*ny) and no single mcast rectangle addresses the whole reduction
# group. The root then broadcasts 1/rms by UNICAST to each member. These shapes force a genuine
# ragged grid (asserted below) so the unicast leg is actually exercised, not silently skipped.
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 50),  # fp32: 13 cores in 8x2 bbox (ragged), 1 tile-row
        (2, 1, 128, 100),  # fp32: 25 cores in 8x4 bbox (ragged), multi-round (8 tile-rows)
        (4, 8, 32, 47),  # fp32: 12 cores in 8x2 bbox (ragged), multi-round (32 tile-rows)
    ],
)
@pytest.mark.parametrize("with_gamma", [False, True])
def test_rms_norm_row_major_width_ragged_unicast(device, shape, with_gamma):
    """R5b: RM WIDTH with a RAGGED shard grid -> unicast broadcast-back. fp32 (w_gran=4) makes
    these ncores overflow into a partial row; assert the grid really is ragged, then verify
    the cross-core unicast reduction matches the reference and keeps the input's shard spec."""
    torch.manual_seed(0)
    dtype = ttnn.float32
    W = shape[-1]
    torch_input = torch.randn(shape, dtype=torch.float32)
    gamma_t = torch.randn(W, dtype=torch.float32) if with_gamma else None
    mem_cfg = auto_shard_config(list(shape), WIDTH, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=device)

    grid = mem_cfg.shard_spec.grid
    bb = grid.bounding_box()
    nx = int(bb.end.x) - int(bb.start.x) + 1
    ny = int(bb.end.y) - int(bb.start.y) + 1
    assert grid.num_cores() != nx * ny, f"expected a ragged grid, got {grid.num_cores()} in {nx}x{ny}"

    ttnn_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_cfg
    )
    ttnn_gamma = (
        ttnn.from_torch(gamma_t.reshape(1, 1, 1, W), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        if with_gamma
        else None
    )
    out = rms_norm(ttnn_input, gamma=ttnn_gamma, compute_kernel_config=_cfg(), memory_config=ttnn_input.memory_config())
    assert out.memory_config().memory_layout == WIDTH
    actual = ttnn.to_torch(out).reshape(shape)
    expected = pytorch_rms_norm(torch_input, gamma=gamma_t)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), PCC[dtype])


# Refinement 5c — RM cross-core + TILE gamma (fp32/bf16). Each RM cross-core core owns a
# sub-tile global W-column offset (w_col_start = i*Ws), so a TILE-stored gamma can't be read
# as whole tiles aligned to local col 0. The reader extracts the containing global gamma
# tile(s)' ROW-0 sub-columns (face-aware L1 byte copy) into cb_gamma_rm and reuses the
# RM-gamma compute tilize leg unchanged. bf8b TILE gamma stays excluded (5d — block-float
# sub-tile extraction needs an in-reader dequant).
R5C_TILE_GAMMA_SHAPES = [
    ((1, 1, 32, 64), WIDTH),  # sub-tile Ws=8
    ((1, 1, 32, 4096), WIDTH),  # tile-aligned Ws=64 (2 gamma tiles/core)
    ((1, 1, 32, 50), WIDTH),  # w_non (bf16 rectangular, fp32 ragged)
    ((1, 1, 256, 512), BLOCK),  # design loose case
    ((1, 1, 64, 17), BLOCK),  # w_non BLOCK
    ((1024, 1024), BLOCK),  # 2D, per_w=4 (W_BLOCK_TILES=4), multi-round
]


@pytest.mark.parametrize("shape,memory_layout", R5C_TILE_GAMMA_SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("gamma_dtype", [ttnn.bfloat16, ttnn.float32])
def test_rms_norm_row_major_cross_core_tile_gamma(device, shape, memory_layout, dtype, gamma_dtype):
    """R5c: RM WIDTH/BLOCK cross-core with a TILE-stored gamma (fp32/bf16, incl. mixed
    precision) matches the reference and keeps the input's shard spec."""
    torch.manual_seed(0)
    W = shape[-1]
    tdt = _TORCH_DTYPE[dtype]
    gdt = _TORCH_DTYPE[gamma_dtype]
    torch_input = torch.randn(shape, dtype=tdt)
    gamma_t = torch.randn(W, dtype=gdt)
    mem_cfg = auto_shard_config(list(shape), memory_layout, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=device)
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_cfg
    )
    ttnn_gamma = ttnn.from_torch(gamma_t.reshape(1, 1, 1, W), dtype=gamma_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = rms_norm(ttnn_input, gamma=ttnn_gamma, compute_kernel_config=_cfg(), memory_config=ttnn_input.memory_config())
    assert out.memory_config().memory_layout == memory_layout
    actual = ttnn.to_torch(out).reshape(shape)
    expected = pytorch_rms_norm(torch_input, gamma=gamma_t)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), PCC[dtype])


# Refinement 5d — RM cross-core + bf8b TILE gamma. The last 5c carve-out: a bf8b tile is
# block-float (16 elements share one 8-bit exponent), so the row-0 sub-column extraction R5c
# does for fp32/bf16 is not a byte copy — the reader DEQUANTS each row-0 datum (block-float
# decode) into the float cb_gamma_rm stick and reuses the SAME RM-gamma compute tilize leg.
# bf8b gamma is always TILE (bf8b + RM is INVALID), so this names exactly the deferred cells.
R5D_BF8B_GAMMA_SHAPES = [
    ((1, 1, 32, 64), WIDTH),  # sub-tile Ws=8
    ((1, 1, 32, 4096), WIDTH),  # tile-aligned Ws=64 (spans 2 gamma tiles/core)
    ((1, 1, 32, 50), WIDTH),  # w_non (bf16 rectangular, fp32 ragged)
    ((1, 1, 256, 512), BLOCK),  # design loose case
    ((1, 1, 64, 17), BLOCK),  # w_non BLOCK
    ((1024, 1024), BLOCK),  # 2D, per_w=4, multi-round
]


@pytest.mark.parametrize("shape,memory_layout", R5D_BF8B_GAMMA_SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_rms_norm_row_major_cross_core_bf8b_tile_gamma(device, shape, memory_layout, dtype):
    """R5d: RM WIDTH/BLOCK cross-core with a bf8b TILE-stored gamma (in-reader block-float
    dequant) matches the reference and keeps the input's shard spec. The input-dtype tolerance
    absorbs the bf8b gamma quantization (same contract as the R2 INTERLEAVED bf8b-gamma cells)."""
    torch.manual_seed(0)
    W = shape[-1]
    tdt = _TORCH_DTYPE[dtype]
    torch_input = torch.randn(shape, dtype=tdt)
    # bf8b gamma is TILE-only; reference uses the pre-quant bf16 gamma (tolerance absorbs it).
    gamma_t = torch.randn(W, dtype=torch.bfloat16)
    mem_cfg = auto_shard_config(list(shape), memory_layout, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=device)
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_cfg
    )
    ttnn_gamma = ttnn.from_torch(
        gamma_t.reshape(1, 1, 1, W), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    out = rms_norm(ttnn_input, gamma=ttnn_gamma, compute_kernel_config=_cfg(), memory_config=ttnn_input.memory_config())
    assert out.memory_config().memory_layout == memory_layout
    actual = ttnn.to_torch(out).reshape(shape)
    expected = pytorch_rms_norm(torch_input, gamma=gamma_t)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), PCC[dtype])
