# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Functional test for the Metal 2.0 / DataflowBuffer (DFB) path of
`ttnn.experimental.quasar.unary_lut` — the UNARY analog of the binary-ADD DFB op.

The op applies an embedded piecewise-polynomial LUT activation (the proven deg-2 /
4-segment sigmoid approximation, NO range reduction) to a fully height/block-sharded
bf16 L1 input, through the DFB framework (single input DataflowBuffer + output DFB,
piecewise-LUT SFPU eval as the compute).

The golden REPLICATES the exact baked-in LUT (boundaries + per-segment Horner
coefficients) — the same values as the C++ kernel default and the tt-llk Quasar
test's DEFAULT_BOUNDARIES / DEFAULT_COEFFS. So the PCC isolates the DFB-path
correctness (the kernel produces the LUT eval, not zeros), not the fit accuracy of
the approximation.

Run on the Quasar simulator:
    TT_METAL_SIMULATOR=<path>/libttsim.so ARCH_NAME=quasar CHIP_ARCH=quasar \
        TT_METAL_SLOW_DISPATCH_MODE=1 \
        pytest tests/ttnn/unit_tests/operations/experimental/quasar/test_unary_lut_dfb.py
"""

import numpy as np
import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# The default LUT baked into the compute kernel (unary_lut_sfpu.h). MUST match the
# C++ kernel defaults and the tt-llk Quasar test exactly.
DEFAULT_NUM_SEGMENTS = 4
DEFAULT_POLY_DEGREE = 2
DEFAULT_BOUNDARIES = [-4.0, -2.0, 0.0, 2.0, 4.0]
DEFAULT_COEFFS = [
    [0.38296354, 0.17515847, 0.02109685],  # seg0: c0, c1, c2
    [0.50329190, 0.27505103, 0.04113654],  # seg1
    [0.49670810, 0.27505103, -0.04113654],  # seg2
    [0.61703646, 0.17515847, -0.02109685],  # seg3
]

_PCC = 0.99


def _lut_golden(x_np):
    """Replicate the kernel's piecewise-LUT eval: clamp to [b0, bN], select segment by
    boundary, clamp x to that segment's sub-interval, Horner-eval. fp32 throughout."""
    b = DEFAULT_BOUNDARIES
    x = np.clip(x_np.astype(np.float32), b[0], b[-1])
    out = np.empty_like(x)
    for seg in range(DEFAULT_NUM_SEGMENTS):
        lo, hi = b[seg], b[seg + 1]
        # The cumulative select assigns segment SEG to lanes with x >= b[seg]; the last
        # such segment wins. Equivalent to: pick seg s.t. b[seg] <= x (clamped),
        # highest seg.
        mask = x >= b[seg]
        xs = np.clip(x, lo, hi)
        c = DEFAULT_COEFFS[seg]
        acc = np.full_like(x, c[DEFAULT_POLY_DEGREE], dtype=np.float32)
        for k in range(DEFAULT_POLY_DEGREE - 1, -1, -1):
            acc = acc * xs + np.float32(c[k])
        out[mask] = acc[mask]
    return out


def _height_sharded_config(shard_shape, core_grid):
    return ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _block_sharded_config(shard_shape, core_grid):
    return ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _run_unary_lut(device, mem_config_fn, shard_shape, core_grid, total_shape):
    torch.manual_seed(0)
    # Inputs in [-5, 5] to exercise all 4 segments + the [b0, bN] clamp.
    x_pt = (torch.rand(total_shape, dtype=torch.float32) * 10.0 - 5.0).to(torch.bfloat16)

    mem_config = mem_config_fn(shard_shape, core_grid)
    x_tt = ttnn.from_torch(x_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_config)

    out_tt = ttnn.experimental.quasar.unary_lut(x_tt)

    golden = torch.from_numpy(_lut_golden(x_pt.to(torch.float32).numpy())).to(torch.bfloat16)
    assert_with_pcc(ttnn.to_torch(out_tt), golden, _PCC)
    return out_tt


# A 4-tall height-sharded column (y=0..3), 1 tile/shard. Fits WH (8x8) and Quasar (8x4).
_HEIGHT_GRID = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))})
_HEIGHT_SHARD = [32, 32]
_HEIGHT_SHAPE = torch.Size([4 * 32, 32])

# A block-sharded 4x2 grid: shard [32,32] over an 8-shard block layout. Fits 8x4.
_BLOCK_GRID = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 1))})
_BLOCK_SHARD = [32, 32]
_BLOCK_SHAPE = torch.Size([2 * 32, 4 * 32])


def test_unary_lut_height_sharded(device):
    _run_unary_lut(device, _height_sharded_config, _HEIGHT_SHARD, _HEIGHT_GRID, _HEIGHT_SHAPE)


def test_unary_lut_block_sharded(device):
    _run_unary_lut(device, _block_sharded_config, _BLOCK_SHARD, _BLOCK_GRID, _BLOCK_SHAPE)


def test_unary_lut_multitile_shard(device):
    # 4 tiles per core to exercise the compute chunk loop.
    grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3))})
    shard = [4 * 32, 32]
    shape = torch.Size([4 * 4 * 32, 32])
    _run_unary_lut(device, _height_sharded_config, shard, grid, shape)
