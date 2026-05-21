# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Extended nightly validation for ttnn.outer

Coverage
- All batched and unbatched torch.outer variants (1D, batched same-rank,
  broadcast-on-batch, rank-mismatch, degenerate)
- All float and block-float dtypes (bfloat16, float32, bfloat8_b;
  bfloat4_b is intentionally not supported -- see test_outer_bfloat4b_unsupported)
- All input tensor spec combinations (layouts x memory configs x sharded)
- Output tensor spec variants (DRAM/L1, default, sharded)

The op's contract: a:[..., N], b:[..., M] -> [..., N, M] via
a.unsqueeze(-1) * b.unsqueeze(-2).

"""

from functools import partial
from itertools import product

import pytest
import torch
import ttnn

from models.common.utility_functions import torch_random
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device


# ---------------------------------------------------------------------------
# Shape catalog
# ---------------------------------------------------------------------------

SHAPES_1D_ALIGNED = [
    ([32], [32]),
    ([64], [64]),
    ([256], [256]),
    ([2048], [64]),  # LLaMA rotary
    ([4096], [128]),  # LLaMA-2 rotary
    ([64], [2048]),  # asymmetric
]

SHAPES_1D_UNALIGNED = [
    ([33], [33]),
    ([63], [129]),
    ([1023], [127]),
]

SHAPES_BATCHED_SAME_RANK = [
    ([2, 32], [2, 32]),
    ([4, 64], [4, 64]),
    ([2, 3, 32], [2, 3, 32]),
    ([4, 8, 32], [4, 8, 32]),
    ([2, 3, 4, 32], [2, 3, 4, 32]),  # rank-4 inputs -> rank-5 output
]

SHAPES_BROADCAST_BATCH = [
    ([2, 32], [1, 32]),
    ([1, 32], [2, 32]),
    ([1, 8, 32], [4, 8, 32]),
    ([4, 1, 32], [4, 8, 32]),
    ([2, 3, 32], [1, 3, 32]),
]

SHAPES_RANK_MISMATCH = [
    ([2, 3, 32], [32]),
    ([32], [2, 3, 32]),
    ([2, 3, 4, 32], [32]),
]

SHAPES_DEGENERATE = [
    ([1], [1]),
    ([1], [256]),
    ([256], [1]),
    ([2, 1, 32], [2, 1, 32]),
]

ALL_SHAPES = (
    SHAPES_1D_ALIGNED
    + SHAPES_1D_UNALIGNED
    + SHAPES_BATCHED_SAME_RANK
    + SHAPES_BROADCAST_BATCH
    + SHAPES_RANK_MISMATCH
    + SHAPES_DEGENERATE
)

# Subset for cross-product axes
SHAPES_MEMCFG_SUBSET = [
    ([32], [32]),
    ([2048], [64]),
    ([2, 3, 32], [2, 3, 32]),
    ([1, 8, 32], [4, 8, 32]),
    ([2, 3, 32], [32]),
]

SHAPES_ROW_MAJOR = [
    ([32], [32]),
    ([2, 32], [2, 32]),
    ([4, 8, 32], [4, 8, 32]),
    ([1, 8, 32], [4, 8, 32]),
    ([30], [30]),  # non-tile-aligned RM
]

SHAPES_TILE_ALIGNED = [
    ([32], [32]),
    ([64], [64]),
    ([256], [256]),
    ([2, 32], [2, 32]),
    ([4, 8, 32], [4, 8, 32]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DTYPES = [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32]


def _pcc(dtype):
    return 0.98 if dtype == ttnn.bfloat8_b else 0.9999


def _gen(shape, dtype, low=-100, high=100):
    return gen_func_with_cast_tt(partial(torch_random, low=low, high=high, dtype=torch.float32), dtype)(shape)


def _golden(a_pt, b_pt):
    return a_pt.unsqueeze(-1) * b_pt.unsqueeze(-2)


def _check(a_pt, b_pt, out_tt, dtype):
    out_pt = ttnn.to_torch(out_tt)
    golden = _golden(a_pt, b_pt)
    assert tuple(out_pt.shape) == tuple(golden.shape), f"shape mismatch: tt={out_pt.shape}, golden={golden.shape}"
    assert_with_pcc(golden, out_pt, pcc=_pcc(dtype))


# ---------------------------------------------------------------------------
# Test 1: shape x dtype matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a_shape, b_shape", ALL_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_outer_shape_dtype_matrix(a_shape, b_shape, dtype, device):
    torch.manual_seed(0)
    a_pt = _gen(a_shape, dtype)
    b_pt = _gen(b_shape, dtype)
    a_tt = ttnn.from_torch(a_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    out_tt = ttnn.outer(a_tt, b_tt)
    _check(a_pt, b_pt, out_tt, dtype)


# ---------------------------------------------------------------------------
# Test 2: input/output memory config matrix (TILE, interleaved)
# ---------------------------------------------------------------------------

INTERLEAVED_MEMCFG_PAIRS = list(
    product(
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],  # mem_a
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],  # mem_b
        [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],  # output
    )
)


@pytest.mark.parametrize("a_shape, b_shape", SHAPES_MEMCFG_SUBSET)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("mem_a, mem_b, out_mem", INTERLEAVED_MEMCFG_PAIRS)
def test_outer_interleaved_memcfg_matrix(a_shape, b_shape, dtype, mem_a, mem_b, out_mem, device):
    torch.manual_seed(0)
    a_pt = _gen(a_shape, dtype)
    b_pt = _gen(b_shape, dtype)
    a_tt = ttnn.from_torch(a_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_a)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_b)
    out_tt = ttnn.outer(a_tt, b_tt, memory_config=out_mem)
    _check(a_pt, b_pt, out_tt, dtype)


# ---------------------------------------------------------------------------
# Test 3: ROW_MAJOR input layout (bf16/fp32 only; block-float requires TILE)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a_shape, b_shape", SHAPES_ROW_MAJOR)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("out_mem", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_outer_row_major_inputs(a_shape, b_shape, dtype, out_mem, device):
    torch.manual_seed(0)
    a_pt = _gen(a_shape, dtype)
    b_pt = _gen(b_shape, dtype)
    a_tt = ttnn.from_torch(a_pt, dtype=dtype, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    out_tt = ttnn.outer(a_tt, b_tt, memory_config=out_mem)
    _check(a_pt, b_pt, out_tt, dtype)


# ---------------------------------------------------------------------------
# Test 4: mixed input layouts (one TILE, one ROW_MAJOR)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a_shape, b_shape", SHAPES_ROW_MAJOR[:3])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "a_layout, b_layout",
    [
        (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
    ],
)
def test_outer_mixed_layouts(a_shape, b_shape, dtype, a_layout, b_layout, device):
    torch.manual_seed(0)
    a_pt = _gen(a_shape, dtype)
    b_pt = _gen(b_shape, dtype)
    a_tt = ttnn.from_torch(a_pt, dtype=dtype, device=device, layout=a_layout)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype, device=device, layout=b_layout)
    out_tt = ttnn.outer(a_tt, b_tt)
    _check(a_pt, b_pt, out_tt, dtype)


# ---------------------------------------------------------------------------
# Test 5: sharded inputs
# ---------------------------------------------------------------------------


def _h_shard_2d_256_32():
    """Height-sharded [256, 32]: 8 height tiles split across 8 cores."""
    return ttnn.create_sharded_memory_config(
        [32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 1))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _w_shard_2d_32_256():
    """Width-sharded [32, 256]: 8 width tiles split across 8 cores."""
    return ttnn.create_sharded_memory_config(
        [32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 1))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _b_shard_2d_256_64():
    """Block-sharded [256, 64]: [128, 32] blocks across a 2x2 grid."""
    return ttnn.create_sharded_memory_config(
        [128, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 1))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# Each case ties a specific shape to a shard config that fits it. The sharded
# input is whichever side is non-"interleaved"; the other input uses
# interleaved DRAM. This validates that ttnn.outer's sharded_to_interleaved
# fallback handles each shard strategy correctly.
SHARDED_INPUT_CASES = [
    # (a_shape, b_shape, a_shard_fn, b_shard_fn)  (None means interleaved)
    ([256, 32], [256, 32], _h_shard_2d_256_32, None),
    ([256, 32], [256, 32], None, _h_shard_2d_256_32),
    ([256, 32], [256, 32], _h_shard_2d_256_32, _h_shard_2d_256_32),
    ([32, 256], [32, 256], _w_shard_2d_32_256, None),  # falls back via sharded_to_interleaved
    ([256, 64], [256, 64], _b_shard_2d_256_64, None),  # falls back
]


@pytest.mark.parametrize("a_shape, b_shape, a_shard_fn, b_shard_fn", SHARDED_INPUT_CASES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_outer_sharded_inputs(a_shape, b_shape, a_shard_fn, b_shard_fn, dtype, device):
    """Sharded inputs must produce numerically correct output. ttnn::outer's
    sharded_to_interleaved fallback handles shard strategies that the
    unsqueeze (reshape) layer can't natively transform."""
    torch.manual_seed(0)
    a_mem = a_shard_fn() if a_shard_fn else ttnn.DRAM_MEMORY_CONFIG
    b_mem = b_shard_fn() if b_shard_fn else ttnn.DRAM_MEMORY_CONFIG
    a_pt = _gen(a_shape, dtype)
    b_pt = _gen(b_shape, dtype)
    a_tt = ttnn.from_torch(a_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=a_mem)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=b_mem)
    out_tt = ttnn.outer(a_tt, b_tt)
    _check(a_pt, b_pt, out_tt, dtype)


# ---------------------------------------------------------------------------
# Test 6: sharded output config (interleaved inputs, sharded output)
# ---------------------------------------------------------------------------


def _h_shard_for_outer_output(out_height_tiles):
    """Build a sharded memory config that fits the [N, M] output of an outer
    product. The output's height is out_height_tiles * 32."""
    return ttnn.create_sharded_memory_config(
        [32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (out_height_tiles - 1, 0))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize(
    "a_shape, b_shape, out_height_tiles",
    [
        ([128], [32], 4),  # output [128, 32] -> 4 tiles of [32, 32]
        ([256], [32], 8),  # output [256, 32] -> 8 tiles
    ],
)
def test_outer_sharded_output(a_shape, b_shape, out_height_tiles, device):
    torch.manual_seed(0)
    dtype = ttnn.bfloat16
    out_mem = _h_shard_for_outer_output(out_height_tiles)
    a_pt = _gen(a_shape, dtype)
    b_pt = _gen(b_shape, dtype)
    a_tt = ttnn.from_torch(a_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    out_tt = ttnn.outer(a_tt, b_tt, memory_config=out_mem)
    _check(a_pt, b_pt, out_tt, dtype)


# ---------------------------------------------------------------------------
# Test 7: default memory_config (no kwarg)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a_shape, b_shape", SHAPES_MEMCFG_SUBSET)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_outer_default_memory_config(a_shape, b_shape, dtype, device):
    torch.manual_seed(0)
    a_pt = _gen(a_shape, dtype)
    b_pt = _gen(b_shape, dtype)
    a_tt = ttnn.from_torch(a_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    out_tt = ttnn.outer(a_tt, b_tt)
    _check(a_pt, b_pt, out_tt, dtype)


# ---------------------------------------------------------------------------
# Test 8: numerical stress (wide value ranges)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ([256], [256]),
        ([4, 8, 32], [4, 8, 32]),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "low, high",
    [
        (-1000, 1000),  # wide
        (-0.01, 0.01),  # narrow / small magnitude
        (0, 1000),  # positive only
        (-1000, 0),  # negative only
    ],
)
def test_outer_value_range_stress(a_shape, b_shape, dtype, low, high, device):
    torch.manual_seed(0)
    a_pt = _gen(a_shape, dtype, low=low, high=high)
    b_pt = _gen(b_shape, dtype, low=low, high=high)
    a_tt = ttnn.from_torch(a_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    out_tt = ttnn.outer(a_tt, b_tt)
    _check(a_pt, b_pt, out_tt, dtype)


# ---------------------------------------------------------------------------
# Test 9: bfloat4_b is intentionally not supported
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="ttnn.unsqueeze routes through reshape_device_operation which only accepts "
    "BFLOAT16/UINT32/FLOAT32/INT32. Tracked in "
    "https://github.com/tenstorrent/tt-metal/issues/44919",
)
def test_outer_bfloat4b_unsupported(device):
    """ttnn.unsqueeze routes through reshape_device_operation which only
    accepts BFLOAT16/UINT32/FLOAT32/INT32. bfloat4_b inputs to ttnn.outer
    must therefore fail with a clear TT_FATAL at the reshape layer, not
    crash or produce silently wrong results. This test pins that behavior
    so any future relaxation in reshape is flagged."""
    a_pt = torch.rand([32], dtype=torch.bfloat16)
    b_pt = torch.rand([32], dtype=torch.bfloat16)
    a_tt = ttnn.from_torch(a_pt, dtype=ttnn.bfloat4_b, device=device, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b_pt, dtype=ttnn.bfloat4_b, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn.outer(a_tt, b_tt)
