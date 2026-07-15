# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Extended nightly validation for ttnn.outer

Coverage
- All batched and unbatched torch.outer variants (1D, batched same-rank,
  broadcast-on-batch, rank-mismatch, degenerate)
- All float and block-float dtypes (bfloat16, float32, bfloat8_b;
  bfloat4_b/uint16/uint8 are intentionally not supported -- see
  test_outer_unsupported_dtype_rejected)
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

# Sentinel written into the implicit tile-pad region of every TILE_LAYOUT input.
# If ttnn.outer ever reads pad slots into the result, the multiply produces
# sentinel * valid values far outside the input range and the PCC against
# torch.outer fails. Zero-padding (the default) would hide such bugs because
# 0 * valid = 0 within the pad region, which to_torch strips off.
ADVERSARIAL_PAD_VALUE = 1.0e6


def _pcc(dtype):
    return 0.98 if dtype == ttnn.bfloat8_b else 0.9999


def _gen(shape, dtype, low=-100, high=100):
    return gen_func_with_cast_tt(partial(torch_random, low=low, high=high, dtype=torch.float32), dtype)(shape)


def _to_device(tensor_pt, dtype, device, layout, memory_config=None):
    """ttnn.from_torch wrapper. For TILE_LAYOUT inputs in non-block-float
    dtypes (which carry implicit tile padding), poison the pad region with
    ADVERSARIAL_PAD_VALUE so any pad-region leakage into the result surfaces
    as a PCC failure. Block-float dtypes (bfloat8_b/bfloat4_b) are skipped
    because their per-tile shared exponent is pulled toward the sentinel,
    destroying precision of real in-tile values."""
    kwargs = {"dtype": dtype, "device": device, "layout": layout}
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    t = ttnn.from_torch(tensor_pt, **kwargs)
    block_float = {ttnn.bfloat8_b, ttnn.bfloat4_b}
    if layout == ttnn.TILE_LAYOUT and dtype not in block_float:
        t = ttnn.fill_implicit_tile_padding(t, ADVERSARIAL_PAD_VALUE)
    return t


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
    a_tt = _to_device(a_pt, dtype, device, ttnn.TILE_LAYOUT)
    b_tt = _to_device(b_pt, dtype, device, ttnn.TILE_LAYOUT)
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
    a_tt = _to_device(a_pt, dtype, device, ttnn.TILE_LAYOUT, memory_config=mem_a)
    b_tt = _to_device(b_pt, dtype, device, ttnn.TILE_LAYOUT, memory_config=mem_b)
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
    a_tt = _to_device(a_pt, dtype, device, a_layout)
    b_tt = _to_device(b_pt, dtype, device, b_layout)
    out_tt = ttnn.outer(a_tt, b_tt)
    _check(a_pt, b_pt, out_tt, dtype)


# ---------------------------------------------------------------------------
# Test 4b: batched inputs with non-tile-aligned last dim (padding focus)
# ---------------------------------------------------------------------------
# The default suite covers 1D unaligned (SHAPES_1D_UNALIGNED) and batched
# aligned, but never both at once. This test plugs that gap. All TILE_LAYOUT
# inputs in this file are routed through _to_device, which poisons the
# implicit tile-pad region with ADVERSARIAL_PAD_VALUE — so any pad-region
# leakage into the result shows up as a PCC failure here too.

SHAPES_PADDING_BATCHED = [
    ([3, 33], [3, 33]),  # 2D batched + unaligned last dim
    ([2, 65], [2, 17]),  # both inputs unaligned, asymmetric M/N
    ([5, 7, 33], [5, 7, 33]),  # 3D batched + unaligned
    ([1, 1, 1, 30], [1, 1, 1, 30]),  # rank-4 with small unaligned last dim
]


@pytest.mark.parametrize("a_shape, b_shape", SHAPES_PADDING_BATCHED)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_outer_padding_batched_unaligned(a_shape, b_shape, dtype, device):
    torch.manual_seed(0)
    a_pt = _gen(a_shape, dtype)
    b_pt = _gen(b_shape, dtype)
    a_tt = _to_device(a_pt, dtype, device, ttnn.TILE_LAYOUT)
    b_tt = _to_device(b_pt, dtype, device, ttnn.TILE_LAYOUT)
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
# interleaved DRAM. Height-sharded inputs stay sharded all the way through
# the op; width- and block-sharded inputs are materialized as interleaved
# DRAM internally (the unsqueeze touches the width dim).
SHARDED_INPUT_CASES = [
    # (a_shape, b_shape, a_shard_fn, b_shard_fn)  (None means interleaved)
    # Height-sharded (stays sharded through the op)
    ([256, 32], [256, 32], _h_shard_2d_256_32, None),
    ([256, 32], [256, 32], None, _h_shard_2d_256_32),
    ([256, 32], [256, 32], _h_shard_2d_256_32, _h_shard_2d_256_32),
    # Width-sharded (falls back via sharded_to_interleaved)
    ([32, 256], [32, 256], _w_shard_2d_32_256, None),
    ([32, 256], [32, 256], None, _w_shard_2d_32_256),
    ([32, 256], [32, 256], _w_shard_2d_32_256, _w_shard_2d_32_256),
    # Block-sharded (falls back via sharded_to_interleaved)
    ([256, 64], [256, 64], _b_shard_2d_256_64, None),
    ([256, 64], [256, 64], None, _b_shard_2d_256_64),
    ([256, 64], [256, 64], _b_shard_2d_256_64, _b_shard_2d_256_64),
]


@pytest.mark.parametrize("a_shape, b_shape, a_shard_fn, b_shard_fn", SHARDED_INPUT_CASES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_outer_sharded_inputs(a_shape, b_shape, a_shard_fn, b_shard_fn, dtype, device):
    """Sharded inputs must produce numerically correct output. Height-sharded
    layouts flow through unchanged; width- and block-sharded layouts are
    materialized as interleaved DRAM inside ttnn::outer before the unsqueeze."""
    torch.manual_seed(0)
    a_mem = a_shard_fn() if a_shard_fn else ttnn.DRAM_MEMORY_CONFIG
    b_mem = b_shard_fn() if b_shard_fn else ttnn.DRAM_MEMORY_CONFIG
    a_pt = _gen(a_shape, dtype)
    b_pt = _gen(b_shape, dtype)
    a_tt = _to_device(a_pt, dtype, device, ttnn.TILE_LAYOUT, memory_config=a_mem)
    b_tt = _to_device(b_pt, dtype, device, ttnn.TILE_LAYOUT, memory_config=b_mem)
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
    a_tt = _to_device(a_pt, dtype, device, ttnn.TILE_LAYOUT)
    b_tt = _to_device(b_pt, dtype, device, ttnn.TILE_LAYOUT)
    out_tt = ttnn.outer(a_tt, b_tt, memory_config=out_mem)
    _check(a_pt, b_pt, out_tt, dtype)


def _w_shard_for_outer_output(out_width_tiles):
    """Width-sharded memory config for an [N, M] outer-product output: shards
    the M dimension across out_width_tiles cores in a single row."""
    return ttnn.create_sharded_memory_config(
        [32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (out_width_tiles - 1, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _b_shard_for_outer_output():
    """Block-sharded memory config for a [64, 64] outer-product output: [32, 32]
    blocks across a 2x2 grid."""
    return ttnn.create_sharded_memory_config(
        [32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (1, 1))}),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize(
    "a_shape, b_shape, out_mem_fn",
    [
        # Width-sharded output [32, 256] across 8 cores ([32, 32] shards)
        ([32], [256], lambda: _w_shard_for_outer_output(8)),
        # Width-sharded output [32, 128] across 4 cores
        ([32], [128], lambda: _w_shard_for_outer_output(4)),
        # Block-sharded output [64, 64] across 2x2 grid
        ([64], [64], _b_shard_for_outer_output),
    ],
)
def test_outer_sharded_output_width_block(a_shape, b_shape, out_mem_fn, device):
    """Width- and block-sharded output memory configs (interleaved inputs)."""
    torch.manual_seed(0)
    dtype = ttnn.bfloat16
    out_mem = out_mem_fn()
    a_pt = _gen(a_shape, dtype)
    b_pt = _gen(b_shape, dtype)
    a_tt = _to_device(a_pt, dtype, device, ttnn.TILE_LAYOUT)
    b_tt = _to_device(b_pt, dtype, device, ttnn.TILE_LAYOUT)
    out_tt = ttnn.outer(a_tt, b_tt, memory_config=out_mem)
    _check(a_pt, b_pt, out_tt, dtype)


# ---------------------------------------------------------------------------
# Test 6b: sharded inputs + sharded output combined
# ---------------------------------------------------------------------------


def _w_shard_1d_256():
    """Width-sharded for a 1D [256] tensor (stored as [32, 256] in TILE_LAYOUT,
    so naturally width-sharded across 8 tile columns)."""
    return ttnn.create_sharded_memory_config(
        [32, 32],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 1))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.parametrize(
    "a_shape, b_shape, a_shard_fn, b_shard_fn, out_mem_fn",
    [
        # Width-sharded 1D a (forces fallback) + interleaved b + height-sharded output.
        # Exercises that the input-fallback + caller-supplied sharded output combo works.
        ([256], [32], _w_shard_1d_256, None, lambda: _h_shard_for_outer_output(8)),
        # Same but with width-sharded output instead of height
        ([32], [256], None, _w_shard_1d_256, lambda: _w_shard_for_outer_output(8)),
    ],
)
def test_outer_sharded_input_and_output(a_shape, b_shape, a_shard_fn, b_shard_fn, out_mem_fn, device):
    """Sharded inputs combined with a sharded output memory config in one call."""
    torch.manual_seed(0)
    dtype = ttnn.bfloat16
    a_mem = a_shard_fn() if a_shard_fn else ttnn.DRAM_MEMORY_CONFIG
    b_mem = b_shard_fn() if b_shard_fn else ttnn.DRAM_MEMORY_CONFIG
    out_mem = out_mem_fn()
    a_pt = _gen(a_shape, dtype)
    b_pt = _gen(b_shape, dtype)
    a_tt = _to_device(a_pt, dtype, device, ttnn.TILE_LAYOUT, memory_config=a_mem)
    b_tt = _to_device(b_pt, dtype, device, ttnn.TILE_LAYOUT, memory_config=b_mem)
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
    a_tt = _to_device(a_pt, dtype, device, ttnn.TILE_LAYOUT)
    b_tt = _to_device(b_pt, dtype, device, ttnn.TILE_LAYOUT)
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
    a_tt = _to_device(a_pt, dtype, device, ttnn.TILE_LAYOUT)
    b_tt = _to_device(b_pt, dtype, device, ttnn.TILE_LAYOUT)
    out_tt = ttnn.outer(a_tt, b_tt)
    _check(a_pt, b_pt, out_tt, dtype)


# ---------------------------------------------------------------------------
# Test 8b: integer dtypes (int32, uint32)
#
# Both downstream ops (ttnn.unsqueeze via reshape, and ttnn.multiply) accept
# int32 and uint32, so ttnn.outer supports them too. Integer multiply is exact
# when results stay in range, so we compare with assert_equal rather than PCC.
# ---------------------------------------------------------------------------


INTEGER_DTYPE_CASES = [
    (ttnn.int32, -100, 100),
    (ttnn.uint32, 0, 200),
]


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ([32], [32]),
        ([64], [128]),
        ([2, 3, 32], [2, 3, 32]),
        ([1, 8, 32], [4, 8, 32]),
    ],
    ids=["1d", "1d_asym", "batched", "broadcast_batch"],
)
@pytest.mark.parametrize("dtype, low, high", INTEGER_DTYPE_CASES, ids=["int32", "uint32"])
def test_outer_integer_dtypes(a_shape, b_shape, dtype, low, high, device):
    torch.manual_seed(0)
    a_pt = _gen(a_shape, dtype, low=low, high=high)
    b_pt = _gen(b_shape, dtype, low=low, high=high)
    a_tt = _to_device(a_pt, dtype, device, ttnn.TILE_LAYOUT)
    b_tt = _to_device(b_pt, dtype, device, ttnn.TILE_LAYOUT)
    out_tt = ttnn.outer(a_tt, b_tt)
    out_pt = ttnn.to_torch(out_tt)
    golden = _golden(a_pt, b_pt)
    assert tuple(out_pt.shape) == tuple(golden.shape), f"shape mismatch: tt={out_pt.shape}, golden={golden.shape}"
    # ttnn.to_torch returns torch.uint32 for UINT32 inputs while gen_func_with_cast_tt
    # casts UINT32 to torch.int32. torch.equal refuses mismatched signed/unsigned
    # dtypes, so widen both to int64 (safe: values stay well within range).
    assert torch.equal(out_pt.to(torch.int64), golden.to(torch.int64)), f"integer outer mismatch (dtype={dtype})"


# ---------------------------------------------------------------------------
# Test 9: dtypes outside the documented whitelist are rejected up front
# ---------------------------------------------------------------------------
# The supported set advertised by the nanobind docstring is
# {BFLOAT16, BFLOAT8_B, FLOAT32, INT32, UINT32}. Anything else (bfloat4_b,
# uint16, uint8) is rejected at entry with a "ttnn.outer: unsupported dtype"
# error rather than falling through to an opaque reshape_device_operation /
# multiply failure (related: https://github.com/tenstorrent/tt-metal/issues/44919).


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat4_b, ttnn.uint16, ttnn.uint8],
    ids=["bfloat4_b", "uint16", "uint8"],
)
def test_outer_unsupported_dtype_rejected(device, dtype, expect_error):
    a_pt = torch.rand([32], dtype=torch.bfloat16)
    b_pt = torch.rand([32], dtype=torch.bfloat16)
    a_tt = ttnn.from_torch(a_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "ttnn.outer: unsupported dtype"):
        ttnn.outer(a_tt, b_tt)


# ---------------------------------------------------------------------------
# Test 9b: dtype mismatch between inputs is rejected up front
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype_a, dtype_b",
    [
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.float32, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.int32, ttnn.uint32),
    ],
    ids=["bf16_fp32", "fp32_bf16", "bf16_bf8b", "i32_u32"],
)
def test_outer_dtype_mismatch_rejected(device, dtype_a, dtype_b, expect_error):
    """ttnn.outer requires both inputs to have the same dtype. Pin the explicit
    check so mismatched-dtype inputs raise a clearly attributable ttnn.outer
    error rather than bubbling up a generic multiply dtype-mismatch."""
    a_pt = torch.rand([32], dtype=torch.bfloat16)
    b_pt = torch.rand([32], dtype=torch.bfloat16)
    a_tt = ttnn.from_torch(a_pt, dtype=dtype_a, device=device, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype_b, device=device, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "ttnn.outer: inputs must have the same dtype"):
        ttnn.outer(a_tt, b_tt)


# ---------------------------------------------------------------------------
# Test 10: rank-0 (scalar) inputs are rejected with a clear ttnn.outer message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ([], [32]),
        ([32], []),
        ([], []),
    ],
    ids=["scalar_a", "scalar_b", "scalar_both"],
)
def test_outer_rank0_rejected(device, shape_a, shape_b, expect_error):
    """ttnn.outer requires both inputs to be at least 1D. Pin the explicit
    rank check so a scalar input raises a clearly attributable ttnn.outer
    error rather than bubbling up an opaque 'Dimension out of range' from
    ttnn.unsqueeze."""
    a_tt = ttnn.from_torch(torch.rand(shape_a, dtype=torch.bfloat16), dtype=ttnn.bfloat16, device=device)
    b_tt = ttnn.from_torch(torch.rand(shape_b, dtype=torch.bfloat16), dtype=ttnn.bfloat16, device=device)
    with expect_error(RuntimeError, "ttnn.outer: inputs must be at least 1D"):
        ttnn.outer(a_tt, b_tt)
