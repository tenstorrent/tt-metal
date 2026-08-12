# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for `ttnn.operations.tilize.tilize` — IMMUTABLE.

This file is the SPEC. The implementer must not modify it.

`tilize` is a pure layout conversion: ROW_MAJOR in, TILE out, values unchanged.
There is no arithmetic golden — the PyTorch reference is the **identity**:

    to_torch(tilize(from_torch(x, ROW_MAJOR))) == x

and, when padding is requested, the pad-then-identity pair:

    out.cpu().to_torch_with_padded_shape() == F.pad(x, ..., value=pad_value)
    to_torch(out)                          == x          (logical shape unchanged)

Scope note: this file spans the whole op contract, not just Phase 0. Tests
covering capabilities a later refinement lands (dtypes beyond bfloat16, the
padded path, sharded I/O, tiny tiles, higher/lower ranks) fail until that
refinement lands — that is the intended behaviour of an acceptance spec. The
per-phase gate is the golden suite (`eval/golden_tests/tilize/`), whose xfail
machinery tracks Phase 0's narrower SUPPORTED rectangle.

See `ttnn/ttnn/operations/tilize/op_design.md` for the design this tests.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.tilize import tilize


# PCC thresholds keyed by dtype — identical to the golden suite's. Do NOT
# tighten these because "a layout op should be exact"; the harness-wide
# convention is what other ops are held to.
PCC = {
    ttnn.bfloat16: 0.995,
    ttnn.float32: 0.999,
    ttnn.bfloat8_b: 0.99,
    ttnn.uint32: 0.999,
    ttnn.uint16: 0.999,
    ttnn.int32: 0.999,
    ttnn.uint8: 0.999,
}

# Refusal message contract: both structural refusals (an unaligned input with no
# padding arguments, and a retile that also asks for a pad) MUST name padding in
# their message, so CI log triage and this test can match on it. Pinned in
# op_design.md under "Refusal message contract".
REFUSAL_MENTIONS_PAD = r"(?i)pad"

# The four canonical interleaved shapes: single-tile, multi-tile, non-square
# (wide-short — the width-parallelism regime), multi-batch.
SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile, square-ish
    (1, 1, 32, 512),  # non-square / wide-short: nt_h == 1, Wt == 16
    (2, 3, 64, 96),  # multi-batch, non-square
]


def _torch_source(shape, dtype):
    """Deterministic source tensor for `dtype`. torch.randn + manual_seed(42)."""
    torch.manual_seed(42)
    if dtype in (ttnn.uint8, ttnn.uint16, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    if dtype == ttnn.int32:
        return torch.randint(-1000, 1000, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.randn(shape, dtype=torch.float32)
    return torch.randn(shape).bfloat16()


def _cmp(expected, actual, dtype):
    """Compare in float32 at the dtype's PCC threshold."""
    assert list(actual.shape) == list(
        expected.shape
    ), f"shape mismatch: got {list(actual.shape)}, want {list(expected.shape)}"
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), PCC[dtype])


def _skip_unless_blackhole(what):
    """Retiling is Blackhole-only: a SKIP there is the correct outcome, not a
    failure, and must not be read as missing support. Mirrors
    `eval/golden_tests/tilize/helpers.py::skip_if_retile_unsupported`. Plain
    tiny tiles are NOT arch-gated and are never routed through here."""
    if not ttnn.device.is_blackhole():
        pytest.skip(f"{what} requires Blackhole (LLK unavailable on this arch)")


def _to_device(torch_tensor, device, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


# ---------------------------------------------------------------------------
# 1. The core contract: identity, every shape, single- and multi-core
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("use_multicore", [False, True], ids=["single_core", "multi_core"])
def test_tilize_identity(device, shape, use_multicore):
    dtype = ttnn.bfloat16
    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype)

    tt_output = tilize(tt_input, use_multicore=use_multicore)

    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert list(tt_output.shape) == list(shape), "logical shape must be preserved"
    _cmp(torch_input, ttnn.to_torch(tt_output), dtype)


# ---------------------------------------------------------------------------
# 2. Output memory config (interleaved DRAM / L1) and L1 input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "in_mem, out_mem",
    [
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        (ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ],
    ids=["dram_to_dram", "dram_to_l1", "l1_to_l1", "l1_to_dram"],
)
def test_tilize_memory_config(device, in_mem, out_mem):
    shape, dtype = (1, 1, 64, 128), ttnn.bfloat16
    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype, memory_config=in_mem)

    tt_output = tilize(tt_input, memory_config=out_mem)

    _cmp(torch_input, ttnn.to_torch(tt_output), dtype)


# ---------------------------------------------------------------------------
# 3. Rank-agnostic folding of the leading dims (rank 2 / 3 / 4 / 5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(64, 128), (2, 32, 64), (1, 1, 64, 64), (1, 2, 3, 64, 32)],
    ids=["rank2", "rank3", "rank4", "rank5"],
)
def test_tilize_rank(device, shape):
    dtype = ttnn.bfloat16
    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype)

    tt_output = tilize(tt_input)

    assert list(tt_output.shape) == list(shape)
    _cmp(torch_input, ttnn.to_torch(tt_output), dtype)


# ---------------------------------------------------------------------------
# 4. dtype passthrough and the value-preserving output cast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.float32, ttnn.uint32, ttnn.uint8],
    ids=["bfloat16", "float32", "uint32", "uint8"],
)
def test_tilize_dtype_passthrough(device, dtype):
    shape = (1, 1, 64, 128)
    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype)

    tt_output = tilize(tt_input)

    assert tt_output.dtype == dtype
    _cmp(torch_input, ttnn.to_torch(tt_output), dtype)


@pytest.mark.parametrize(
    "dtype, output_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat16),  # identity: no reconfigure
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.float32, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.bfloat8_b),
    ],
    ids=["bf16_to_bf16", "bf16_to_fp32", "fp32_to_bf16", "bf16_to_bf8b"],
)
def test_tilize_output_dtype_cast(device, dtype, output_dtype):
    shape = (1, 1, 64, 128)
    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype)

    tt_output = tilize(tt_input, dtype=output_dtype)

    assert tt_output.dtype == output_dtype
    # A narrowing cast is value-preserving to the target format's precision, so
    # compare at the OUTPUT dtype's threshold.
    _cmp(torch_input, ttnn.to_torch(tt_output), output_dtype)


# ---------------------------------------------------------------------------
# 5. Padding is never implicit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 50, 64), (1, 1, 64, 50), (1, 1, 50, 50)],
    ids=["h_non_aligned", "w_non_aligned", "hw_non_aligned"],
)
def test_tilize_unaligned_without_pad_args_raises(device, shape, expect_error):
    """A caller who did not ask for padding must never receive a tensor whose
    padded shape differs from its input's. The refusal message must mention
    padding (see op_design.md 'Refusal message contract')."""
    dtype = ttnn.bfloat16
    tt_input = _to_device(_torch_source(shape, dtype), device, dtype)

    with expect_error((ValueError, RuntimeError), REFUSAL_MENTIONS_PAD):
        tilize(tt_input)


# ---------------------------------------------------------------------------
# 6. Padding: auto tile-rounding, explicit target, whole pad tiles, fill signs
# ---------------------------------------------------------------------------


def _check_padded(tt_output, torch_input, padded_shape, pad_value, dtype):
    """Both readback views must hold: padded == F.pad(x), logical == x."""
    H, W = torch_input.shape[-2], torch_input.shape[-1]
    Hp, Wp = padded_shape[-2], padded_shape[-1]
    expected = F.pad(torch_input.to(torch.float32), (0, Wp - W, 0, Hp - H), value=float(pad_value))

    padded_readback = tt_output.cpu().to_torch_with_padded_shape()
    assert list(padded_readback.shape) == list(
        padded_shape
    ), f"padded shape {list(padded_readback.shape)} != target {list(padded_shape)}"
    _cmp(expected, padded_readback, dtype)

    # The LOGICAL shape must NOT have been promoted to the pad target.
    assert list(tt_output.shape) == list(torch_input.shape), (
        f"logical shape promoted to {list(tt_output.shape)}; " f"must still be {list(torch_input.shape)}"
    )
    _cmp(torch_input, ttnn.to_torch(tt_output), dtype)


@pytest.mark.parametrize(
    "shape, pad_value",
    [
        ((1, 1, 30, 32), 0.0),  # H tail only
        ((1, 1, 32, 50), 3.5),  # W tail only, nonzero fill
        ((1, 1, 50, 50), -18.0),  # both tails, negative fill
        ((1, 1, 64, 64), 0.0),  # already aligned: a legal no-op pad
    ],
    ids=["h_tail", "w_tail_positive", "both_tails_negative", "noop_pad"],
)
def test_tilize_pad_auto(device, shape, pad_value):
    """pad_mode="auto": target inferred by rounding H and W up to a multiple of 32."""
    dtype = ttnn.bfloat16
    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype)

    tt_output = tilize(tt_input, pad_value=pad_value)

    padded_shape = list(shape[:-2]) + [
        ((shape[-2] + 31) // 32) * 32,
        ((shape[-1] + 31) // 32) * 32,
    ]
    _check_padded(tt_output, torch_input, padded_shape, pad_value, dtype)


@pytest.mark.parametrize(
    "shape, padded_shape, pad_value",
    [
        # tile-rounded target: W tail + H tail only
        ((1, 1, 50, 50), [1, 1, 64, 64], 10.0),
        # BEYOND the tile-rounded target in both dims: adds WHOLE PAD TILES
        ((1, 1, 50, 50), [1, 1, 128, 128], -18.0),
        # W-only beyond tile-round: W tail + whole pad tile-columns, no H pad
        ((1, 1, 32, 50), [1, 1, 32, 128], 3.5),
        # H-only beyond tile-round, rank 3
        ((3, 100, 128), [3, 128, 128], 10.2),
        # rank 2, zero fill
        ((50, 50), [64, 64], 0.0),
    ],
    ids=["tile_rounded", "whole_pad_tiles_hw", "whole_pad_tiles_w", "rank3_h", "rank2_zero"],
)
def test_tilize_pad_explicit(device, shape, padded_shape, pad_value):
    """pad_mode="explicit": honour output_padded_shape, including targets that
    exceed the tile-rounded shape (the whole-pad-tile path)."""
    dtype = ttnn.bfloat16
    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype)

    tt_output = tilize(tt_input, output_padded_shape=padded_shape, pad_value=pad_value)

    _check_padded(tt_output, torch_input, padded_shape, pad_value, dtype)


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.float32, ttnn.uint32],
    ids=["bfloat16", "float32", "uint32"],
)
def test_tilize_pad_dtype(device, dtype):
    """The fill is packed in the INPUT element format and, for sub-word dtypes,
    replicated across the 32-bit store word. A NONZERO fill is what catches a
    value written only once per word."""
    shape, padded_shape = (1, 1, 50, 50), [1, 1, 64, 64]
    pad_value = 7 if dtype == ttnn.uint32 else 7.0
    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype)

    tt_output = tilize(tt_input, output_padded_shape=padded_shape, pad_value=pad_value)

    _check_padded(tt_output, torch_input, padded_shape, pad_value, dtype)


def test_tilize_pad_with_cast(device):
    """A padded call that ALSO casts: the fill must be encoded in the input
    dtype, not the output dtype."""
    shape, padded_shape, pad_value = (1, 1, 50, 50), [1, 1, 64, 64], -4.25
    dtype, output_dtype = ttnn.float32, ttnn.bfloat16
    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype)

    tt_output = tilize(tt_input, dtype=output_dtype, output_padded_shape=padded_shape, pad_value=pad_value)

    assert tt_output.dtype == output_dtype
    _check_padded(tt_output, torch_input, padded_shape, pad_value, output_dtype)


def test_tilize_pad_scalar(device):
    """rank 0 padded out to a single tile: the data region is the one input
    value, every other position is the fill."""
    dtype, pad_value = ttnn.bfloat16, 42.0
    torch.manual_seed(42)
    torch_input = torch.randn(()).bfloat16()
    tt_input = _to_device(torch_input, device, dtype)

    tt_output = tilize(tt_input, output_padded_shape=[32, 32], pad_value=pad_value)

    padded = tt_output.cpu().to_torch_with_padded_shape().to(torch.float32)
    assert list(padded.shape) == [32, 32]
    assert padded[0, 0].item() == pytest.approx(torch_input.to(torch.float32).item())
    mask = torch.ones(32, 32, dtype=torch.bool)
    mask[0, 0] = False
    assert torch.all(padded[mask] == pytest.approx(pad_value))


# ---------------------------------------------------------------------------
# 7. use_double_buffer — identity is unchanged either way (only L1 differs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_double_buffer", [False, True], ids=["depth1", "depth2"])
def test_tilize_double_buffer(device, use_double_buffer):
    shape, dtype = (1, 1, 64, 512), ttnn.bfloat16
    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype)

    tt_output = tilize(tt_input, use_double_buffer=use_double_buffer)

    _cmp(torch_input, ttnn.to_torch(tt_output), dtype)


# ---------------------------------------------------------------------------
# 8. Sharded I/O — same-spec zero-copy (HEIGHT / WIDTH / BLOCK) and crossovers
# ---------------------------------------------------------------------------


def _shard_mem_config(scheme, grid, shard_shape, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    return ttnn.MemoryConfig(
        scheme,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard_shape, orientation),
    )


def _grid(end_x, end_y, device):
    dev_grid = device.compute_with_storage_grid_size()
    if end_x > dev_grid.x - 1 or end_y > dev_grid.y - 1:
        pytest.skip(f"shard grid ({end_x},{end_y}) exceeds device grid")
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))})


@pytest.mark.parametrize(
    "shape, scheme, grid_xy, shard_shape",
    [
        ((1, 1, 512, 64), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (3, 0), (128, 64)),
        ((1, 1, 64, 512), ttnn.TensorMemoryLayout.WIDTH_SHARDED, (3, 0), (64, 128)),
        ((1, 1, 128, 128), ttnn.TensorMemoryLayout.BLOCK_SHARDED, (1, 1), (64, 64)),
    ],
    ids=["height", "width", "block"],
)
def test_tilize_sharded_same_spec(device, shape, scheme, grid_xy, shard_shape):
    """Same-spec L1-sharded in -> out: zero DRAM traffic on either side."""
    dtype = ttnn.bfloat16
    grid = _grid(*grid_xy, device)
    mem = _shard_mem_config(scheme, grid, shard_shape)

    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype, memory_config=mem)

    tt_output = tilize(tt_input, memory_config=mem)

    _cmp(torch_input, ttnn.to_torch(tt_output), dtype)


def test_tilize_interleaved_to_sharded(device):
    """DRAM-interleaved RM in -> HEIGHT-sharded TILE out (split reader)."""
    shape, dtype = (1, 1, 128, 64), ttnn.bfloat16
    grid = _grid(3, 0, device)
    out_mem = _shard_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, grid, (32, 64))

    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype)

    tt_output = tilize(tt_input, memory_config=out_mem)

    _cmp(torch_input, ttnn.to_torch(tt_output), dtype)


def test_tilize_sharded_to_interleaved(device):
    """HEIGHT-sharded RM in -> DRAM-interleaved TILE out (split writer)."""
    shape, dtype = (1, 1, 128, 64), ttnn.bfloat16
    grid = _grid(3, 0, device)
    in_mem = _shard_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, grid, (32, 64))

    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype, memory_config=in_mem)

    tt_output = tilize(tt_input, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    _cmp(torch_input, ttnn.to_torch(tt_output), dtype)


def test_tilize_sharded_cross_spec(device):
    """Input shard spec != output shard spec — the general cross-core reshard
    path. Must move through L1, never stage through DRAM."""
    shape, dtype = (1, 1, 128, 64), ttnn.bfloat16
    in_mem = _shard_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _grid(3, 0, device), (32, 64))
    out_mem = _shard_mem_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _grid(1, 0, device), (64, 64))

    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype, memory_config=in_mem)

    tt_output = tilize(tt_input, memory_config=out_mem)

    _cmp(torch_input, ttnn.to_torch(tt_output), dtype)


def test_tilize_sharded_col_major(device):
    """COL_MAJOR shard orientation: same math, column-first shard-grid walk."""
    shape, dtype = (1, 1, 256, 64), ttnn.bfloat16
    grid = _grid(0, 3, device)
    mem = _shard_mem_config(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        grid,
        (64, 64),
        ttnn.ShardOrientation.COL_MAJOR,
    )

    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype, memory_config=mem)

    tt_output = tilize(tt_input, memory_config=mem)

    _cmp(torch_input, ttnn.to_torch(tt_output), dtype)


# ---------------------------------------------------------------------------
# 9. Tile geometry: tiny tiles (all parts) and retile (Blackhole-only, skip)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tile_height", [16, 8, 4, 2, 1])
def test_tilize_tiny_tile(device, tile_height):
    """A 32-wide tile with fewer than 32 rows. NOT arch-gated — must work
    everywhere, including the degenerate 1x32 (one stick per tile)."""
    shape, dtype = (1, 1, 128, 256), ttnn.bfloat16
    torch_input = _torch_source(shape, dtype)
    tt_input = _to_device(torch_input, device, dtype)

    tt_output = tilize(tt_input, tile=ttnn.Tile([tile_height, 32]))

    assert tt_output.layout == ttnn.TILE_LAYOUT
    _cmp(torch_input, ttnn.to_torch(tt_output), dtype)


@pytest.mark.parametrize("in_tile_height, out_tile_height", [(32, 8), (1, 32)], ids=["shrink_32_to_8", "grow_1_to_32"])
def test_tilize_retile(device, in_tile_height, out_tile_height):
    """A TILE-layout input re-tiled to a different tile height. Blackhole-only:
    SKIP (not fail) on a part without the LLK support."""
    _skip_unless_blackhole(f"retile {in_tile_height}->{out_tile_height}")

    shape, dtype = (1, 1, 128, 256), ttnn.bfloat16
    torch_input = _torch_source(shape, dtype)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        tile=ttnn.Tile([in_tile_height, 32]),
    )

    tt_output = tilize(tt_input, tile=ttnn.Tile([out_tile_height, 32]))

    _cmp(torch_input, ttnn.to_torch(tt_output), dtype)


def test_tilize_retile_with_pad_is_refused(device, expect_error):
    """Retiling and padding are mutually exclusive — a TILE input is
    tile-aligned by construction, so there is nothing to pad."""
    _skip_unless_blackhole("retile")

    shape, dtype = (1, 1, 128, 256), ttnn.bfloat16
    tt_input = ttnn.from_torch(
        _torch_source(shape, dtype),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        tile=ttnn.Tile([32, 32]),
    )

    with expect_error((ValueError, RuntimeError), REFUSAL_MENTIONS_PAD):
        tilize(tt_input, tile=ttnn.Tile([8, 32]), pad_value=0.0)


# ---------------------------------------------------------------------------
# 10. Program cache: second call with the same spec must hit
# ---------------------------------------------------------------------------


def test_tilize_program_cache_hit(device):
    shape, dtype = (1, 1, 64, 128), ttnn.bfloat16
    torch_input = _torch_source(shape, dtype)

    tt_input = _to_device(torch_input, device, dtype)
    out_1 = tilize(tt_input)
    _cmp(torch_input, ttnn.to_torch(out_1), dtype)
    entries_after_first = device.num_program_cache_entries()

    tt_input_2 = _to_device(torch_input, device, dtype)
    out_2 = tilize(tt_input_2)
    _cmp(torch_input, ttnn.to_torch(out_2), dtype)

    assert device.num_program_cache_entries() == entries_after_first, (
        "second call with identical shape/dtype/mem_config/pad args must hit the " "program cache (no new entry)"
    )
