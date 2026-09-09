# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for `tilize` — the immutable spec for the implementer.

DO NOT MODIFY. If a case here fails, the implementation is wrong, not the test.

`tilize` re-lays a ROW_MAJOR tensor into TILE layout. There is no torch
counterpart that re-lays bytes into 32x32 four-face tiles, so the PyTorch
reference is the IDENTITY: reading the result back through `ttnn.to_torch`
(which untilizes) must reproduce the input. The layout / dtype / shape
assertions are therefore load-bearing — an implementation that returned its
input unchanged in ROW_MAJOR would pass a value comparison alone.

Design under test: `ttnn/ttnn/operations/tilize/op_design.md`.
Phase 0 rectangle exercised here: bfloat16, rank 4, tile-aligned,
interleaved DRAM -> interleaved DRAM, 32x32 tile, no padding, low_l1=False.
Later refinements extend DTYPES / SHAPES rather than editing the assertions.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.tilize import tilize


# Same thresholds as the golden suite. tilize does no arithmetic, so these are
# floors rather than error budgets — do not tighten or loosen them here.
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

# Phase 0 supports bfloat16 only; a dtype refinement adds entries here.
DTYPES = [ttnn.bfloat16]


def torch_tilize(torch_input):
    """PyTorch reference: tilize is value- and position-preserving, so the
    reference for the logical view of the output is the input itself."""
    return torch_input


# (shape, id, what it pins)
SHAPES = [
    # --- the four required shape classes -------------------------------------
    ((1, 1, 32, 32), "single_tile"),  # R=1, C=1 — the degenerate grid
    ((1, 1, 64, 128), "multi_tile"),  # R=2, C=4
    ((1, 1, 32, 128), "non_square_wide"),  # R=1, C=4
    ((1, 1, 128, 32), "non_square_tall"),  # R=4, C=1
    ((2, 3, 64, 96), "multi_batch"),  # R = 2*3*2 = 12 from the LEADING-dim fold, C=3
    # --- regime-pinned (op_design.md -> Work Distribution) -------------------
    # num_w_chunks == 1 on every arch (C=2, R >= num_cores) -> grid2d_full_width
    ((1, 1, 2048, 64), "regime_full_width__tall_narrow"),
    # R == 1, C=64 -> w_chunks_for_occupancy > 1 on any multi-core arch
    # -> grid2d_width_chunked. This is the geometry a row-only split collapses on.
    ((1, 1, 32, 2048), "regime_width_chunked__short_wide"),
    # Both axes carry real extent; neither dominates -> the block must be 2-D.
    ((1, 1, 1024, 1024), "square_large"),
]


def _make_input(shape, dtype, device):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        torch_input = torch_input.bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return torch_input, tt_input


def _check_contract(tt_output, torch_input, *, dtype, shape):
    """Layout / dtype / tile geometry / logical shape, then values.

    Order matters: for a LAYOUT op the layout assertion IS the point, and a
    wrong dtype or shape makes a value comparison meaningless.
    """
    assert tt_output.layout == ttnn.TILE_LAYOUT, f"expected TILE_LAYOUT, got {tt_output.layout}"
    assert tt_output.dtype == dtype, f"expected {dtype}, got {tt_output.dtype}"
    # A padded call grows only the PADDED shape; the logical shape must not grow.
    assert list(tt_output.shape) == list(shape), f"logical shape changed: {list(tt_output.shape)} != {list(shape)}"

    torch_output = ttnn.to_torch(tt_output)
    assert list(torch_output.shape) == list(shape)
    assert_with_pcc(torch_tilize(torch_input).float(), torch_output.float(), PCC[dtype])


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d))
@pytest.mark.parametrize("shape", [s for s, _ in SHAPES], ids=[i for _, i in SHAPES])
def test_tilize(device, shape, dtype):
    """The core contract: ROW_MAJOR in, TILE out, values and positions preserved."""
    torch_input, tt_input = _make_input(shape, dtype, device)
    tt_output = tilize(tt_input)
    _check_contract(tt_output, torch_input, dtype=dtype, shape=shape)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d))
def test_tilize_explicit_memory_config(device, dtype):
    """`memory_config=` names the output placement explicitly."""
    shape = (1, 1, 64, 128)
    torch_input, tt_input = _make_input(shape, dtype, device)
    tt_output = tilize(tt_input, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert tt_output.memory_config().buffer_type == ttnn.BufferType.DRAM
    _check_contract(tt_output, torch_input, dtype=dtype, shape=shape)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d))
def test_tilize_explicit_dtype(device, dtype):
    """`dtype=` is always accepted; passing the input's own dtype is the
    no-cast path and must behave exactly like omitting it."""
    shape = (1, 1, 64, 128)
    torch_input, tt_input = _make_input(shape, dtype, device)
    tt_output = tilize(tt_input, dtype=dtype)
    _check_contract(tt_output, torch_input, dtype=dtype, shape=shape)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d))
def test_tilize_leading_dim_fold(device, dtype):
    """The tile-row count comes from `prod(shape[:-2]) * ceil(H/tile_h)`, not
    from `shape[-2]` alone. An op that derives its row count from `shape[-2]`
    mis-sizes the work split here and drops or duplicates whole images."""
    shape = (8, 1, 64, 128)  # R = 8*2 = 16 tile-rows, C = 4
    torch_input, tt_input = _make_input(shape, dtype, device)
    tt_output = tilize(tt_input)
    _check_contract(tt_output, torch_input, dtype=dtype, shape=shape)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d))
def test_tilize_program_cache_hit(device, dtype):
    """A second call at the same shape / dtype / memory_config must reuse the
    cached program: the work split and block sizing are derived from the shape
    and the tile geometry, and that derivation belongs in the cached descriptor.
    A second call that re-enters it is a cache miss wearing a hit's clothing.
    """
    shape = (1, 1, 64, 128)
    torch_input, tt_input = _make_input(shape, dtype, device)

    tilize(tt_input)  # warm
    entries_before = device.num_program_cache_entries()
    tt_output = tilize(tt_input)
    entries_after = device.num_program_cache_entries()

    assert entries_after == entries_before, (
        f"tilize added {entries_after - entries_before} program cache entries on a "
        f"repeat call at an identical shape/dtype/memory_config — the derivation "
        f"is not inside the cached descriptor"
    )
    _check_contract(tt_output, torch_input, dtype=dtype, shape=shape)


@pytest.mark.parametrize(
    "shape,ids",
    [((1, 1, 32, 2048), "short_wide"), ((1, 1, 2048, 64), "tall_narrow")],
    ids=["short_wide", "tall_narrow"],
)
def test_tilize_transposed_geometry_pair(device, shape, ids):
    """The same tile count in two transposed geometries. Both must produce a
    correct result under one scheme — this pair is what makes the
    per-geometry claim in op_design.md checkable rather than assertable.
    """
    dtype = ttnn.bfloat16
    torch_input, tt_input = _make_input(shape, dtype, device)
    tt_output = tilize(tt_input)
    _check_contract(tt_output, torch_input, dtype=dtype, shape=shape)


# --- malformed requests: ValueError / RuntimeError, NOT a support refusal ----
#
# These are malformed REQUESTS, distinct from the registry support-refusals
# validate() raises (UnsupportedAxisValue / ExcludedCell, both
# NotImplementedError) for cells outside SUPPORTED — so the malformed checks
# must run AHEAD of the per-axis loop inside validate(). `expect_error` needs a
# match pattern; "tilize" is deliberately the loosest useful one, so the test
# pins the exception TYPE and its provenance without freezing the wording.

_ERRS = (ValueError, RuntimeError)


def test_tilize_rejects_unaligned_input_without_padding(device, expect_error):
    """Padding is opt-in: given no padding argument, a non-tile-aligned input
    is refused rather than silently padded."""
    torch_input = torch.randn((1, 1, 32, 50), dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(_ERRS, "tilize"):
        tilize(tt_input)


def test_tilize_rejects_bad_tile_geometry(device, expect_error):
    """A `tile` whose width is not 32 must be refused by the op.

    `ttnn.Tile([16, 16])` is deliberately chosen because it is a geometry the
    HARDWARE accepts (so `ttnn.Tile` itself constructs it and the refusal really
    is the op's), while tilize's contract requires width == 32. A height that is
    not a power-of-two fraction of 32 cannot be used as the witness here: the
    `Tile` constructor throws on it before the op is ever called."""
    torch_input, tt_input = _make_input((1, 1, 32, 64), ttnn.bfloat16, device)
    with expect_error(_ERRS, "tilize"):
        tilize(tt_input, tile=ttnn.Tile([16, 16]))


def test_tilize_rejects_tile_input_without_tile_kwarg(device, expect_error):
    """A TILE input with no `tile=` has nothing to re-tile to."""
    torch_input = torch.randn((1, 1, 32, 64), dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(_ERRS, "tilize"):
        tilize(tt_input)
