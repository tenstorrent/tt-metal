# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_pad (Phase 0: bfloat16, TILE, DRAM interleaved).

Immutable spec for the implementer. Production shapes only (BB-22). The gate is bit
equality of every logical output position (helpers.TOLERANCES is ("exact", None) for
every dtype); the output's own tile padding is not compared.

Covers both built regimes of op_design.md:
  A  tile_padding_extension (in place, last tile row of each slice, metadata-only view)
  B  new_storage (one pass, boundary tiles assembled in L1, fill-only tiles never read)
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_pad import g6_pad

TILE = 32


def torch_reference(x, padding, value):
    """torch.nn.functional.pad with the contract's pair order (leading -> trailing, aligned to
    the trailing dims) mapped to torch's flat list (last dim first). Computed in x's dtype."""
    flat = []
    for low, high in reversed([tuple(p) for p in padding]):
        flat += [int(low), int(high)]
    return torch.nn.functional.pad(x, flat, mode="constant", value=value)


def highs_of(shape, padding):
    return [0] * (len(shape) - len(padding)) + [p[1] for p in padding]


def is_extension(shape, padding):
    highs = highs_of(shape, padding)
    out_h = shape[-2] + highs[-2]
    only_h = highs[-2] > 0 and all(h == 0 for i, h in enumerate(highs) if i != len(shape) - 2)
    return only_h and -(-shape[-2] // TILE) == -(-out_h // TILE)


def assert_bit_equal(actual, expected):
    assert list(actual.shape) == list(expected.shape), f"shape {list(actual.shape)} != {list(expected.shape)}"
    assert actual.dtype == expected.dtype
    a = actual.contiguous().view(torch.int16)
    e = expected.contiguous().view(torch.int16)
    mismatches = int((a != e).sum())
    assert mismatches == 0, f"{mismatches} of {a.numel()} logical positions differ bitwise"


# (shape, padding, value) -- all Phase 0 production cases (feature_spec.py SCENARIOS ids in comments)
CASES = [
    # regime A, single tile (one tile row, one tile column)              -- case 269
    ((1, 31, 12), ((0, 0), (0, 1), (0, 0)), 0.0),
    # regime A, multi-batch: 16 slices of one tile row                   -- case 254
    ((16, 1, 1536), ((0, 0), (0, 31), (0, 0)), 0.0),
    # regime A, multi-tile-row slice (storage takeover required)         -- case 292
    ((1, 100, 1536), ((0, 0), (0, 28), (0, 0)), 0.0),
    # regime A, tall: 121 tile rows, only the last one may be touched    -- case 247
    ((1, 3844, 1536), ((0, 0), (0, 28), (0, 0)), 0.0),
    # regime A, wide short (p2, p3, p4)                                  -- cases 258, 253, 291
    ((1, 1, 2, 62080), ((0, 0), (0, 0), (0, 30), (0, 0)), 0.0),
    ((1, 1, 8, 62080), ((0, 0), (0, 0), (0, 8), (0, 0)), 0),
    ((1, 1, 8, 62080), ((0, 0), (0, 0), (0, 24), (0, 0)), 0.0),
    # regime B, dim -2 padded across tile rows (fill-only tile rows)     -- case 68
    ((1, 185, 128), ((0, 71), (0, 0)), 0),
    # regime B, both dims padded, non-square boundary, c = -2^127        -- cases 65, 72
    ((1, 1, 196, 196), ((0, 60), (0, 60)), -1.7014118346046923e38),
    ((1, 1, 207, 207), ((0, 49), (0, 49)), -1.7014118346046923e38),
]


def _ids(case):
    shape, padding, value = case
    return f"{'x'.join(map(str, shape))}-pad{'.'.join(str(h) for h in highs_of(shape, padding))}-v{value}"


def _make_input(shape, device, poison=None):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if poison is not None:
        tt_input = ttnn.fill_implicit_tile_padding(tt_input, poison)
    return torch_input, tt_input


@pytest.mark.parametrize("case", CASES, ids=[_ids(c) for c in CASES])
@pytest.mark.parametrize("poison", [None, 1000.0], ids=["zero_padded", "poisoned_padding"])
def test_g6_pad(device, case, poison):
    shape, padding, value = case
    torch_input, tt_input = _make_input(shape, device, poison)
    expected = torch_reference(torch_input, padding, value)
    raw_before = tt_input.cpu().to_torch_with_padded_shape().clone()

    tt_output = g6_pad(tt_input, padding, value=value)

    assert tt_output.dtype == ttnn.bfloat16
    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert_bit_equal(ttnn.to_torch(tt_output), expected)

    if is_extension(shape, padding):
        # Storage takeover: out is x's buffer at x's padded shape; x's logical values unchanged.
        assert tt_output.buffer_address() == tt_input.buffer_address()
        assert list(tt_output.padded_shape) == list(tt_input.padded_shape)
        assert list(tt_input.shape) == list(shape)
        assert_bit_equal(ttnn.to_torch(tt_input), torch_input)
        # Only the last tile row of each slice may change (raw padded buffer).
        raw_after = tt_input.cpu().to_torch_with_padded_shape()
        before = raw_before.reshape(-1, raw_before.shape[-2], raw_before.shape[-1])
        after = raw_after.reshape(-1, raw_after.shape[-2], raw_after.shape[-1])
        keep_rows = after.shape[-2] - TILE
        assert_bit_equal(after[:, :keep_rows, :], before[:, :keep_rows, :])
    else:
        assert tt_output.buffer_address() != tt_input.buffer_address()
        assert_bit_equal(ttnn.to_torch(tt_input), torch_input)


def test_g6_pad_value_positional(device):
    """`value` passed positionally (production call pattern)."""
    shape, padding, value = (1, 77, 1536), ((0, 0), (0, 19), (0, 0)), 0.0  # case 276
    torch_input, tt_input = _make_input(shape, device, poison=1000.0)
    expected = torch_reference(torch_input, padding, value)
    tt_output = g6_pad(tt_input, padding, value)
    assert_bit_equal(ttnn.to_torch(tt_output), expected)


def test_g6_pad_same_shape_two_values(device):
    """Program cache: same shape and padding, two fill values, then another padding (BB-07)."""
    shape = (1, 1, 8, 62080)
    for padding, value in (
        (((0, 0), (0, 0), (0, 8), (0, 0)), 0),
        (((0, 0), (0, 0), (0, 8), (0, 0)), 7.0),
        (((0, 0), (0, 0), (0, 24), (0, 0)), 0.1),
    ):
        torch_input, tt_input = _make_input(shape, device, poison=1000.0)
        expected = torch_reference(torch_input, padding, value)
        tt_output = g6_pad(tt_input, padding, value=value)
        assert_bit_equal(ttnn.to_torch(tt_output), expected)


@pytest.mark.parametrize(
    "padding, value",
    [
        (((0, 0), (1, 28), (0, 0)), 0.0),  # low != 0
        (((0, 0), (0, 28), (0, -4)), 0.0),  # high < 0
        (((0, 0), (0, 0), (0, 0), (0, 28), (0, 0)), 0.0),  # more pairs than dims
        ((), 0.0),  # no pair
        (((0, 0), (0, 28), (0, 0)), "0"),  # value not a number
    ],
    ids=["low_nonzero", "high_negative", "too_many_pairs", "no_pairs", "value_not_number"],
)
def test_g6_pad_refusals(device, expect_error, padding, value):
    _, tt_input = _make_input((1, 100, 1536), device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_pad(tt_input, padding, value=value)
