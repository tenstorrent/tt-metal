# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_permute, Phase 0.

Cell: bfloat16, TILE, DRAM interleaved, rank 4, H = shape[-2] not a multiple of 32,
W = shape[-1] a multiple of 32, dims (0, 2, 1, 3). A permute moves values and computes
none, so every case is graded bit for bit (helpers.TOLERANCES[bfloat16] = ("exact", None)).
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_permute import g6_permute

ROWS = (0, 2, 1, 3)

# The eight Qwen production shapes (golden cases 301-308).
PRODUCTION_SHAPES = [
    (1, 1, 16, 64),
    (1, 16, 1, 64),
    (1, 6, 16, 64),
    (1, 16, 6, 64),
    (1, 1, 8, 64),
    (1, 8, 1, 64),
    (1, 6, 8, 64),
    (1, 8, 6, 64),
]

# Other shapes of the same cell: the index map must hold for every shape validate()
# accepts, not only for A, B <= 16 and W = 64.
CELL_SHAPES = [
    (1, 1, 5, 32),  # one input tile
    (2, 6, 16, 64),  # batch above 1
    (1, 40, 6, 64),  # A > 32: two output tile rows per (n, b)
    (1, 6, 40, 64),  # B > 32: second input tile row partial, live rows past face row 16
    (2, 34, 47, 96),  # every axis past one tile, odd tile-column count
    (1, 64, 100, 256),  # many input tiles per core
]

# bfloat16 bit patterns that must survive the move unchanged.
SPECIAL_BITS = (
    0x7F80,  # +inf
    0xFF80,  # -inf
    0x7FC0,  # quiet NaN
    0xFFC1,  # negative NaN with a payload bit
    0x8000,  # -0
    0x0001,  # smallest positive denormal
    0x807F,  # negative denormal, every mantissa bit set
    0x0040,  # positive denormal
)


def _shape_id(shape):
    return "x".join(str(d) for d in shape)


def _randn(shape):
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)


def _to_device(torch_input, device):
    return ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _reference(torch_input, dims):
    return torch.permute(torch_input, dims).contiguous()


def _check_placement(ttnn_output, expected_shape):
    assert list(ttnn_output.shape) == list(expected_shape)
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.TILE_LAYOUT
    memory_config = ttnn_output.memory_config()
    assert memory_config.buffer_type == ttnn.BufferType.DRAM
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def _assert_bit_exact(actual, expected):
    # Compared as 16-bit patterns so NaN payloads, -0 and denormals count.
    assert list(actual.shape) == list(expected.shape)
    actual_bits = actual.to(torch.bfloat16).contiguous().view(torch.int16)
    expected_bits = expected.to(torch.bfloat16).contiguous().view(torch.int16)
    mismatch = actual_bits != expected_bits
    if bool(mismatch.any()):
        index = tuple(int(i) for i in torch.nonzero(mismatch)[0])
        pytest.fail(
            f"{int(mismatch.sum())} of {mismatch.numel()} values differ; first at {list(index)}: "
            f"got {int(actual_bits[index]) & 0xFFFF:#06x}, expected {int(expected_bits[index]) & 0xFFFF:#06x}"
        )


def _permute_and_check(ttnn_input, call_dims, source, canonical_dims=ROWS):
    expected = _reference(source, canonical_dims)
    ttnn_output = g6_permute(ttnn_input, call_dims)
    _check_placement(ttnn_output, expected.shape)
    actual = ttnn.to_torch(ttnn_output)
    _assert_bit_exact(actual, expected)
    return actual


@pytest.mark.parametrize("shape", PRODUCTION_SHAPES, ids=[_shape_id(s) for s in PRODUCTION_SHAPES])
def test_production_shapes(shape, device):
    torch.manual_seed(42)
    torch_input = _randn(shape)
    _permute_and_check(_to_device(torch_input, device), ROWS, torch_input)


@pytest.mark.parametrize("shape", CELL_SHAPES, ids=[_shape_id(s) for s in CELL_SHAPES])
def test_other_shapes_of_the_cell(shape, device):
    torch.manual_seed(42)
    torch_input = _randn(shape)
    _permute_and_check(_to_device(torch_input, device), ROWS, torch_input)


def test_special_values_move_bit_for_bit(device):
    torch.manual_seed(42)
    shape = (1, 6, 16, 64)
    bits = _randn(shape).contiguous().view(torch.int16).reshape(-1).clone()
    special = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL_BITS], dtype=torch.int16)
    positions = torch.arange(0, bits.numel(), 5)
    bits[positions] = special[torch.arange(positions.numel()) % special.numel()]
    torch_input = bits.view(torch.bfloat16).reshape(shape)

    ttnn_input = _to_device(torch_input, device)
    # Grade against the bits that are on the device.
    source = ttnn.to_torch(ttnn_input)
    _permute_and_check(ttnn_input, ROWS, source)


@pytest.mark.parametrize(
    "shape, poison",
    [((1, 6, 16, 64), float("inf")), ((1, 16, 1, 64), 1000.0), ((1, 6, 40, 64), float("-inf"))],
    ids=["inf", "1000", "neg_inf_two_tile_rows"],
)
def test_input_padding_never_reaches_output(shape, poison, device):
    torch.manual_seed(42)
    torch_input = _randn(shape)
    ttnn_input = _to_device(torch_input, device)
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, poison, memory_config=ttnn_input.memory_config())
    _permute_and_check(ttnn_input, ROWS, torch_input)


def test_negative_dims_equal_canonical(device):
    torch.manual_seed(42)
    torch_input = _randn((1, 6, 16, 64))
    _permute_and_check(_to_device(torch_input, device), (0, -2, 1, -1), torch_input)


@pytest.mark.parametrize(
    "dims",
    [(0, 2, 1), (0, 1, 1, 3), (0, 1, 2, 4), (0, 2, 1, -5)],
    ids=["short", "repeated", "above_range", "below_range"],
)
def test_malformed_dims_raise(dims, device, expect_error):
    torch.manual_seed(42)
    ttnn_input = _to_device(_randn((1, 6, 16, 64)), device)
    # Any message: the wording is the implementer's.
    with expect_error((ValueError, RuntimeError), ""):
        g6_permute(ttnn_input, dims)


def test_repeat_call_reuses_program(device):
    torch.manual_seed(42)
    shape = (1, 6, 16, 64)
    first = _randn(shape)
    second = _randn(shape)

    out_first = _permute_and_check(_to_device(first, device), ROWS, first)
    entries = device.num_program_cache_entries()
    out_again = _permute_and_check(_to_device(first, device), ROWS, first)
    _permute_and_check(_to_device(second, device), ROWS, second)

    assert device.num_program_cache_entries() == entries
    _assert_bit_exact(out_again, out_first)
