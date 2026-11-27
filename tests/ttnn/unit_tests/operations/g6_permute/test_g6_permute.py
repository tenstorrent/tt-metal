# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_permute, Phase 0: bfloat16, TILE, DRAM interleaved, rank 4,
"h_non_aligned", dims (0, 2, 1, 3). The gate is bit-exact (helpers.TOLERANCES: ("exact", None))."""

import math

import pytest
import torch
import ttnn

from ttnn.operations.g6_permute import g6_permute, g6_permute_plan

ROWS = (0, 2, 1, 3)

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

# Shapes of the same SUPPORTED cell beyond production: the index map and the block rule must
# hold for every shape validate() accepts.
GENERAL_SHAPES = [
    pytest.param((1, 1, 1, 32), id="single_tile"),
    pytest.param((3, 5, 7, 96), id="multi_batch_odd"),
    pytest.param((1, 40, 6, 64), id="a_two_tiles_ragged"),
    pytest.param((1, 6, 50, 64), id="b_two_tiles_ragged"),
    pytest.param((2, 40, 50, 128), id="multi_block_cores"),
    pytest.param((1, 33, 100, 512), id="runs_cross_band_and_tile"),
]

# bfloat16 bit patterns: +inf, -inf, quiet NaN, negative NaN with payload, -0, denormals.
SPECIAL_BITS = (0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x8000, 0x0001, 0x807F, 0x0040)


def torch_reference(x, dims):
    return torch.permute(x, dims).contiguous()


def to_device(x, device):
    return ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def randn_bf16(shape):
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)


def check_placement(out, expected_shape):
    assert list(out.shape) == list(expected_shape)
    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def assert_bit_exact(actual, expected):
    assert list(actual.shape) == list(expected.shape), f"shape {list(actual.shape)} != {list(expected.shape)}"
    a = actual.to(torch.bfloat16).contiguous().view(torch.int16)
    e = expected.to(torch.bfloat16).contiguous().view(torch.int16)
    mismatch = a != e
    if bool(mismatch.any()):
        index = tuple(int(i) for i in torch.nonzero(mismatch)[0])
        raise AssertionError(
            f"{int(mismatch.sum())} of {mismatch.numel()} values differ; first at {list(index)}: "
            f"got bits {int(a[index]) & 0xFFFF:#06x}, expected {int(e[index]) & 0xFFFF:#06x}"
        )


def run_and_check(torch_input, device, dims=ROWS, expected=None):
    ttnn_input = to_device(torch_input, device)
    if expected is None:
        expected = torch_reference(torch_input, tuple(d % 4 for d in dims))
    out = g6_permute(ttnn_input, dims)
    check_placement(out, expected.shape)
    actual = ttnn.to_torch(out)
    assert_bit_exact(actual, expected)
    return actual


@pytest.mark.parametrize("shape", PRODUCTION_SHAPES, ids=["x".join(map(str, s)) for s in PRODUCTION_SHAPES])
def test_production_shapes(shape, device):
    torch.manual_seed(42)
    run_and_check(randn_bf16(shape), device)


@pytest.mark.parametrize("shape", GENERAL_SHAPES)
def test_general_shapes(shape, device):
    torch.manual_seed(42)
    run_and_check(randn_bf16(shape), device)


def test_special_values_bit_exact(device):
    torch.manual_seed(42)
    shape = (1, 6, 16, 64)
    bits = randn_bf16(shape).contiguous().view(torch.int16).reshape(-1).clone()
    special = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL_BITS], dtype=torch.int16)
    positions = torch.arange(0, bits.numel(), 5)
    bits[positions] = special[torch.arange(positions.numel()) % special.numel()]
    torch_input = bits.view(torch.bfloat16).reshape(shape)

    ttnn_input = to_device(torch_input, device)
    on_device = ttnn.to_torch(ttnn_input)
    v = on_device.to(torch.float32)
    present = [
        bool((v == float("inf")).any()),
        bool((v == float("-inf")).any()),
        bool(torch.isnan(v).any()),
        bool(((v == 0) & torch.signbit(v)).any()),
        bool(((v != 0) & (v.abs() < torch.finfo(torch.float32).tiny)).any()),
    ]
    if not all(present):
        pytest.skip("host conversion dropped a special-value class; nothing to grade")

    out = g6_permute(ttnn_input, ROWS)
    check_placement(out, (1, 16, 6, 64))
    assert_bit_exact(ttnn.to_torch(out), torch_reference(on_device, ROWS))


@pytest.mark.parametrize(
    "shape, poison",
    [
        pytest.param((1, 6, 16, 64), float("inf"), id="inf_1x6x16x64"),
        pytest.param((1, 16, 1, 64), 1000.0, id="1000_1x16x1x64"),
    ],
)
def test_poisoned_input_padding_never_reaches_output(shape, poison, device):
    torch.manual_seed(42)
    torch_input = randn_bf16(shape)
    ttnn_input = to_device(torch_input, device)
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, poison, memory_config=ttnn_input.memory_config())
    out = g6_permute(ttnn_input, ROWS)
    expected = torch_reference(torch_input, ROWS)
    check_placement(out, expected.shape)
    assert_bit_exact(ttnn.to_torch(out), expected)


def test_negative_dims(device):
    torch.manual_seed(42)
    torch_input = randn_bf16((1, 6, 16, 64))
    run_and_check(torch_input, device, dims=(0, -2, 1, -1), expected=torch_reference(torch_input, ROWS))


@pytest.mark.parametrize(
    "dims",
    [(0, 2, 1), (0, 1, 1, 3), (0, 1, 2, 4), (0, 2, 1, -5)],
    ids=["short", "repeated", "out_of_range", "negative_out_of_range"],
)
def test_malformed_dims_raise(dims, device, expect_error):
    torch.manual_seed(42)
    ttnn_input = to_device(randn_bf16((1, 6, 16, 64)), device)
    # The contract names the exception types, not the message text.
    with expect_error((ValueError, RuntimeError), ""):
        g6_permute(ttnn_input, dims)


def test_repeat_call_is_bit_identical_and_cached(device):
    torch.manual_seed(42)
    torch_input = randn_bf16((1, 6, 16, 64))
    ttnn_input = to_device(torch_input, device)
    first = ttnn.to_torch(g6_permute(ttnn_input, ROWS))
    entries = device.num_program_cache_entries()
    second = ttnn.to_torch(g6_permute(ttnn_input, ROWS))
    assert device.num_program_cache_entries() == entries, "a repeat call built a new program"
    assert_bit_exact(second, first)
    assert_bit_exact(first, torch_reference(torch_input, ROWS))


@pytest.mark.parametrize("shape", PRODUCTION_SHAPES, ids=["x".join(map(str, s)) for s in PRODUCTION_SHAPES])
def test_plan_reports_rows_tile_multicore(shape, device):
    torch.manual_seed(42)
    ttnn_input = to_device(randn_bf16(shape), device)
    plan = g6_permute_plan(ttnn_input, ROWS)
    grid = device.compute_with_storage_grid_size()
    n, a, b, w = shape
    output_tiles = n * b * math.ceil(a / 32) * math.ceil(w / 32)
    assert plan.regime == "rows_tile"
    assert output_tiles >= 2
    assert 2 <= plan.num_cores <= grid.x * grid.y
    assert len(plan.core_roles) == plan.num_cores
