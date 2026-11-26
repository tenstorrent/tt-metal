# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_permute, Phase 0.

Phase 0 cell: bfloat16, TILE, DRAM interleaved, rank 4, h_non_aligned, dims (0, 2, 1, 3).
The op moves values and computes none, so every value check is bit-exact (the golden
suite's gate for bfloat16 is "exact"). Shapes beyond the eight production cases exist
because a SUPPORTED cell must hold for every shape validate() accepts; they are chosen so
that every transfer path of op_design.md (closed-cell regroup, a-split regroup, direct
row writes, row-run reads in both face bands) runs on any worker grid of 64 to 139 cores.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.g6_permute import describe_plan, g6_permute

ROWS = (0, 2, 1, 3)

PRODUCTION_SHAPES = [
    pytest.param((1, 1, 16, 64), id="qwen301"),
    pytest.param((1, 16, 1, 64), id="qwen302"),
    pytest.param((1, 6, 16, 64), id="qwen303"),
    pytest.param((1, 16, 6, 64), id="qwen304"),
    pytest.param((1, 1, 8, 64), id="qwen305"),
    pytest.param((1, 8, 1, 64), id="qwen306"),
    pytest.param((1, 6, 8, 64), id="qwen307"),
    pytest.param((1, 8, 6, 64), id="qwen308"),
]

GENERAL_SHAPES = [
    pytest.param((1, 1, 1, 32), id="single_tile"),
    pytest.param((2, 3, 5, 96), id="multi_batch_direct"),
    pytest.param((1, 70, 3, 64), id="a_beyond_one_tile_row_regroup"),
    pytest.param((1, 32, 20, 640), id="a_split_regroup"),
    pytest.param((3, 40, 50, 512), id="closed_cells_multi_batch"),
    pytest.param((1, 5, 47, 32), id="row_runs_lower_face_band"),
]

# bfloat16 bit patterns: +inf, -inf, quiet NaN, negative NaN with payload, -0, denormals.
SPECIAL_BITS = (0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x8000, 0x0001, 0x807F, 0x0040)


def pytorch_reference(x, dims):
    return torch.permute(x, tuple(dims)).contiguous()


def _random_input(shape, seed=42):
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)


def _to_device(x, device):
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _assert_placement(out, shape):
    assert list(out.shape) == list(shape), f"shape {list(out.shape)} != {list(shape)}"
    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.TILE_LAYOUT
    memory_config = out.memory_config()
    assert memory_config.buffer_type == ttnn.BufferType.DRAM
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def _assert_bit_exact(actual, expected):
    """Equal bit patterns; widening bfloat16 to float32 is exact, NaN payloads included."""
    assert list(actual.shape) == list(expected.shape), f"shape {list(actual.shape)} != {list(expected.shape)}"
    a = actual.to(torch.float32).contiguous().view(torch.int32)
    e = expected.to(torch.float32).contiguous().view(torch.int32)
    mismatch = a != e
    if bool(mismatch.any()):
        index = tuple(int(i) for i in torch.nonzero(mismatch)[0])
        pytest.fail(
            f"{int(mismatch.sum())} of {mismatch.numel()} values differ; first at {list(index)}: "
            f"got {actual[index].item()!r}, expected {expected[index].item()!r}"
        )


def _check(x_torch, x_device, dims=ROWS):
    expected = pytorch_reference(x_torch, dims)
    out = g6_permute(x_device, dims)
    _assert_placement(out, expected.shape)
    actual = ttnn.to_torch(out)
    _assert_bit_exact(actual, expected)
    return actual


@pytest.mark.parametrize("shape", PRODUCTION_SHAPES + GENERAL_SHAPES)
def test_g6_permute(shape, device):
    x = _random_input(shape)
    _check(x, _to_device(x, device))


def test_negative_dims(device):
    """(0, -2, 1, -1) is the cell (0, 2, 1, 3)."""
    x = _random_input((1, 6, 16, 64))
    _check(x, _to_device(x, device), dims=(0, -2, 1, -1))


@pytest.mark.parametrize(
    "shape, poison",
    [
        pytest.param((1, 6, 16, 64), float("inf"), id="direct_inf"),
        pytest.param((1, 16, 1, 64), 1000.0, id="direct_whole_tile_read_1000"),
        pytest.param((1, 70, 3, 64), float("inf"), id="regroup_inf"),
    ],
)
def test_input_padding_never_reaches_output(shape, poison, device):
    """Loud values in the input's tile padding rows must not reach a logical output position."""
    x = _random_input(shape)
    x_device = _to_device(x, device)
    x_device = ttnn.fill_implicit_tile_padding(x_device, poison, memory_config=x_device.memory_config())
    _check(x, x_device)


@pytest.mark.parametrize("shape", [(1, 6, 16, 64), (1, 32, 20, 640)], ids=["direct", "regroup"])
def test_special_values_move_bit_exact(shape, device):
    """inf, NaN (with payload), -0 and denormals keep their bits."""
    x = _random_input(shape)
    bits = x.contiguous().view(torch.int16).reshape(-1).clone()
    special = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL_BITS], dtype=torch.int16)
    positions = torch.arange(0, bits.numel(), 5)
    bits[positions] = special[torch.arange(positions.numel()) % special.numel()]
    x = bits.view(torch.bfloat16).reshape(shape)

    x_device = _to_device(x, device)
    # Grade against the bits on the device: the host conversion is not under test.
    on_device = ttnn.to_torch(x_device)
    v = on_device.to(torch.float32)
    classes = (
        torch.isnan(v),
        torch.isinf(v),
        (v == 0) & torch.signbit(v),
        (v != 0) & (v.abs() < torch.finfo(torch.float32).tiny),
    )
    if not all(bool(mask.any()) for mask in classes):
        pytest.skip("the device input lost a special-value class before the op ran")
    _check(on_device, x_device)


@pytest.mark.parametrize(
    "dims",
    [(0, 2, 1), (0, 1, 1, 3), (0, 1, 2, 4), (0, 1, 2, -5)],
    ids=["short", "repeated", "out_of_range", "out_of_range_negative"],
)
def test_malformed_dims_raise(dims, device, expect_error):
    x_device = _to_device(_random_input((1, 6, 16, 64)), device)
    with expect_error((ValueError, RuntimeError), "dims"):
        g6_permute(x_device, dims)


def test_repeat_call_is_bit_identical(device):
    x = _random_input((1, 16, 6, 64))
    x_device = _to_device(x, device)
    first = ttnn.to_torch(g6_permute(x_device, ROWS))
    second = ttnn.to_torch(g6_permute(x_device, ROWS))
    _assert_bit_exact(second, first)


def test_cache_hit_uses_fresh_addresses(device):
    """A repeat call with the same signature adds no program and reads/writes the new tensors."""
    shape = (1, 6, 16, 64)
    x1 = _random_input(shape, seed=1)
    x1_device = _to_device(x1, device)
    _check(x1, x1_device)
    entries = device.num_program_cache_entries()

    # x1_device stays alive, so the second input cannot reuse its address.
    x2 = _random_input(shape, seed=2)
    x2_device = _to_device(x2, device)
    _check(x2, x2_device)
    _check(x2, x2_device, dims=(0, -2, 1, -1))
    assert device.num_program_cache_entries() == entries, "a repeat call built a new program"


@pytest.mark.parametrize("shape", PRODUCTION_SHAPES + GENERAL_SHAPES)
def test_plan_report(shape, device):
    """describe_plan reports the selected plan (BB-08) and each core's role (BB-21)."""
    x_device = _to_device(_random_input(shape), device)
    plan = describe_plan(x_device, ROWS)

    for key in (
        "regime",
        "block_a",
        "block_b",
        "block_wt",
        "num_blocks",
        "num_cores",
        "depth_in",
        "depth_staging",
        "regroup",
        "cores",
    ):
        assert key in plan, f"plan lacks {key!r}"
    assert plan["regime"] == "rows_tile"
    assert plan["block_a"] in (1, 2, 4, 8, 16, 32)
    assert plan["block_b"] in (1, 2, 4, 8, 16, 32)
    assert plan["block_wt"] >= 1
    assert plan["regroup"] == (plan["block_a"] > 1)

    n, a, b, w = shape
    wt = math.ceil(w / 32)
    expected_blocks = (
        n * math.ceil(a / plan["block_a"]) * math.ceil(b / plan["block_b"]) * math.ceil(wt / plan["block_wt"])
    )
    assert plan["num_blocks"] == expected_blocks

    grid = device.compute_with_storage_grid_size()
    assert 1 <= plan["num_cores"] <= min(plan["num_blocks"], grid.x * grid.y)
    assert len(plan["cores"]) == plan["num_cores"]
    cores = sorted(plan["cores"], key=lambda c: c["start_block"])
    next_block = 0
    for core in cores:
        assert core["start_block"] == next_block, "core block ranges must tile [0, num_blocks)"
        assert core["num_blocks"] >= 1
        assert core["role"]
        next_block += core["num_blocks"]
    assert next_block == plan["num_blocks"]

    output_pages = n * b * math.ceil(a / 32) * wt
    if output_pages > 1:
        assert plan["num_cores"] > 1, "an output of more than one page must run on more than one core"

    assert describe_plan(x_device, ROWS) == plan, "the plan must be deterministic"


def test_plan_paths_covered(device):
    """The shapes of this file run the regroup path, the direct path and row-run reads."""
    grid = device.compute_with_storage_grid_size()
    if not 64 <= grid.x * grid.y <= 139:
        pytest.skip(f"path coverage of this file is designed for 64 to 139 worker cores, not {grid.x * grid.y}")
    plans = []
    for param in PRODUCTION_SHAPES + GENERAL_SHAPES:
        (shape,) = param.values
        plans.append(describe_plan(_to_device(_random_input(shape), device), ROWS))
    assert any(p["regroup"] for p in plans), "no shape exercised the regroup path"
    assert any(not p["regroup"] for p in plans), "no shape exercised the direct path"
    assert any(p["block_b"] < 32 for p in plans), "no shape exercised row-run reads"
