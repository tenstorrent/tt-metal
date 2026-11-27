# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_reshape Refinement 1: the rm_row_gather regime (ROW_MAJOR, DRAM interleaved)
for int32 / uint32 / bfloat16 / float32, plus int32 / uint32 on the TILE regimes. Bit-exact throughout."""

import importlib

import pytest
import torch

import ttnn
from ttnn.operations.g6_reshape import describe_plan, g6_reshape

_op_module = importlib.import_module("ttnn.operations.g6_reshape.g6_reshape")

L1_OP_BUDGET = 1347776

_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.int32: torch.int32,
    ttnn.uint32: torch.uint32,
}

# The int32 / uint32 ROW_MAJOR production cases of feature_spec.INPUTS, plus the [16] -> [4, 4] sequence target.
INT_RM_CASES = [
    ((1, 3844), (3844, 1), ttnn.int32),
    ((16,), (16, 1), ttnn.uint32),
    ((16,), (16, 1), ttnn.int32),
    ((8, 1), (8,), ttnn.uint32),
    ((8,), (8, 1), ttnn.uint32),
    ((8, 1), (8,), ttnn.int32),
    ((8,), (8, 1), ttnn.int32),
    ((8, 1), (1, 1, 1, 8), ttnn.uint32),
    ((1, 1, 1, 32), (32, 1), ttnn.uint32),
    ((1, 64), (64, 1), ttnn.int32),
    ((1, 31), (31, 1), ttnn.int32),
    ((1, 33), (33, 1), ttnn.int32),
    ((1, 256), (256, 1), ttnn.int32),
    ((1, 77), (77, 1), ttnn.int32),
    ((1, 1000), (1000, 1), ttnn.int32),
    ((1, 129), (129, 1), ttnn.int32),
    ((1, 4096), (4096, 1), ttnn.int32),
    ((1, 100), (100, 1), ttnn.int32),
    ((1, 1024), (1024, 1), ttnn.int32),
    ((1, 128), (128, 1), ttnn.int32),
    ((16,), (4, 4), ttnn.int32),
]

# Cells the per-axis SUPPORTED also claims, one acceptance case each (and a few general geometries):
# float RM (2- and 4-byte elements, odd widths, wide rows, column -> row) and int32 / uint32 TILE.
OTHER_RM_CASES = [
    ((32, 64), (64, 32), ttnn.bfloat16),
    ((3, 5, 7), (7, 15), ttnn.bfloat16),
    ((1, 5120), (5120,), ttnn.bfloat16),
    ((16, 1, 5120), (1, 1, 16, 5120), ttnn.bfloat16),
    ((2, 33, 40), (66, 40), ttnn.float32),
    ((1, 185), (1, 1, 185, 1), ttnn.float32),
    ((4096, 1), (1, 4096), ttnn.int32),
    ((7, 9), (3, 21), ttnn.uint32),
]

INT_TILE_CASES = [
    ((1, 512), (1, 1, 512, 1), ttnn.int32),
    ((16, 1, 64), (1, 1, 16, 64), ttnn.uint32),
    ((1, 1, 16, 4608), (16, 1, 4608), ttnn.int32),
]


def _case_id(case):
    src, dst, dtype = case[:3]
    return "x".join(map(str, src)) + "-to-" + "x".join(map(str, dst)) + "-" + str(dtype).split(".")[-1].lower()


def _input(shape, dtype, extremes=False):
    torch.manual_seed(7)
    if dtype == ttnn.int32:
        if extremes:
            vals = torch.tensor([2**31 - 1, -(2**31), 2**24 + 1, -(2**24) - 1, 0, -1], dtype=torch.int64)
            n = torch.Size(shape).numel()
            return vals.repeat(-(-n // 6))[:n].reshape(shape).to(torch.int32)
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int64).to(torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**32, shape, dtype=torch.int64).to(torch.uint32)
    return torch.randn(shape).to(_TORCH_DTYPE[dtype])


def _bits(t):
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).to(torch.int64) & 0xFFFF
    if t.dtype == torch.float32:
        return t.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return t.to(torch.int64) & 0xFFFFFFFF


def _to_device(t, dtype, layout, device):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, preserve_nan_values=True
    )


def _run_and_check(torch_input, target, dtype, layout, device):
    ttnn_input = _to_device(torch_input, dtype, layout, device)
    output = g6_reshape(ttnn_input, list(target))
    expected = torch.reshape(torch_input, tuple(target))
    assert tuple(output.shape) == tuple(expected.shape)
    assert output.dtype == dtype and output.layout == layout
    memory_config = output.memory_config()
    assert memory_config.buffer_type == ttnn.BufferType.DRAM
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    got = ttnn.to_torch(output)
    assert tuple(got.shape) == tuple(expected.shape)
    mismatch = _bits(got) != _bits(expected)
    assert not mismatch.any(), f"{int(mismatch.sum())} of {mismatch.numel()} values differ"
    return ttnn_input


@pytest.mark.parametrize("case", INT_RM_CASES + OTHER_RM_CASES, ids=_case_id)
def test_row_major_cases(case, device):
    src, dst, dtype = case
    ttnn_input = _run_and_check(_input(src, dtype), dst, dtype, ttnn.ROW_MAJOR_LAYOUT, device)
    plan = describe_plan(ttnn_input, list(dst))
    assert plan["regime"] == "rm_row_gather"
    assert plan["l1_bytes_per_core"] <= L1_OP_BUDGET


@pytest.mark.parametrize("case", INT_TILE_CASES, ids=_case_id)
def test_integer_tile_cases(case, device):
    src, dst, dtype = case
    _run_and_check(_input(src, dtype), dst, dtype, ttnn.TILE_LAYOUT, device)


def test_flagged_plan_and_extremes(device):
    """The perf-flagged [1,4096] -> [4096,1] int32: rm_row_gather on the full grid, within the L1 budget,
    bit-exact on full-range values (plain and through the inferred -1 entry)."""
    torch_input = _input((1, 4096), ttnn.int32, extremes=True)
    ttnn_input = _to_device(torch_input, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, device)
    plan = describe_plan(ttnn_input, [4096, 1])
    assert plan["regime"] == "rm_row_gather"
    assert plan["num_cores"] > 1
    assert plan["l1_bytes_per_core"] <= L1_OP_BUDGET
    _run_and_check(torch_input, (4096, 1), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, device)
    _run_and_check(torch_input, (-1, 1), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, device)


def test_sequence_same_input(device):
    """One input, two targets whose page maps differ: the descriptor cache must not replay the first."""
    torch_input = _input((16,), ttnn.int32)
    ttnn_input = _to_device(torch_input, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, device)
    for target in ([16, 1], [4, 4], [16, 1]):
        got = ttnn.to_torch(g6_reshape(ttnn_input, target))
        assert torch.equal(got.to(torch.int64), torch_input.reshape(target).to(torch.int64))


# Multi-block, depth-2 slot rotation, one mover, one core, and no read-through gaps: none of these run on
# the golden shapes (one block per mover on the full grid).
KNOB_SETTINGS = {
    "block1": {"MAX_CORES": 8, "MAX_BLOCK_ATOMS": 1},
    "block5_depth2": {"MAX_CORES": 8, "MAX_BLOCK_ATOMS": 5, "STAGE_DEPTH": 2, "IMAGE_DEPTH": 2},
    "one_mover": {"MAX_CORES": 8, "MOVERS_PER_CORE": 1, "MAX_BLOCK_ATOMS": 7},
    "one_core": {"MAX_CORES": 1, "MAX_BLOCK_ATOMS": 64},
    "no_read_gap": {"MAX_CORES": 8, "RM_READ_GAP_BYTES": 0},
}


@pytest.mark.parametrize("knobs", KNOB_SETTINGS.values(), ids=KNOB_SETTINGS.keys())
@pytest.mark.parametrize(
    "case",
    [((1, 4096), (4096, 1), ttnn.int32), ((3, 5, 7), (7, 15), ttnn.bfloat16), ((4096, 1), (1, 4096), ttnn.uint32)],
    ids=_case_id,
)
def test_row_major_block_knobs(case, knobs, device, monkeypatch):
    for name, value in knobs.items():
        monkeypatch.setattr(_op_module, name, value)
    src, dst, dtype = case
    torch_input = _input(src, dtype)
    ttnn_input = _run_and_check(torch_input, dst, dtype, ttnn.ROW_MAJOR_LAYOUT, device)
    plan = describe_plan(ttnn_input, list(dst))
    assert plan["l1_bytes_per_core"] <= L1_OP_BUDGET
