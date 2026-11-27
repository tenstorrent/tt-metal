# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_permute regime blocks_rm_gather (Refinement 1).

Cell: bfloat16, ROW_MAJOR, DRAM interleaved, rank 6, dims (0, 1, 4, 2, 5, 3):
out[n, c, i, h, j, w] = x[n, c, h, w, i, j]. Graded bit for bit. Besides the two FLUX
production shapes, the index map must hold for every shape validate() accepts: other W, I, J, H,
an odd J, input pages (J * 2 B) and output rows (W * 2 B) that are not multiples of 16 B,
input pages longer than one 64 B DRAM slot, and every alignment tag.
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import ExcludedCell
from ttnn.operations.g6_permute import g6_permute, g6_permute_plan

op_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute")
regime_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute_blocks_rm_gather")

BLOCKS = (0, 1, 4, 2, 5, 3)

# FLUX.2-dev production shapes (golden cases 73, 74).
PRODUCTION_SHAPES = [
    (1, 32, 64, 32, 2, 2),
    (1, 64, 128, 32, 2, 2),
]

# id, shape [N, C, H, W, I, J]
CELL_SHAPES = [
    ("odd_j_odd_w", (1, 3, 5, 7, 3, 3)),  # 6 B input pages, 14 B output rows
    ("j5_i1", (1, 2, 4, 16, 1, 5)),  # 10 B input pages, I = 1
    ("batch2", (2, 4, 6, 32, 2, 2)),  # N > 1
    ("w33", (1, 2, 3, 33, 2, 2)),  # 66 B output rows: two 64 B DRAM slots
    ("j36_long_page", (1, 1, 2, 64, 2, 36)),  # 72 B input pages: 128 B bank stride
    ("j8_w40", (1, 2, 3, 40, 4, 8)),  # 16 B input pages, 80 B output rows
    ("h_non_aligned_j32", (1, 2, 3, 8, 3, 32)),  # alignment tag h_non_aligned
    ("tile_aligned", (1, 1, 2, 4, 32, 32)),  # alignment tag tile_aligned
    ("i4_j1", (1, 3, 7, 24, 4, 1)),  # 2 B input pages
]

SPECIAL_BITS = (0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x8000, 0x0001, 0x807F, 0x0040)


def _shape_id(shape):
    return "x".join(str(d) for d in shape)


def _randn(shape):
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)


def _to_device(torch_input, device, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _bits(t):
    return t.to(torch.bfloat16).contiguous().view(torch.int16)


def _assert_bit_exact(actual, expected):
    assert list(actual.shape) == list(expected.shape)
    mismatch = _bits(actual) != _bits(expected)
    if bool(mismatch.any()):
        index = tuple(int(i) for i in torch.nonzero(mismatch)[0])
        pytest.fail(
            f"{int(mismatch.sum())} of {mismatch.numel()} values differ; first at {list(index)}: "
            f"got {int(_bits(actual)[index]) & 0xFFFF:#06x}, expected {int(_bits(expected)[index]) & 0xFFFF:#06x}"
        )


def _permute_and_check(torch_input, device, call_dims=BLOCKS):
    expected = torch.permute(torch_input, BLOCKS).contiguous()
    ttnn_output = g6_permute(_to_device(torch_input, device), call_dims)
    assert list(ttnn_output.shape) == list(expected.shape)
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT
    memory_config = ttnn_output.memory_config()
    assert memory_config.buffer_type == ttnn.BufferType.DRAM
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    actual = ttnn.to_torch(ttnn_output)
    _assert_bit_exact(actual, expected)
    return actual


@pytest.mark.parametrize("shape", PRODUCTION_SHAPES, ids=[_shape_id(s) for s in PRODUCTION_SHAPES])
def test_production_shapes(shape, device):
    torch.manual_seed(42)
    _permute_and_check(_randn(shape), device)


@pytest.mark.parametrize("shape", [c[1] for c in CELL_SHAPES], ids=[c[0] for c in CELL_SHAPES])
def test_other_shapes_of_the_cell(shape, device):
    torch.manual_seed(42)
    _permute_and_check(_randn(shape), device)


def test_special_values_move_bit_for_bit(device):
    torch.manual_seed(42)
    shape = (1, 4, 6, 32, 2, 2)
    bits = _randn(shape).contiguous().view(torch.int16).reshape(-1).clone()
    special = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL_BITS], dtype=torch.int16)
    positions = torch.arange(0, bits.numel(), 3)
    bits[positions] = special[torch.arange(positions.numel()) % special.numel()]
    _permute_and_check(bits.view(torch.bfloat16).reshape(shape), device)


def test_negative_dims_equal_canonical(device):
    torch.manual_seed(42)
    _permute_and_check(_randn((1, 4, 6, 32, 2, 2)), device, call_dims=(0, 1, -2, 2, -1, 3))


def test_plan_reports_regime(device):
    torch.manual_seed(42)
    shape = (1, 64, 128, 32, 2, 2)
    plan = g6_permute_plan(_to_device(_randn(shape), device), BLOCKS)
    assert plan.regime == "blocks_rm_gather"
    assert plan.tensor_units == 64 * 128
    assert plan.block_h == min(plan.units_per_core_g1, plan.block_cap)
    assert plan.num_cores == len(plan.assignment)
    assert sum(n for _, _, n in plan.assignment) == plan.tensor_units


@pytest.mark.parametrize(
    "shape, layout, dims",
    [
        ((1, 4, 6, 32, 2, 2), ttnn.TILE_LAYOUT, BLOCKS),
        ((1, 6, 16, 64), ttnn.ROW_MAJOR_LAYOUT, (0, 2, 1, 3)),
    ],
    ids=["tile_rank6", "row_major_rank4"],
)
def test_cells_without_regime_are_excluded(shape, layout, dims, device, expect_error):
    ttnn_input = _to_device(_randn(shape), device, layout=layout)
    with expect_error(ExcludedCell, ""):
        g6_permute(ttnn_input, dims)


def test_repeat_call_reuses_program(device):
    torch.manual_seed(42)
    shape = (1, 4, 6, 32, 2, 2)
    first = _randn(shape)
    second = _randn(shape)
    out_first = _permute_and_check(first, device)
    entries = device.num_program_cache_entries()
    out_again = _permute_and_check(first, device)
    _permute_and_check(second, device)
    assert device.num_program_cache_entries() == entries
    _assert_bit_exact(out_again, out_first)


@pytest.fixture
def knobs(monkeypatch):
    def force(block_limit, input_depth):
        monkeypatch.setattr(regime_module, "BLOCK_H_LIMIT", block_limit)
        monkeypatch.setattr(regime_module, "INPUT_DEPTH", input_depth)

    op_module._PROGRAM_CACHE.clear()
    yield force
    op_module._PROGRAM_CACHE.clear()


# id, shape, block_h limit, input_depth: multi-block rings, ragged last blocks, blocks across (n, c).
FORCED_BLOCKS = [
    ("depth2_ragged", (1, 32, 64, 32, 2, 2), 5, 2),
    ("depth1", (1, 32, 64, 32, 2, 2), 4, 1),
    ("ragged_crosses_nc_odd_j", (2, 8, 37, 7, 3, 3), 2, 2),
    ("depth3_block1_w33", (1, 3, 141, 33, 2, 2), 1, 3),
]


@pytest.mark.parametrize(
    "shape, block_limit, input_depth", [c[1:] for c in FORCED_BLOCKS], ids=[c[0] for c in FORCED_BLOCKS]
)
def test_forced_blocks_bit_exact(shape, block_limit, input_depth, device, knobs):
    knobs(block_limit, input_depth)
    torch.manual_seed(7)
    torch_input = _randn(shape)
    plan = g6_permute_plan(_to_device(torch_input, device), BLOCKS)
    assert plan.block_h <= block_limit
    if plan.units_per_core_g1 > block_limit:
        assert plan.max_blocks_per_core > 1
        assert plan.input_depth == min(input_depth, plan.max_blocks_per_core)
    _permute_and_check(torch_input, device)
