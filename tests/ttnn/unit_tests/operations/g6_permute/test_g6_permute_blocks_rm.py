# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_permute regimes blocks_rm_gather (Refinement 1) and blocks_rm_bank_coalesced
(Refinement 5): the same cell, one read per input page or one read of whole slots per bank run.

Cell: bfloat16, ROW_MAJOR, DRAM interleaved, rank 6, dims (0, 1, 4, 2, 5, 3):
out[n, c, i, h, j, w] = x[n, c, h, w, i, j]. Graded bit for bit. Besides the two FLUX
production shapes, the index map must hold for every shape validate() accepts: other W, I, J, H,
an odd J, input pages (J * 2 B) and output rows (W * 2 B) that are not multiples of 16 B,
input pages longer than one 64 B DRAM slot, and every alignment tag.
"""

import importlib
from fractions import Fraction

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
    # DRAM input reads whole slots per bank run.
    assert plan.regime == "blocks_rm_bank_coalesced"
    assert plan.coalesced
    assert plan.tensor_units == 64 * 128
    pages_per_unit = shape[3] * shape[4]
    block_target = -(-regime_module.COALESCED_BLOCK_PAGES[ttnn.BufferType.DRAM] // pages_per_unit)
    assert plan.block_h == min(plan.units_per_core_g1, plan.block_cap, block_target)
    assert plan.read_alt_share == regime_module.COALESCED_ALT_NOC_SHARE
    assert not plan.bank_major
    assert plan.page_stride == 4  # 4 B payloads packed word by word
    assert plan.staging_depth == min(regime_module.STAGING_DEPTH, plan.max_blocks_per_core)
    # 64 pages per unit: every block starts on a whole bank round.
    assert plan.staging_rounds == plan.block_h * pages_per_unit // plan.num_banks
    assert plan.num_cores == len(plan.assignment)
    assert sum(n for _, _, n in plan.assignment) == plan.tensor_units


def test_plan_reports_gather_regime(device, monkeypatch):
    monkeypatch.setattr(regime_module, "READ_MODE", "gather")
    torch.manual_seed(42)
    shape = (1, 64, 128, 32, 2, 2)
    plan = g6_permute_plan(_to_device(_randn(shape), device), BLOCKS)
    assert plan.regime == "blocks_rm_gather"
    assert not plan.coalesced
    pages_per_unit = shape[3] * shape[4]
    block_target = -(-regime_module.BLOCK_PAGES // pages_per_unit)
    assert plan.block_h == min(plan.units_per_core_g1, plan.block_cap, block_target)
    # DRAM input: half the page rounds on the second NoC, issued bank by bank.
    assert plan.read_alt_share == Fraction(1, 2)
    assert plan.bank_major
    assert plan.page_stride == plan.slot_bytes


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
    def force(block_limit, input_depth, **extra):
        monkeypatch.setattr(regime_module, "READ_MODE", "gather")
        monkeypatch.setattr(regime_module, "BLOCK_PAGES", None)
        monkeypatch.setattr(regime_module, "BLOCK_H_LIMIT", block_limit)
        monkeypatch.setattr(regime_module, "INPUT_DEPTH", input_depth)
        for name, value in extra.items():
            monkeypatch.setattr(regime_module, name, value)

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


# id, shape, block_h limit, input_depth, read-side knobs: bank-major chunks that end mid-round or mid-unit,
# blocks whose page count is not a whole bank round, NoC shares other than 1/2, stagger, page-major.
FORCED_READ_PATHS = [
    ("bm_chunk1_odd_pages", (2, 8, 37, 7, 3, 3), 3, 2, {"BANK_RUN_ROUNDS": 1}),
    ("bm_chunk3_stagger", (1, 3, 141, 33, 2, 2), 2, 3, {"BANK_RUN_ROUNDS": 3, "BANK_STAGGER": True}),
    ("bm_alt1_4_i4_j1", (1, 3, 7, 24, 4, 1), 4, 2, {"READ_ALT_NOC_SHARE": Fraction(1, 4)}),
    (
        "bm_alt3_8_whole_block",
        (1, 32, 64, 32, 2, 2),
        5,
        2,
        {"READ_ALT_NOC_SHARE": Fraction(3, 8), "BANK_RUN_ROUNDS": None},
    ),
    ("pm_alt1_2_stagger", (2, 8, 37, 7, 3, 3), 2, 2, {"BANK_MAJOR": False, "BANK_STAGGER": True}),
    ("one_noc_bm", (1, 2, 3, 40, 4, 8), 1, 2, {"READ_ALT_NOC_SHARE": Fraction(0)}),
]


@pytest.mark.parametrize(
    "shape, block_limit, input_depth, extra", [c[1:] for c in FORCED_READ_PATHS], ids=[c[0] for c in FORCED_READ_PATHS]
)
def test_forced_read_paths_bit_exact(shape, block_limit, input_depth, extra, device, knobs):
    knobs(block_limit, input_depth, **extra)
    torch.manual_seed(11)
    _permute_and_check(_randn(shape), device)


@pytest.fixture
def coalesced_knobs(monkeypatch):
    def force(block_limit, staging_depth, **extra):
        monkeypatch.setattr(regime_module, "READ_MODE", "coalesced")
        monkeypatch.setattr(regime_module, "COALESCED_BLOCK_PAGES", None)
        monkeypatch.setattr(regime_module, "BLOCK_H_LIMIT", block_limit)
        monkeypatch.setattr(regime_module, "STAGING_DEPTH", staging_depth)
        for name, value in extra.items():
            monkeypatch.setattr(regime_module, name, value)

    op_module._PROGRAM_CACHE.clear()
    yield force
    op_module._PROGRAM_CACHE.clear()


# id, shape, block_h limit, staging depth, other knobs. Blocks that start mid bank round (pages per unit
# not a multiple of the bank count) and ragged last blocks, fewer pages than banks, one page per slot
# word and pages longer than one batch, uneven NoC shares, stagger, the page-by-page copies.
FORCED_COALESCED = [
    ("odd_j_odd_w_mid_round", (1, 3, 5, 7, 3, 3), 2, 2, {}),
    ("ragged_crosses_nc_depth1", (2, 8, 37, 7, 3, 3), 3, 1, {"COALESCED_BANK_STAGGER": True}),
    ("i4_j1_two_byte_pages", (1, 3, 7, 24, 4, 1), 4, 2, {"COALESCED_ALT_NOC_SHARE": Fraction(1, 4)}),
    ("j36_long_pages", (1, 1, 2, 64, 2, 36), 1, 2, {"COALESCED_ALT_NOC_SHARE": Fraction(3, 8)}),
    ("w5_fewer_pages_than_banks", (1, 2, 9, 5, 1, 2), 1, 2, {"COALESCED_ALT_NOC_SHARE": Fraction(0)}),
    ("flux73_page_by_page", (1, 32, 64, 32, 2, 2), 5, 2, {"COMPACT_BATCH_WORDS": 1, "GATHER_PAIR_BATCH": 1}),
    ("flux73_odd_batches", (1, 32, 64, 32, 2, 2), 3, 1, {"COMPACT_BATCH_WORDS": 3, "GATHER_PAIR_BATCH": 3}),
    ("j8_w40_batches", (1, 2, 3, 40, 4, 8), 1, 2, {"COMPACT_BATCH_WORDS": 8, "GATHER_PAIR_BATCH": 5}),
]


@pytest.mark.parametrize(
    "shape, block_limit, staging_depth, extra",
    [c[1:] for c in FORCED_COALESCED],
    ids=[c[0] for c in FORCED_COALESCED],
)
def test_forced_coalesced_bit_exact(shape, block_limit, staging_depth, extra, device, coalesced_knobs):
    coalesced_knobs(block_limit, staging_depth, **extra)
    torch.manual_seed(13)
    torch_input = _randn(shape)
    plan = g6_permute_plan(_to_device(torch_input, device), BLOCKS)
    assert plan.coalesced and plan.regime == "blocks_rm_bank_coalesced"
    assert plan.block_h <= block_limit
    _permute_and_check(torch_input, device)


@pytest.mark.parametrize("shape", [(1, 32, 64, 32, 2, 2), (1, 3, 5, 7, 3, 3)], ids=["flux73", "odd_j_odd_w"])
def test_coalesced_in_l1_bit_exact(shape, device):
    """L1 input reads whole slots too (16 B slots, one bank per worker), in blocks of the L1 knob."""
    op_module._PROGRAM_CACHE.clear()
    torch.manual_seed(17)
    torch_input = _randn(shape)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    plan = g6_permute_plan(ttnn_input, BLOCKS)
    assert plan.regime == "blocks_rm_bank_coalesced"
    assert plan.read_alt_share == 0
    pages_per_unit = shape[3] * shape[4]
    block_target = -(-regime_module.COALESCED_BLOCK_PAGES[ttnn.BufferType.L1] // pages_per_unit)
    assert plan.block_h == min(plan.units_per_core_g1, plan.block_cap, block_target)
    actual = ttnn.to_torch(g6_permute(ttnn_input, BLOCKS))
    op_module._PROGRAM_CACHE.clear()
    _assert_bit_exact(actual, torch.permute(torch_input, BLOCKS).contiguous())
