# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_permute regime nhwc_rm_transpose (Refinement 4).

Cell: bfloat16, ROW_MAJOR, DRAM or L1 interleaved, rank 4, dims (0, 3, 1, 2):
out[n, c, h, w] = x[n, h, w, c]. Graded bit for bit. Besides Panoptic p4, the regime must hold for
every shape validate() accepts: rows cut into units with a tail (W not a multiple of the unit, W not
a multiple of 8), whole-row units narrower than 8 values, odd C, input pages (C * 2 B) that are not
multiples of 16 B or are longer than one 64 B DRAM slot, N > 1, every alignment tag.
"""

import importlib
from fractions import Fraction

import pytest
import torch
import ttnn

from ttnn.operations.g6_permute import g6_permute, g6_permute_plan

op_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute")
regime_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute_nhwc_rm_transpose")

NHWC = (0, 3, 1, 2)
MEMORY_CONFIG = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}
BUFFER_TYPE = {"dram": ttnn.BufferType.DRAM, "l1": ttnn.BufferType.L1}

PANOPTIC_P4 = (1, 128, 256, 32)

# id, shape [N, H, W, C], buffer
CELL_SHAPES = [
    ("odd_c_w_tail", (1, 3, 100, 3), "dram"),  # 6 B input pages, W not a multiple of 8
    ("c5_l1", (1, 4, 40, 5), "l1"),  # 10 B input pages
    ("c1", (1, 2, 64, 1), "dram"),  # 2 B input pages
    ("w5_whole_rows", (2, 7, 5, 32), "l1"),  # rows narrower than one write granule
    ("c64_long_page", (1, 8, 48, 64), "dram"),  # 128 B input pages: two DRAM slots
    ("batch2_tile_aligned", (2, 32, 64, 32), "l1"),
    ("h_non_aligned_c64", (1, 5, 40, 64), "dram"),  # tag h_non_aligned (the tagger's H is W = 40)
    ("w_non_aligned", (1, 6, 33, 70), "l1"),  # tag w_non_aligned, odd W
    ("one_long_row", (1, 1, 1000, 16), "dram"),  # one row cut over many cores
    ("many_rows", (1, 200, 96, 24), "l1"),
]

SPECIAL_BITS = (0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x8000, 0x0001, 0x807F, 0x0040)


def _randn(shape):
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)


def _to_device(torch_input, device, buffer):
    return ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=MEMORY_CONFIG[buffer],
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


def _permute_and_check(torch_input, device, buffer, call_dims=NHWC):
    expected = torch.permute(torch_input, NHWC).contiguous()
    ttnn_output = g6_permute(_to_device(torch_input, device, buffer), call_dims)
    assert list(ttnn_output.shape) == list(expected.shape)
    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT
    memory_config = ttnn_output.memory_config()
    assert memory_config.buffer_type == BUFFER_TYPE[buffer]
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    actual = ttnn.to_torch(ttnn_output)
    _assert_bit_exact(actual, expected)
    return actual


@pytest.mark.parametrize("buffer", ["l1", "dram"])
def test_panoptic_p4(buffer, device):
    torch.manual_seed(44)
    _permute_and_check(_randn(PANOPTIC_P4), device, buffer)


@pytest.mark.parametrize("shape, buffer", [c[1:] for c in CELL_SHAPES], ids=[c[0] for c in CELL_SHAPES])
def test_other_shapes_of_the_cell(shape, buffer, device):
    torch.manual_seed(42)
    _permute_and_check(_randn(shape), device, buffer)


def test_special_values_move_bit_for_bit(device):
    torch.manual_seed(42)
    shape = (1, 6, 40, 32)
    bits = _randn(shape).contiguous().view(torch.int16).reshape(-1).clone()
    special = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL_BITS], dtype=torch.int16)
    positions = torch.arange(0, bits.numel(), 3)
    bits[positions] = special[torch.arange(positions.numel()) % special.numel()]
    _permute_and_check(bits.view(torch.bfloat16).reshape(shape), device, "l1")


@pytest.mark.parametrize("buffer", ["l1", "dram"])
def test_every_bfloat16_pattern_moves_bit_for_bit(buffer, device):
    # The transpose runs through unpack and pack: every one of the 65536 patterns must survive.
    bits = torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32).to(torch.int16)
    torch_input = bits[torch.randperm(bits.numel(), generator=torch.Generator().manual_seed(3))]
    _permute_and_check(torch_input.view(torch.bfloat16).reshape(1, 2, 1024, 32), device, buffer)


def test_negative_dims_equal_canonical(device):
    torch.manual_seed(42)
    _permute_and_check(_randn((1, 6, 40, 32)), device, "dram", call_dims=(0, -1, 1, -2))


def test_plan_reports_regime(device):
    torch.manual_seed(42)
    plan = g6_permute_plan(_to_device(_randn(PANOPTIC_P4), device, "l1"), NHWC)
    assert plan.regime == "nhwc_rm_transpose"
    N, H, W, C = PANOPTIC_P4
    assert plan.block_w % 8 == 0 and plan.block_w <= W
    assert plan.units_per_row == -(-W // plan.block_w)
    assert plan.tensor_units == N * H * plan.units_per_row
    assert plan.num_cores == len(plan.assignment)
    assert sum(n for _, _, n in plan.assignment) == plan.tensor_units
    # L1 input: one NoC, page order.
    assert plan.read_alt_share == 0 and not plan.bank_major
    # The output's own L1 pages (p4: 38 x 512 B per core) count against the budget.
    assert plan.out_l1_bytes_per_core == 38 * 512
    assert plan.l1_bytes_per_core + plan.out_l1_bytes_per_core <= regime_module.L1_OP_BUDGET_BYTES


def test_repeat_call_reuses_program(device):
    torch.manual_seed(42)
    shape = (1, 6, 40, 32)
    first = _randn(shape)
    second = _randn(shape)
    out_first = _permute_and_check(first, device, "l1")
    entries = device.num_program_cache_entries()
    out_again = _permute_and_check(first, device, "l1")
    _permute_and_check(second, device, "l1")
    assert device.num_program_cache_entries() == entries
    _assert_bit_exact(out_again, out_first)


@pytest.fixture
def knobs(monkeypatch):
    def force(**values):
        for name, value in values.items():
            monkeypatch.setattr(regime_module, name, value)

    op_module._PROGRAM_CACHE.clear()
    yield force
    op_module._PROGRAM_CACHE.clear()


# id, shape, buffer, knob overrides: unit widths with and without a tail (sub-tile, multi-tile with a
# partial last w-tile, the 8-tile maximum, whole rows), every input depth, one rows page, and each DRAM
# read path of the shared reader.
FORCED_PLANS = [
    ("w8_depth3", (1, 4, 60, 32), "l1", dict(BLOCK_W=8, INPUT_DEPTH=3)),
    ("w24_tail", (2, 5, 100, 6), "dram", dict(BLOCK_W=24, INPUT_DEPTH=2)),
    ("whole_rows_depth1", (1, 30, 72, 16), "l1", dict(BLOCK_W=10**6, INPUT_DEPTH=1)),
    ("w40_partial_w_tile_rows1", (1, 6, 130, 40), "l1", dict(BLOCK_W=40, ROWS_DEPTH=1)),
    ("w256_eight_w_tiles", (1, 3, 600, 32), "l1", dict(BLOCK_W=256)),
    ("w16_page_order_one_noc", (1, 9, 50, 32), "dram", dict(BLOCK_W=16, BANK_MAJOR=False, READ_ALT_NOC_SHARE=0)),
    ("w32_bank_major_quarter_alt", (1, 9, 70, 32), "dram", dict(BLOCK_W=32, READ_ALT_NOC_SHARE=Fraction(1, 4))),
]


@pytest.mark.parametrize("shape, buffer, overrides", [c[1:] for c in FORCED_PLANS], ids=[c[0] for c in FORCED_PLANS])
def test_forced_plans_bit_exact(shape, buffer, overrides, device, knobs):
    knobs(**overrides)
    torch.manual_seed(7)
    torch_input = _randn(shape)
    plan = g6_permute_plan(_to_device(torch_input, device, buffer), NHWC)
    if overrides["BLOCK_W"] < shape[2]:
        assert plan.block_w == overrides["BLOCK_W"]
    else:
        assert plan.block_w == shape[2] and plan.units_per_row == 1
    _permute_and_check(torch_input, device, buffer)
