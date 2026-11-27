# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""same_bank_slot_writes (R7): one tile column, 2 * W shorter than the output's page slot.

The production cells (286, 308) are tile-aligned DRAM outputs, where every block starts on bank 0
and holds a whole number of bank rounds. These off-production shapes pin the run arithmetic where
that does not hold: images with H % 32 != 0 inside a block, H < 32, fewer rows than banks, and L1
outputs (16-B and 32-B slots, many more banks). Every value is graded bit for bit.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_to_layout import g6_to_layout, select_plan

RM = ttnn.ROW_MAJOR_LAYOUT
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def _to_device(torch_input, device):
    return ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)


def _bits(t):
    return t.contiguous().view(torch.int16)


@pytest.mark.parametrize(
    "shape, out_memory_config, regime",
    [
        pytest.param((3, 50, 5), DRAM, "same_bank_slot_writes", id="h-ragged-images"),
        pytest.param((7, 5, 3), DRAM, "same_bank_slot_writes", id="h-below-tile"),
        pytest.param((1, 1, 2), DRAM, "same_bank_slot_writes", id="one-row"),
        pytest.param((2, 1000, 16), DRAM, "same_bank_slot_writes", id="w16-h-ragged"),
        pytest.param((1, 64, 300, 3), L1, "same_bank_slot_writes", id="l1-slot16"),
        pytest.param((1, 32, 600, 12), L1, "same_bank_slot_writes", id="l1-slot32"),
        pytest.param((1, 2, 100, 8), L1, "face_direct", id="l1-full-slot"),
    ],
)
@pytest.mark.parametrize("poison", [None, float("nan")], ids=["randn", "poison_nan"])
def test_slot_writes(device, shape, out_memory_config, regime, poison):
    torch.manual_seed(7)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_input = _to_device(torch_input, device)
    if poison is not None:
        ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, poison)
    assert select_plan(ttnn_input, RM, memory_config=out_memory_config)["regime"] == regime

    output = g6_to_layout(ttnn_input, RM, memory_config=out_memory_config)

    actual = ttnn.to_torch(output).reshape(shape)
    differ = _bits(actual) != _bits(torch_input)
    count = int(differ.sum())
    assert count == 0, f"{count} of {differ.numel()} values differ bitwise"


# The R7 reader's trid ring addresses each in-flight block's cb_left_faces page from the ring base
# (the CB write pointer lags the blocks issued ahead of it). Off-default read knobs make the ring wrap
# with several blocks in flight; a wrong landing page overwrites a block before it is built.
@pytest.mark.parametrize(
    "read_depth, read_depth_max, stagger",
    [(3, 3, False), (1, 4, False), (2, 4, True)],
    ids=["depth3", "adaptive4", "depth2-adaptive4-stagger"],
)
@pytest.mark.parametrize(
    "shape",
    [pytest.param((40, 1000, 7), id="h-ragged-12-blocks"), pytest.param((1, 1024, 1024, 3), id="c286")],
)
def test_read_ring_knobs(device, shape, read_depth, read_depth_max, stagger, monkeypatch):
    import sys

    import ttnn.operations.g6_to_layout.g6_to_layout_program_descriptor as pd

    op_mod = sys.modules["ttnn.operations.g6_to_layout.g6_to_layout"]
    monkeypatch.setattr(pd, "SLOT_READ_DEPTH", read_depth)
    monkeypatch.setattr(pd, "SLOT_READ_DEPTH_MAX", read_depth_max)
    monkeypatch.setattr(pd, "SLOT_FACES_DEPTH", read_depth_max + 2)
    monkeypatch.setattr(pd, "SLOT_READ_STAGGER", stagger)
    op_mod._PLAN_CACHE.clear()
    op_mod._PROGRAM_CACHE.clear()
    try:
        torch.manual_seed(11)
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
        ttnn_input = ttnn.fill_implicit_tile_padding(_to_device(torch_input, device), float("nan"))
        plan = select_plan(ttnn_input, RM)
        assert plan["regime"] == "same_bank_slot_writes"
        assert (plan["slot_read_depth"], plan["slot_read_depth_max"]) == (read_depth, read_depth_max)

        actual = ttnn.to_torch(g6_to_layout(ttnn_input, RM)).reshape(shape)
        differ = _bits(actual) != _bits(torch_input)
        assert int(differ.sum()) == 0, f"{int(differ.sum())} of {differ.numel()} values differ bitwise"
    finally:
        op_mod._PLAN_CACHE.clear()
        op_mod._PROGRAM_CACHE.clear()
