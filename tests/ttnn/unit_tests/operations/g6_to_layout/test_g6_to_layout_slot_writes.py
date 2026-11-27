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
