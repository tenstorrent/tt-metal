# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The block-cyclic position <-> (chip, slot) contract of tt/kv_layout.py. Pure CPU."""

import pytest
import torch

from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout


@pytest.mark.parametrize("num_chips, block", [(32, 128), (8, 128), (4, 32), (1, 128)])
def test_position_round_trip(num_chips, block):
    layout = BlockCyclicLayout(num_chips, block)
    positions = torch.arange(3 * num_chips * block + 17)
    chip, slot = layout.chip(positions), layout.slot(positions)
    assert torch.equal(layout.position(chip, slot), positions)
    assert chip.max() < num_chips
    # no two positions share a (chip, slot)
    assert torch.unique(chip * 10**9 + slot).numel() == positions.numel()
    assert layout.position(layout.chip(12345), layout.slot(12345)) == 12345


def test_blocks_are_dealt_round_robin():
    layout = BlockCyclicLayout()
    assert [layout.chip(p) for p in (0, 127, 128, 31 * 128, 32 * 128, 33 * 128 + 5)] == [0, 0, 1, 31, 0, 1]
    assert [layout.slot(p) for p in (0, 127, 128, 31 * 128, 32 * 128, 33 * 128 + 5)] == [0, 127, 0, 0, 128, 133]
    assert [layout.slots(n) for n in (1, 128, 129, 4096, 4097, 1_000_000)] == [128, 128, 128, 128, 256, 31360]


@pytest.mark.parametrize("length", [1, 100, 128, 4096, 5000, 10_000])
def test_split_merge_round_trip_and_even_load(length):
    layout = BlockCyclicLayout()
    x = torch.randn(3, length, 5, generator=torch.Generator().manual_seed(length))
    stripes = layout.split(x, dim=1)
    assert stripes.shape == (32, 3, layout.slots(length), 5)
    assert torch.equal(layout.merge(stripes, length, dim=1), x)
    filled = layout.filled(length)
    assert filled.sum() == length
    assert (stripes[~filled.unsqueeze(1).expand(-1, 3, -1)] == 0).all()
    per_chip = filled.sum(dim=1)
    assert per_chip.max() - per_chip.min() <= layout.block
