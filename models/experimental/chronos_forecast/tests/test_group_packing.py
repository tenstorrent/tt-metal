# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for group block packing and the data-parallel mask layout."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.experimental.chronos_forecast.tt.group_attention import pack_group_blocks
from models.experimental.chronos_forecast.tt.model import TtChronos


@pytest.mark.parametrize("num_chips", [1, 4, 8, 32])
@pytest.mark.parametrize(
    "group_sizes",
    [
        pytest.param([4] * 256, id="groups_of_4"),
        pytest.param([1, 2, 3, 4, 5, 6, 7] * 14, id="mixed"),
        pytest.param([33, 33, 33, 1], id="groups_of_33"),
    ],
)
def test_pack_group_blocks_whole_blocks_per_chip(num_chips, group_sizes):
    group_ids = torch.repeat_interleave(torch.arange(len(group_sizes)), torch.tensor(group_sizes))
    group_ids = group_ids[torch.randperm(group_ids.numel(), generator=torch.Generator().manual_seed(0))]
    packing = pack_group_blocks(group_ids, preferred_block=64, num_blocks_multiple=num_chips)

    assert packing.num_blocks % num_chips == 0
    assert packing.rows.numel() == packing.num_blocks * packing.block
    assert torch.equal(packing.rows[packing.output_rows], torch.arange(group_ids.numel()))
    block_of = packing.output_rows // packing.block
    for group in group_ids.unique():
        assert block_of[group_ids == group].unique().numel() == 1

    # Real series attend exactly within their group; dummies only to themselves.
    packed_group = torch.full((packing.rows.numel(),), -1, dtype=torch.long)
    packed_group[packing.output_rows] = group_ids
    packed_group = packed_group.reshape(packing.num_blocks, packing.block)
    visible = packing.mask[:, 0] == 0
    same = (packed_group[:, :, None] == packed_group[:, None, :]) & (packed_group[:, :, None] >= 0)
    eye = torch.eye(packing.block, dtype=torch.bool).expand_as(same)
    assert torch.equal(visible, same | eye)


@pytest.mark.parametrize("num_chips", [1, 4])
def test_time_major_group_mask_splits_per_chip(num_chips):
    blocks, seq_len, g = 8, 3, 32
    mask = torch.arange(blocks, dtype=torch.float32).reshape(blocks, 1, 1, 1).expand(blocks, 1, g, g)
    full = TtChronos._time_major_group_mask(SimpleNamespace(num_devices=num_chips), mask, seq_len)
    local_blocks = blocks // num_chips
    # A dim-0 split must give each chip the single-chip (time-major) layout of its own blocks.
    for chip, rows in enumerate(full.chunk(num_chips)):
        local = mask[chip * local_blocks : (chip + 1) * local_blocks]
        assert torch.equal(rows, local.repeat(seq_len, 1, 1, 1))
