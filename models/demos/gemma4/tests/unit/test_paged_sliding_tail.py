# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for the paged sliding-window tail of a cached-prefix continuation: which blocks are read,
how null blocks are handled, and which mesh shard holds the owner lane's page-table row."""
from types import SimpleNamespace

import torch

from models.demos.gemma4.tt.attention.prefill import _owner_lane_device_index, _paged_tail_block_range

W, BS, POOL = 1024, 64, 4096


def _row(ids):
    return torch.tensor(ids, dtype=torch.int32)


def test_full_window_inside_the_hit():
    row = _row(list(range(1, 21)))  # blocks 1..20 for tokens [0, 1280)
    assert _paged_tail_block_range(row, 1280, W, BS, POOL) == (256, list(range(5, 21)))


def test_short_prefix_takes_everything_cached():
    row = _row([7, 8, 9, 10, 0, 0])
    assert _paged_tail_block_range(row, 256, W, BS, POOL) == (0, [7, 8, 9, 10])


def test_alignment_past_the_kept_window_drops_the_null_block():
    # vLLM kept 16 real blocks for a hit at 1088 tokens; the SDPA alignment moved the offset to 1024, so the
    # window [0, 1024) starts one block before the kept range, which vLLM already nulled (block 0).
    row = _row([0] + list(range(2, 18)) + [0, 0])
    start, blocks = _paged_tail_block_range(row, 1024, W, BS, POOL)
    assert start == 64 and blocks == list(range(2, 17))


def test_unusable_rows_give_none():
    assert _paged_tail_block_range(_row([0, 0, 0, 0]), 256, W, BS, POOL) is None  # all null
    assert _paged_tail_block_range(_row([1, 2, 3, POOL + 5]), 256, W, BS, POOL) is None  # out of pool
    assert _paged_tail_block_range(_row([1, 2, 3, 4]), 200, W, BS, POOL) is None  # unaligned offset
    assert _paged_tail_block_range(_row([1, 2, 3, 4]), 0, W, BS, POOL) is None  # first chunk


def test_owner_lane_shard_index():
    assert _owner_lane_device_index(None) == 0
    assert _owner_lane_device_index(SimpleNamespace(lane_sharded=False)) == 0
    lanes_on_axis_1 = SimpleNamespace(lane_sharded=True, mesh_shape=(8, 4), sp_axis=1, _g4_single_lane_prefill_owner=2)
    assert _owner_lane_device_index(lanes_on_axis_1) == 2
    lanes_on_axis_0 = SimpleNamespace(lane_sharded=True, mesh_shape=(2, 4), sp_axis=0, _g4_single_lane_prefill_owner=1)
    assert _owner_lane_device_index(lanes_on_axis_0) == 4
    lane_parallel = SimpleNamespace(lane_sharded=True, mesh_shape=(8, 4), sp_axis=1, _g4_single_lane_prefill_owner=None)
    assert _owner_lane_device_index(lane_parallel) == 0
