# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of V4.1's multi-session book-keeping: no device is opened.

* The per-ratio page-table groups (64-row blocks, no ring) and their shared pool sizes.
* Blocks come out ascending, so a fresh session's prompt is one contiguous run of ``comp_kv`` blocks.
* The step packet names the session's slot: ring row, page-table row and window row.

Run::

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_session_pools.py
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.deepseek_v41_flash.tt.model import _WIN_SLOT_ROWS, DeepSeekV41Model
from models.experimental.deepseek_v4_flash.tt.decode.paged_cache import (
    ZERO_BLOCK,
    PagedCacheFull,
    PagedGroup,
    PagedKVManager,
    plan_pool_blocks,
)

BLOCK = 64
MAX_SEQ = 524288


def _groups() -> dict[str, PagedGroup]:
    return {
        f"ratio {r}": PagedGroup(
            layer_type=f"ratio {r}", block_size=BLOCK, sliding_window=0, max_seq=MAX_SEQ, compress_rate=r
        )
        for r in (1, 2)
    }


def test_ratio_groups_cover_max_seq() -> None:
    groups = _groups()
    assert groups["ratio 1"].logical_blocks == MAX_SEQ // BLOCK
    assert groups["ratio 2"].logical_blocks == MAX_SEQ // 2 // BLOCK
    assert all(g.ring_blocks == 0 for g in groups.values())
    # A step at pos that closes a group writes entry pos // r, so the blocks it needs must cover that row.
    for g in groups.values():
        r = g.compress_rate
        for pos in (r - 1, BLOCK * r - 1, BLOCK * r + r - 1, MAX_SEQ - 1):
            assert g.compressed_blocks_for(pos) * BLOCK > pos // r, (g.layer_type, pos)


@pytest.mark.parametrize("sessions", [1, 4])
def test_pools_hold_a_full_session_and_the_shared_budget(sessions) -> None:
    groups = _groups()
    plan = plan_pool_blocks(groups, sessions, sessions * MAX_SEQ)
    for name, g in groups.items():
        assert plan[name] >= g.logical_blocks + 1, "a page-table row must not outgrow its pool"
        assert plan[name] <= 1 + sessions * g.logical_blocks


def test_shared_budget_admits_one_full_session_and_refuses_more(expect_error) -> None:
    groups = _groups()
    paged = PagedKVManager(groups, plan_pool_blocks(groups, 2, MAX_SEQ))
    a, b = paged.open_session(), paged.open_session()
    paged.ensure_capacity(a, MAX_SEQ - 1)
    with expect_error(PagedCacheFull, r"need \d+ blocks, \d+ free of"):
        paged.ensure_capacity(b, MAX_SEQ - 1)
    paged.reset_session(a)
    paged.ensure_capacity(b, MAX_SEQ - 1)


def test_fresh_session_blocks_are_contiguous() -> None:
    groups = _groups()
    paged = PagedKVManager(groups, plan_pool_blocks(groups, 2, 2 * MAX_SEQ))
    sid = paged.open_session()
    prompt = 65536
    paged.ensure_capacity(sid, prompt - 1)
    for name, g in groups.items():
        row = paged.page_row(sid, name)[0].tolist()
        used = g.compressed_blocks_for(prompt - 1)
        assert row[:used] == list(range(1, used + 1)), f"{name}: blocks not one ascending run"
        assert all(b == ZERO_BLOCK for b in row[used:])


@pytest.mark.parametrize("slot, pos", [(0, 5), (3, 1000)])
def test_packet_names_the_slot(slot, pos) -> None:
    model = object.__new__(DeepSeekV41Model)
    model.sliding_window, model._decode_batch = 128, 1
    model._pkt_int_prefix, model._pkt_w = 6, 16
    model._packet_slot = slot
    packet = model._build_packet(7, pos)
    assert packet.shape == (1, 1, 1, 16) and packet.dtype == torch.int32
    w = model.sliding_window
    assert packet[0, 0, 0, :6].tolist() == [7, slot * w + pos % w, pos, slot, slot * w, slot * _WIN_SLOT_ROWS]
    assert not packet[0, 0, 0, 6:].any()
