# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The device-indexed session slot arguments of the KV cache ops.

DeepSeek V4.1 decode keeps several sessions resident and names the step's session on device, so one
captured trace serves them all:

* ``ttnn.fill_cache(..., input_row_offset_tensor=, num_input_rows=)`` copies ``num_input_rows`` rows
  of the input starting at the row the INT32 ``[1]`` tensor holds (a session's ring slot);
* ``ttnn.experimental.paged_update_cache(..., page_table_row_tensor=)`` pages a one-user update
  through row ``r`` of an ``[N, L]`` page table (a session's row).

Run::

    pytest tests/ttnn/unit_tests/operations/transformers/test_cache_session_slots.py
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.experimental.deepseek_v4_flash.tt.decode.attention import _one_row_per_user


def _index(value: int, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _tile(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.parametrize("slot", [0, 3])
@pytest.mark.parametrize("cache_rows, rows", [(640, 128), (32, 32)], ids=["ring_into_sel_tile", "window"])
def test_fill_cache_reads_row_offset(device, slot, cache_rows, rows) -> None:
    """Rows ``[slot * rows, (slot + 1) * rows)`` of the input land in cache rows ``[0, rows)``; the
    cache's other rows keep their contents."""
    torch.manual_seed(0)
    num_slots, dim = 4, 512
    pool = torch.randn(1, 1, num_slots * rows, dim).to(torch.bfloat16)
    prior = torch.randn(1, 1, cache_rows, dim).to(torch.bfloat16)
    cache = _tile(prior, device)
    ttnn.fill_cache(
        cache, _tile(pool, device), 0, input_row_offset_tensor=_index(slot * rows, device), num_input_rows=rows
    )
    expected = prior.clone()
    expected[:, :, :rows] = pool[:, :, slot * rows : (slot + 1) * rows]
    assert torch.equal(ttnn.to_torch(cache).to(torch.bfloat16), expected)


def test_fill_cache_row_offset_is_read_at_run_time(device) -> None:
    """One program, two offsets: rewriting the offset tensor in place moves the copied rows, as a replayed
    trace sees a new session's slot."""
    torch.manual_seed(0)
    rows, dim = 128, 512
    pool = torch.randn(1, 1, 2 * rows, dim).to(torch.bfloat16)
    pool_tt = _tile(pool, device)
    offset = _index(0, device)
    cache = _tile(torch.zeros(1, 1, rows, dim), device)
    for slot in (0, 1):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([slot * rows], dtype=torch.int32), dtype=ttnn.int32), offset
        )
        ttnn.fill_cache(cache, pool_tt, 0, input_row_offset_tensor=offset, num_input_rows=rows)
        got = ttnn.to_torch(cache).to(torch.bfloat16)
        assert torch.equal(got, pool[:, :, slot * rows : (slot + 1) * rows]), f"slot {slot} rows not copied"


@pytest.mark.parametrize("session", [0, 2])
def test_paged_update_cache_page_table_row(device, session) -> None:
    """A one-user update at logical row ``entry`` goes through row ``session`` of an ``[N, L]`` table:
    physical block ``table[session, entry // block]``, row ``entry % block``; nothing else changes."""
    torch.manual_seed(0)
    num_sessions, blocks_per_session, block, dim = 3, 8, 64, 128
    num_blocks = num_sessions * blocks_per_session
    table = torch.randperm(num_blocks).to(torch.int32).reshape(num_sessions, blocks_per_session)
    prior = torch.randn(num_blocks, 1, block, dim).to(torch.bfloat16)
    cache = _tile(prior, device)
    row = torch.randn(1, 1, 1, dim).to(torch.bfloat16)
    entry = 5 * block + 17
    update = ttnn.from_torch(row, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn.experimental.paged_update_cache(
        cache,
        _one_row_per_user(update),
        update_idxs_tensor=_index(entry, device),
        page_table=ttnn.from_torch(
            table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        ),
        page_table_row_tensor=_index(session, device),
    )
    expected = prior.clone()
    expected[int(table[session, entry // block]), 0, entry % block] = row.reshape(-1)
    assert torch.equal(ttnn.to_torch(cache).to(torch.bfloat16), expected)
