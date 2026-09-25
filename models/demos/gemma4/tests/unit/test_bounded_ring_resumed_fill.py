# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bounded sliding ring fill for a prefill chunk that resumes off the ring grid.

vLLM chunked prefill resumes a request at ``num_computed_tokens``. When several
prompts share a step, that offset is ragged (for example 1280 against a 2048-slot
ring). Decode reads position ``p`` from ring slot ``p % modulo``, so the deferred
prefill fill must write the resumed chunk's row ``r`` to slot
``(chunk_offset + r) % modulo`` and leave the request's earlier in-window slots
alone. These tests drive ``flush_deferred_bounded_fills`` the way
``_prefill_forward_single`` stashes it and check every ring slot.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.attention.prefill import _ring_fill_page_table, flush_deferred_bounded_fills

SLIDING_WINDOW = 128
MODULO = 2 * SLIDING_WINDOW  # window plus speculative headroom, as served
BLOCK_SIZE = 64
HEAD_DIM = 64
NUM_KV_HEADS = 1
# Ring blocks are scattered across a larger pool so a wrong page-table lookup
# lands in a block the ring does not own.
RING_BLOCKS = [5, 2, 7, 0]
POOL_BLOCKS = 8


def _fill(device, rows):
    return ttnn.from_torch(rows.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _flush(device, k_cache, v_cache, page_table_tt, k_rows, v_rows, valid_seq_len, chunk_offset):
    pending = {
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_fill": _fill(device, k_rows),
        "v_fill": _fill(device, v_rows),
        "page_table": page_table_tt,
        "user_id": 0,
        "block_size": BLOCK_SIZE,
        "paged_modulo_kwargs": {"cache_position_modulo": MODULO},
        "valid_seq_len": valid_seq_len,
        "modulo": MODULO,
        "chunk_offset": chunk_offset,
    }
    layer = SimpleNamespace(self_attn=SimpleNamespace(config=SimpleNamespace(_deferred_bounded_fill=pending)))
    flush_deferred_bounded_fills([layer])


def _ring_slots(cache_tt):
    """Cache contents in ring-slot order: slot ``s`` lives in ``RING_BLOCKS[s // BLOCK_SIZE]``."""
    cache = ttnn.to_torch(cache_tt)
    return torch.cat([cache[b, 0] for b in RING_BLOCKS], dim=0)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("extra_cols", [0, 2], ids=["ring_width", "wider_table"])
@pytest.mark.parametrize(
    "resumed_valid",
    [
        # Window [64, 192) reaches back into the first chunk's slots.
        64,
        # Positions [128, 328) wrap past the last slot; tile padding follows.
        200,
    ],
    ids=["window_spans_both_chunks", "chunk_wraps_the_ring"],
)
def test_resumed_chunk_lands_on_its_absolute_ring_slots(device, extra_cols, resumed_valid):
    torch.manual_seed(0)
    first_len = 160  # first scheduler grant: positions [0, 160)
    resume_at = 128  # 160 floored to the 128-token SDPA alignment
    resumed_rows = -(-resumed_valid // 32) * 32  # the stash keeps whole tiles

    pos_k = torch.randn(resume_at + resumed_rows, HEAD_DIM).bfloat16()
    pos_v = torch.randn(resume_at + resumed_rows, HEAD_DIM).bfloat16()

    zeros = torch.zeros(POOL_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM).bfloat16()
    k_cache = ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_cache = ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    table = torch.tensor([RING_BLOCKS + list(range(extra_cols))], dtype=torch.int32)
    page_table_tt = ttnn.from_torch(table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    _flush(device, k_cache, v_cache, page_table_tt, pos_k[:first_len], pos_v[:first_len], first_len, 0)
    _flush(
        device,
        k_cache,
        v_cache,
        page_table_tt,
        pos_k[resume_at : resume_at + resumed_rows],
        pos_v[resume_at : resume_at + resumed_rows],
        resumed_valid,
        resume_at,
    )

    last = resume_at + resumed_valid
    for name, cache_tt, ref in (("k", k_cache, pos_k), ("v", v_cache, pos_v)):
        slots = _ring_slots(cache_tt)
        # Decode at position ``last`` reads its window from slots p % MODULO.
        for p in range(last - SLIDING_WINDOW, last):
            got = slots[p % MODULO]
            assert torch.equal(got, ref[p]), f"{name}: slot {p % MODULO} does not hold position {p}"


def test_ring_aligned_offset_keeps_the_table():
    sentinel = object()
    assert _ring_fill_page_table(sentinel, 0, MODULO, BLOCK_SIZE) is sentinel
    assert _ring_fill_page_table(sentinel, 2 * MODULO, MODULO, BLOCK_SIZE) is sentinel


def test_offset_inside_a_block_is_refused(expect_error):
    with expect_error(ValueError, "whole number of 64-token blocks"):
        _ring_fill_page_table(object(), MODULO + 32, MODULO, BLOCK_SIZE)
