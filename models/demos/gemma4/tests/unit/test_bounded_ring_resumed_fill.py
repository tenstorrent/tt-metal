# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bounded sliding ring fill for a prefill chunk that resumes off the ring grid.

vLLM chunked prefill resumes a request at ``num_computed_tokens``. When several
prompts share a step, that offset is ragged (for example 1280 against a 2048-slot
ring). Decode reads position ``p`` from ring slot ``p % modulo``, so the deferred
prefill fill must write the resumed chunk's row ``r`` to slot
``(chunk_offset + r) % modulo``, leave the request's earlier in-window slots
alone, and keep the tile padding of its last tile from overwriting positions the
window still reads. These tests build the fill the way ``_prefill_forward_single``
stashes it, drive ``flush_deferred_bounded_fills`` and check every slot the next
decode step reads.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.attention.prefill import (
    _clone_sliding_prefill_tail,
    _restore_resumed_fill_padding,
    _ring_fill_page_table,
    flush_deferred_bounded_fills,
)

SLIDING_WINDOW = 128
BLOCK_SIZE = 64
HEAD_DIM = 64
NUM_KV_HEADS = 1
TILE = 32
# Ring blocks are scattered across a larger pool so a wrong page-table lookup
# lands in a block the ring does not own.
POOL_BLOCKS = [5, 2, 7, 0, 3, 6, 1, 4]
FIRST_LEN = 160  # first scheduler grant: positions [0, 160)
RESUME_AT = 128  # 160 floored to the 128-token SDPA alignment


def _to_device(device, rows):
    return ttnn.from_torch(rows.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _flush(k_cache, v_cache, page_table_tt, modulo, k_fill, v_fill, valid_seq_len, chunk_offset):
    pending = {
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_fill": k_fill,
        "v_fill": v_fill,
        "page_table": page_table_tt,
        "user_id": 0,
        "block_size": BLOCK_SIZE,
        "paged_modulo_kwargs": {"cache_position_modulo": modulo},
        "valid_seq_len": valid_seq_len,
        "modulo": modulo,
        "chunk_offset": chunk_offset,
    }
    layer = SimpleNamespace(self_attn=SimpleNamespace(config=SimpleNamespace(_deferred_bounded_fill=pending)))
    flush_deferred_bounded_fills([layer])


def _ring_slots(cache_tt, ring_blocks):
    """Cache contents in ring-slot order: slot ``s`` lives in ``ring_blocks[s // BLOCK_SIZE]``."""
    cache = ttnn.to_torch(cache_tt)
    return torch.cat([cache[b, 0] for b in ring_blocks], dim=0)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("extra_cols", [0, 2], ids=["ring_width", "wider_table"])
@pytest.mark.parametrize("modulo", [SLIDING_WINDOW, 2 * SLIDING_WINDOW], ids=["exact_ring", "ring_with_headroom"])
@pytest.mark.parametrize(
    "resumed_valid",
    [
        # Tile-aligned chunk; the window reaches back into the first chunk's slots.
        64,
        # Ends inside a tile: its padding rows take the slots of positions 65..95.
        65,
        # Positions [128, 328) wrap past the last slot.
        200,
    ],
    ids=["tile_aligned", "partial_last_tile", "wraps_the_ring"],
)
def test_resumed_chunk_keeps_the_decode_window(device, extra_cols, modulo, resumed_valid):
    torch.manual_seed(0)
    ring_blocks = POOL_BLOCKS[: modulo // BLOCK_SIZE]
    resumed_rows = -(-resumed_valid // TILE) * TILE  # the stash keeps whole tiles
    # K/V by absolute position. Rows past the resumed chunk's valid length stand
    # in for its tile padding and differ from what the ring holds in their slots.
    pos_k = torch.randn(RESUME_AT + resumed_rows, HEAD_DIM).bfloat16()
    pos_v = torch.randn(RESUME_AT + resumed_rows, HEAD_DIM).bfloat16()

    zeros = torch.zeros(len(POOL_BLOCKS), NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM).bfloat16()
    k_cache = ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    v_cache = ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    table = torch.tensor([ring_blocks + POOL_BLOCKS[len(ring_blocks) :][:extra_cols]], dtype=torch.int32)
    page_table_tt = ttnn.from_torch(table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    first_k = _to_device(device, pos_k[:FIRST_LEN])
    first_v = _to_device(device, pos_v[:FIRST_LEN])
    k_tail, v_tail = _clone_sliding_prefill_tail(first_k, first_v, SLIDING_WINDOW, HEAD_DIM, valid_seq_len=FIRST_LEN)
    _flush(k_cache, v_cache, page_table_tt, modulo, first_k, first_v, FIRST_LEN, 0)

    resumed = slice(RESUME_AT, RESUME_AT + resumed_rows)
    k_fill = _restore_resumed_fill_padding(
        _to_device(device, pos_k[resumed]), resumed_valid, RESUME_AT, modulo, k_tail, FIRST_LEN
    )
    v_fill = _restore_resumed_fill_padding(
        _to_device(device, pos_v[resumed]), resumed_valid, RESUME_AT, modulo, v_tail, FIRST_LEN
    )
    _flush(k_cache, v_cache, page_table_tt, modulo, k_fill, v_fill, resumed_valid, RESUME_AT)

    last = RESUME_AT + resumed_valid
    for name, cache_tt, ref in (("k", k_cache, pos_k), ("v", v_cache, pos_v)):
        slots = _ring_slots(cache_tt, ring_blocks)
        # Decode at position ``last`` reads its window from slots p % modulo.
        for p in range(max(0, last - SLIDING_WINDOW + 1), last):
            assert torch.equal(slots[p % modulo], ref[p]), f"{name}: slot {p % modulo} does not hold position {p}"


def test_ring_aligned_offset_keeps_the_table():
    sentinel = object()
    assert _ring_fill_page_table(sentinel, 0, 2 * SLIDING_WINDOW, BLOCK_SIZE) is sentinel
    assert _ring_fill_page_table(sentinel, 4 * SLIDING_WINDOW, 2 * SLIDING_WINDOW, BLOCK_SIZE) is sentinel


def test_offset_inside_a_block_is_refused(expect_error):
    with expect_error(ValueError, "whole number of 64-token blocks"):
        _ring_fill_page_table(object(), 2 * SLIDING_WINDOW + 32, 2 * SLIDING_WINDOW, BLOCK_SIZE)
