# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The standalone draft call's host inputs for several blocks a rank (speculative decode: the served c32
round), as plain tensors: no ttnn, so the arithmetic is testable on a
workstation (``tests/v41/test_dspark_blocks.py``).

With the drafter's rings per slot (``TtDSparkDrafter(users=U)``: each stage's KV buffer holds ``U`` rings of
``ring_slots`` rows, user ``u``'s at rows ``u ring_slots ..``, then the block tile at ``U ring_slots ..``) one
draft call proposes for up to ``U`` slots of a rank at once: block ``a`` (rows ``5 a .. 5 a + 4`` of the tile) is
``[token, noise x 4]`` at positions ``p + 1 .. p + 5`` of its slot's user ``u``, its index rows list the user's
ring slots of positions ``p - valid + 1 .. p`` (``valid = min(128, p + 1)``: the reference's
``get_dspark_topk_idxs``), then the block's own five rows; and before the block runs, the taps of the slot's
verify rows just committed (``taps``: ``(tile row of the verify step's tap tile, position)``) are written to the
user's ring at ``position % ring_slots`` through the tile rows ``5 a + i`` (the ``sel`` selector picks the tap
row, ``ring_idx`` the ring row, ``main_pos`` the RoPE position; the sentinel on every other row). Every committed
position's tap reaches the ring exactly once (a ring missing the positions before the
accepted row lowers the acceptance).

An entry is a dict: ``user`` (the slot's user index on its rank), ``token`` (the block's first token: the
slot's next token ``x[p + 1]``), ``position`` (``p``: the slot's new frontier minus one) and ``taps`` (the
list above, at most 5 a block; empty on a slot's first call after its prefill, whose seeding wrote the ring).
``rank_inputs`` takes the entries of one rank; ``mesh_inputs`` stacks the ranks' along a leading ``sp`` axis.
"""

from __future__ import annotations

import torch

ROWS = 32  # the decode tile
RING = 128  # the reference's window
BLOCK = 5  # dspark_block_size
NOISE = 128799  # dspark_noise_token_id
SENTINEL = -1  # int32 host value of the uint32 sentinel
INDEX_WIDTH = -(-(RING + BLOCK) // 32) * 32  # 160


def rank_inputs(entries: list[dict], ring_slots: int, users: int) -> dict:
    """The tensors of one rank's draft call (``ROWS`` rows; ``len(entries) <= min(users, 6)`` blocks):
    ``ids [ROWS]`` int32, ``sel [ROWS, ROWS]`` float (row ``r``: a one at the tap row it writes, or zeros),
    ``ring_idx [ROWS]`` int32 (the ring row the tap goes to, the sentinel for no write), ``main_pos [ROWS]``
    int64 (the written tap's position; 0 elsewhere), ``block_pos [ROWS]`` int64 (the block rows' positions; the
    rows past the blocks block 0's continuation), ``index [ROWS, INDEX_WIDTH]`` int32 (the sentinel past the
    valid entries)."""
    block_base = users * ring_slots
    assert len(entries) * BLOCK <= ROWS, len(entries)
    ids = torch.full((ROWS,), NOISE, dtype=torch.int32)
    sel = torch.zeros(ROWS, ROWS)
    ring_idx = torch.full((ROWS,), SENTINEL, dtype=torch.int32)
    main_pos = torch.zeros(ROWS, dtype=torch.int64)
    block_pos = torch.arange(1, ROWS + 1, dtype=torch.int64)  # block 0's continuation at p = 0
    index = torch.full((ROWS, INDEX_WIDTH), SENTINEL, dtype=torch.int32)
    for a, e in enumerate(entries):
        u, p = int(e["user"]), int(e["position"])
        assert 0 <= u < users and p >= 0, (u, users, p)
        taps = list(e.get("taps", ()))
        assert len(taps) <= BLOCK, taps
        r0 = BLOCK * a
        ids[r0] = int(e["token"])
        for i, (tap_row, q) in enumerate(taps):
            assert 0 <= tap_row < ROWS and 0 <= q <= p, (tap_row, q, p)
            sel[r0 + i, tap_row] = 1.0
            ring_idx[r0 + i] = u * ring_slots + q % ring_slots
            main_pos[r0 + i] = q
        valid = min(RING, p + 1)
        if ring_slots == RING:  # the reference's slots ``0 .. valid - 1`` (the written ones, in slot order)
            row = [u * ring_slots + s for s in range(valid)]
        else:  # position-based slots: the positions p - valid + 1 .. p
            row = [u * ring_slots + q % ring_slots for q in range(p - valid + 1, p + 1)]
        row += list(range(block_base + r0, block_base + r0 + BLOCK))
        index[r0 : r0 + BLOCK, : len(row)] = torch.tensor(row, dtype=torch.int32)
        if a == 0:
            block_pos = torch.arange(p + 1, p + 1 + ROWS, dtype=torch.int64)
            # the rows past the blocks (noise rows, never read): block 0's window alone, as the fused call's,
            # never an all-sentinel row
            index[BLOCK * len(entries) :, :valid] = torch.tensor(row[:valid], dtype=torch.int32)
    for a, e in enumerate(entries):
        p = int(e["position"])
        block_pos[BLOCK * a : BLOCK * (a + 1)] = torch.arange(p + 1, p + 1 + BLOCK)
    return {"ids": ids, "sel": sel, "ring_idx": ring_idx, "main_pos": main_pos, "block_pos": block_pos, "index": index}


def mesh_inputs(entries_by_rank: list[list[dict]], ring_slots: int, users: int) -> dict:
    """The ranks' ``rank_inputs`` stacked: ``ids [sp, ROWS]``, ``sel [sp, ROWS, ROWS]``, ``ring_idx [sp, ROWS]``,
    ``main_pos [sp, ROWS]``, ``block_pos [sp, ROWS]``, ``index [sp, ROWS, INDEX_WIDTH]``."""
    ranks = [rank_inputs(entries, ring_slots, users) for entries in entries_by_rank]
    return {k: torch.stack([r[k] for r in ranks]) for k in ranks[0]}


def read_drafts(ids: torch.Tensor, blocks: int) -> list[list[int]]:
    """The draft call's output row (``[ROWS]``: row ``5 a + i`` block ``a``'s draft ``i + 1``) -> per block its
    five drafts."""
    flat = ids.flatten().tolist()
    return [[int(v) for v in flat[BLOCK * a : BLOCK * (a + 1)]] for a in range(blocks)]
