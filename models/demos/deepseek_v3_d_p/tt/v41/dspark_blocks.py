# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The standalone draft call's host inputs for several blocks a rank (speculative decode: the served c32
round), as plain tensors: no ttnn, so the arithmetic is testable on a
workstation (``tests/v41/test_dspark_blocks.py``).

With the drafter's rings per slot (``TtDSparkDrafter(users=U)``: each stage's KV buffer holds ``U`` rings of
``ring_slots`` rows, user ``u``'s at rows ``u ring_slots ..``, then the block tile at ``U ring_slots ..``) one
draft call proposes for up to ``U`` slots of a rank at once: block ``a`` (rows ``5 a .. 5 a + 4`` of the tile)
is
``[token, noise x 4]`` at positions ``p + 1 .. p + 5`` of its slot's user ``u``, its index rows list the
user's
ring slots of positions ``p - valid + 1 .. p`` (``valid = min(128, p + 1)``: the reference's
``get_dspark_topk_idxs``), then the block's own five rows; and before the block runs, the taps of the slot's
verify rows just committed (``taps``: ``(tile row of the verify step's tap tile, position)``) are written to
the
user's ring at ``position % ring_slots`` through the tile rows ``5 a + i`` (the ``sel`` selector picks the tap
row, ``ring_idx`` the ring row, ``main_pos`` the RoPE position; the sentinel on every other row). Every
committed
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
    int64 (the written tap's position; 0 elsewhere), ``block_pos [ROWS]`` int64 (the block rows' positions;
    the
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


def mesh_inputs_reference(entries_by_rank: list[list[dict]], ring_slots: int, users: int) -> dict:
    """The ranks' ``rank_inputs`` stacked: ``ids [sp, ROWS]``, ``sel [sp, ROWS, ROWS]``, ``ring_idx [sp,
    ROWS]``,
    ``main_pos [sp, ROWS]``, ``block_pos [sp, ROWS]``, ``index [sp, ROWS, INDEX_WIDTH]``."""
    ranks = [rank_inputs(entries, ring_slots, users) for entries in entries_by_rank]
    return {k: torch.stack([r[k] for r in ranks]) for k in ranks[0]}


def mesh_inputs(entries_by_rank: list, ring_slots: int, users: int) -> dict:
    """The ranks' ``rank_inputs`` stacked, in one pass over the ranks' entries (the per-rank
    loop was 4 ms a call on the host for eight ranks of four blocks; this is the same tensors, ~5x faster):
    ``ids [sp, ROWS]``, ``sel [sp, ROWS, ROWS]``, ``ring_idx [sp, ROWS]``, ``main_pos [sp, ROWS]``,
    ``block_pos [sp,
    ROWS]``, ``index [sp, ROWS, INDEX_WIDTH]``. ``mesh_inputs_reference`` is the per-rank form it must equal
    (``tests/v41/test_dspark_blocks.py``)."""
    sp = len(entries_by_rank)
    blocks = len(entries_by_rank[0])
    assert all(len(e) == blocks for e in entries_by_rank), [len(e) for e in entries_by_rank]
    assert blocks * BLOCK <= ROWS, blocks
    block_base = users * ring_slots
    # the entries as tensors [sp, blocks]
    user = torch.tensor([[int(e["user"]) for e in es] for es in entries_by_rank], dtype=torch.int64)
    pos = torch.tensor([[int(e["position"]) for e in es] for es in entries_by_rank], dtype=torch.int64)
    tok = torch.tensor([[int(e["token"]) for e in es] for es in entries_by_rank], dtype=torch.int64)
    assert bool(((user >= 0) & (user < users)).all()) and bool((pos >= 0).all()), (user, users, pos)
    ids = torch.full((sp, ROWS), NOISE, dtype=torch.int32)
    sel = torch.zeros(sp, ROWS, ROWS)
    ring_idx = torch.full((sp, ROWS), SENTINEL, dtype=torch.int32)
    main_pos = torch.zeros(sp, ROWS, dtype=torch.int64)
    index = torch.full((sp, ROWS, INDEX_WIDTH), SENTINEL, dtype=torch.int32)
    r0 = torch.arange(blocks) * BLOCK  # [blocks]
    ranks = torch.arange(sp)
    ids[ranks[:, None], r0[None, :]] = tok.to(torch.int32)
    # the taps: a flat list of (rank, row, tap_row, ring row, position)
    taps = []
    for r, es in enumerate(entries_by_rank):
        for a, e in enumerate(es):
            u, p = int(e["user"]), int(e["position"])
            ts = list(e.get("taps", ()))
            assert len(ts) <= BLOCK, ts
            for i, (tap_row, q) in enumerate(ts):
                assert 0 <= tap_row < ROWS and 0 <= q <= p, (tap_row, q, p)
                taps.append((r, BLOCK * a + i, tap_row, u * ring_slots + q % ring_slots, q))
    if taps:
        t = torch.tensor(taps, dtype=torch.int64)
        sel[t[:, 0], t[:, 1], t[:, 2]] = 1.0
        ring_idx[t[:, 0], t[:, 1]] = t[:, 3].to(torch.int32)
        main_pos[t[:, 0], t[:, 1]] = t[:, 4]
    # the window rows: block a's rows list the user's ring slots of positions p - valid + 1 .. p, then its own
    # rows
    valid = torch.clamp(pos + 1, max=RING)  # [sp, blocks]
    k = torch.arange(RING)  # column k of the window part
    if ring_slots == RING:
        slots = k[None, None, :].expand(sp, blocks, RING)
    else:
        slots = (pos[:, :, None] - valid[:, :, None] + 1 + k[None, None, :]) % ring_slots
    win = user[:, :, None] * ring_slots + slots  # [sp, blocks, RING]
    win = torch.where(k[None, None, :] < valid[:, :, None], win, torch.full_like(win, SENTINEL))
    own = block_base + r0[:, None] + torch.arange(BLOCK)[None, :]  # [blocks, BLOCK]
    # row 5 a + i of block a: the window (valid entries, then the sentinel) then the five own rows at valid ..
    rows = torch.full((sp, blocks, INDEX_WIDTH), SENTINEL, dtype=torch.int64)
    rows[:, :, :RING] = win
    col = valid[:, :, None] + torch.arange(BLOCK)[None, None, :]  # [sp, blocks, BLOCK]
    rows.scatter_(2, col, own[None].expand(sp, blocks, BLOCK))
    index[:, : BLOCK * blocks] = rows.repeat_interleave(BLOCK, dim=1).to(torch.int32)
    # the rows past the blocks: block 0's window alone
    if BLOCK * blocks < ROWS:
        index[:, BLOCK * blocks :, :RING] = win[:, 0:1, :].expand(sp, ROWS - BLOCK * blocks, RING).to(torch.int32)
    # the block positions: block a's rows at p_a + 1 .. p_a + 5; the rows past the blocks block 0's
    # continuation
    block_pos = pos[:, 0:1] + 1 + torch.arange(ROWS)[None, :]  # [sp, ROWS]
    bp = (pos[:, :, None] + 1 + torch.arange(BLOCK)[None, None, :]).reshape(sp, BLOCK * blocks)
    block_pos = torch.cat([bp, block_pos[:, BLOCK * blocks :]], dim=1)
    return {"ids": ids, "sel": sel, "ring_idx": ring_idx, "main_pos": main_pos, "block_pos": block_pos, "index": index}


def read_drafts(ids: torch.Tensor, blocks: int) -> list[list[int]]:
    """The draft call's output row (``[ROWS]``: row ``5 a + i`` block ``a``'s draft ``i + 1``) -> per block
    its
    five drafts."""
    flat = ids.flatten().tolist()
    return [[int(v) for v in flat[BLOCK * a : BLOCK * (a + 1)]] for a in range(blocks)]
