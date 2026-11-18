# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Paged-KV bookkeeping for one TP=8 replica: page tables and which chip owns which user.

Global layers have 4 KV heads over 8 chips, so chips ``2g`` and ``2g + 1`` both serve KV head
``g``. Instead of storing that head twice, the pair splits the users: decode slot ``s`` belongs
to the chip with ``chip % rep == s % rep`` (``rep = tp / num_kv_heads``). Each chip only keeps
the KV of the slots it owns, and in decode it sees position -1 for the other slots, which the
cache update and SDPA skip. Prefill computes K/V on both chips; the non-owner's page table
points every block of a foreign slot at one scratch block, so its writes land nowhere useful.

Layers with ``rep == 1`` (SWA: one KV head per chip) use the same table on every chip.

SWA layers page by absolute position as well, like vLLM's SlidingWindowSpec block tables (``SwaBlockTable``):
block j of a slot holds positions [j * block_size, (j + 1) * block_size), but only the blocks some sliding window
can still read are real; the others point at a scratch block.

Under vLLM a global block can be shared by requests in slots of either parity (prefix caching), so the pair cannot
split the users. With MIMO_KV_PAIR_SPLIT (the default) the pair splits every block instead: vLLM's global table has
one block per 128 positions, and each chip keeps its 64-position half of it (reference/kv_pair_split.py). A prefill
chunk's fill table then has one entry per 64 positions, the block on the chip that keeps them and a scratch block on
the other (``pair_fill_rows``).
"""

import math

import torch

import ttnn

from ..reference import kv_pair_split


def kv_replication(tp, num_kv_heads):
    return tp // num_kv_heads if tp > num_kv_heads else 1


def owned_slots(num_users, rep, parity):
    return [s for s in range(num_users) if s % rep == parity]


def page_tables(num_users, max_batch, blocks_per_user, tp, rep, seed=None):
    """Per-chip slot -> physical-block tables, int32 [tp, max_batch, blocks_per_user], and blocks per chip.

    ``seed`` shuffles the physical blocks (tests use it to exercise non-contiguous pages)."""
    n_owned = max(len(owned_slots(num_users, rep, q)) for q in range(rep))
    num_blocks = n_owned * blocks_per_user
    order = torch.arange(num_blocks, dtype=torch.int32)
    if seed is not None:
        order = order[torch.randperm(num_blocks, generator=torch.Generator().manual_seed(seed))]
    scratch = num_blocks if rep > 1 else 0
    tables = torch.full((tp, max_batch, blocks_per_user), scratch, dtype=torch.int32)
    for chip in range(tp):
        for k, slot in enumerate(owned_slots(num_users, rep, chip % rep)):
            tables[chip, slot] = order[k * blocks_per_user : (k + 1) * blocks_per_user]
    return tables, num_blocks + (1 if rep > 1 else 0)


def slot_owner_masks(tp, rep):
    """{parity: [tp, 1, 1, 1, 1]}: 1 on the chips that own the slots of that parity (prefill: one user at a time)."""
    chips = torch.arange(tp)
    return {p: (chips % rep == p).to(torch.bfloat16).reshape(tp, 1, 1, 1, 1) for p in range(rep)}


def owned_positions(positions, tp, rep):
    """positions [B] (-1 = empty) -> per-chip positions [tp, B] with -1 on the slots a chip does not own."""
    slots = torch.arange(positions.numel())
    out = positions.to(torch.int32).reshape(1, -1).repeat(tp, 1)
    for chip in range(tp):
        out[chip, slots % rep != chip % rep] = -1
    return out


def owner_mask(positions, tp, rep):
    """[tp, 1, 1, B, 1]: 1 where the chip owns an active slot. Zeroes the skipped slots' attention output."""
    return (owned_positions(positions, tp, rep) >= 0).to(torch.bfloat16).reshape(tp, 1, 1, -1, 1)


def per_chip(plan, t, dtype, layout, device=True):
    """Put ``t`` [tp, ...] on the replica so chip i gets ``t[i]`` (row of the 1 x tp mesh)."""
    flat = t.reshape(t.shape[0] * t.shape[1], *t.shape[2:]) if t.dim() > 1 else t
    return ttnn.from_torch(
        flat,
        device=plan.mesh_device if device else None,
        dtype=dtype,
        layout=layout,
        mesh_mapper=plan.shard(0),
    )


def replicated(plan, t, dtype, device=True):
    """Row-major ``t`` on every chip of the replica (``device`` False: the host tensor, to copy into one)."""
    return ttnn.from_torch(
        t,
        device=plan.mesh_device if device else None,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=plan.replicate(),
    )


def pair_fill_rows(table, slot, start, padded, scratch):
    """Pair split: table int32 [tp, max_batch, W] (vLLM's global table, one block per 128 positions, the same on every
    chip) -> [tp, 1, ceil(padded / 64)]: the fill table of ``slot``'s chunk of ``padded`` tokens from ``start`` on
    each chip (reference.kv_pair_split.fill_table: its own 64-position halves, ``scratch`` elsewhere)."""
    rows = [
        kv_pair_split.fill_table(table[chip, slot], start, padded, kv_pair_split.parity(chip), scratch)
        for chip in range(table.shape[0])
    ]
    return torch.tensor(rows, dtype=torch.int32).reshape(table.shape[0], 1, -1)


def block_rows(table, slot, first, count, scratch):
    """table int32 [tp, max_batch, W] -> [tp, 1, count]: blocks first .. first + count - 1 of ``slot`` on every chip,
    ``scratch`` past the end of the table (None: the table has no scratch block, so nothing may run past it)."""
    n = max(0, min(count, table.shape[2] - first))
    assert scratch is not None or n == count, f"blocks {first}.. {first + count} run past a table with no scratch block"
    out = torch.full((table.shape[0], 1, count), 0 if scratch is None else scratch, dtype=torch.int32)
    out[:, 0, :n] = table[:, slot, first : first + n]
    return out


class PrefillTables:
    """The block tables one prefill chunk of one user reads through one page table, as small device tensors made per
    chunk (and released once it is enqueued). Their shapes depend only on the chunk's padded length, so no prefill
    program keys on the user, the chunk's start or a block id (attention.py):

    * ``fill`` int32 [1, ceil(padded / block_size)] per chip: the blocks of positions start .. start + padded - 1;
    * ``context`` (a later chunk through a global layer's table) int32 [1, context_blocks] per chip: every block from
      position 0, for chunked MLA prefill, whose kernel reads only those before the chunk's end;
    * ``window`` (a later chunk through a SWA table): one uint32 [4] slice start (block, 0, 0, 0) per block of
      positions [window_start, start) (kv.SwaBlockTable.window_blocks), in position order;
    * ``start`` (a later chunk): int32 [1], the chunk's start position, shared by every table of the chunk.

    ``table``: the host copy of the page table, int32 [tp, max_batch, W]. Entries past its width point at ``scratch``
    (a block nothing reads: the padding of a chunk at the end of the context lands there).

    ``pair_split`` (a global table under MIMO_KV_PAIR_SPLIT, one block per 2 * block_size positions): ``fill`` holds
    each chip's halves (``pair_fill_rows``, still one entry per ``block_size`` positions) and ``context`` the table's
    blocks from position 0, one per 2 * block_size positions (each chip reads its local halves through it)."""

    def __init__(
        self,
        plan,
        table,
        slot,
        start,
        padded,
        block_size,
        scratch,
        context_blocks=None,
        window=None,
        start_tensor=None,
        pair_split=False,
    ):
        fill, context = self.host_rows(table, slot, start, padded, block_size, scratch, context_blocks, pair_split)
        self.fill = per_chip(plan, fill, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.context = None if context is None else per_chip(plan, context, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.window = [
            replicated(plan, torch.tensor([b, 0, 0, 0], dtype=torch.int32), ttnn.uint32) for b in (window or [])
        ]
        self.start = start_tensor

    @staticmethod
    def host_rows(table, slot, start, padded, block_size, scratch, context_blocks=None, pair_split=False):
        """(fill [tp, 1, ceil(padded / block_size)], context [tp, 1, context_blocks] or None): the host values of
        ``fill`` and ``context`` (see the class)."""
        assert not pair_split or block_size == kv_pair_split.SUB, block_size
        span = 2 * block_size if pair_split else block_size  # positions per table entry
        assert start % span == 0, (start, span)
        if pair_split:
            fill = pair_fill_rows(table, slot, start, padded, scratch)
        else:
            fill = block_rows(table, slot, start // block_size, -(-padded // block_size), scratch)
        context = None
        if context_blocks is not None:
            assert context_blocks * span >= start + padded, (context_blocks, start, padded)
            context = block_rows(table, slot, 0, context_blocks, scratch)
        return fill, context

    def release(self):
        """Free the tensors this object made (``start`` belongs to the caller)."""
        for t in [self.fill, self.context, *self.window]:
            if t is not None:
                t.deallocate(True)


class ChunkTables:
    """The PrefillTables of a traced later chunk (TtMiMoModel.setup_prefill_chunk_traces): the same tensors, made once
    (every block a scratch block) and rewritten in place before each replay (``write``). ``start`` is the traced
    chunk's shared start tensor."""

    def __init__(self, plan, padded, block_size, scratch, context_blocks=None, window_blocks=0, start_tensor=None):
        self.plan = plan
        tp = plan.tp
        fill = torch.full((tp, 1, -(-padded // block_size)), scratch, dtype=torch.int32)
        self.fill = per_chip(plan, fill, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.context = None
        if context_blocks is not None:
            context = torch.full((tp, 1, context_blocks), scratch, dtype=torch.int32)
            self.context = per_chip(plan, context, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.window = [
            replicated(plan, torch.tensor([scratch, 0, 0, 0], dtype=torch.int32), ttnn.uint32)
            for _ in range(window_blocks)
        ]
        self.start = start_tensor

    def write(self, fill, context=None, window=None):
        """Copy a chunk's host rows (PrefillTables.host_rows; ``window`` its window blocks, in order) in place."""
        plan = self.plan
        ttnn.copy_host_to_device_tensor(per_chip(plan, fill, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, False), self.fill)
        if self.context is not None:
            host = per_chip(plan, context, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, False)
            ttnn.copy_host_to_device_tensor(host, self.context)
        assert len(window or []) == len(self.window), (window, len(self.window))
        for b, t in zip(window or [], self.window):
            host = replicated(plan, torch.tensor([b, 0, 0, 0], dtype=torch.int32), ttnn.uint32, False)
            ttnn.copy_host_to_device_tensor(host, t)


class BatchFillTables:
    """The KV blocks a batch of first chunks writes through one page table (TtMiMoAttention.forward_prefill with
    ``users`` R: R users, each from position 0 and padded to the same length S): ``fill`` int32 [R, ceil(S /
    block_size)] per chip, row r the blocks of user r's positions 0 .. S - 1 (``batch_fill_rows``), and ``rows``
    uint32 [R] = 0 .. R - 1 (shared by the batch's tables; the caller frees it), the row of ``fill`` each input
    batch entry of the batched ``paged_fill_cache`` writes through. The shapes depend only on R and S."""

    def __init__(self, fill, rows):
        self.fill = fill
        self.rows = rows

    def release(self):
        self.fill.deallocate(True)


def batch_fill_rows(table, slots, padded, block_size, scratch, pair_split=False):
    """table int32 [tp, max_batch, W] -> [tp, R, ceil(padded / block_size)]: the first blocks of each slot's row
    (``pair_split``: each chip's halves of the global table's first blocks, ``pair_fill_rows``)."""
    if pair_split:
        return torch.cat([pair_fill_rows(table, slot, 0, padded, scratch) for slot in slots], dim=1)
    count = -(-padded // block_size)
    return torch.cat([block_rows(table, slot, 0, count, scratch) for slot in slots], dim=1)


# Most padding a prefill chunk gets (model.prefill_padded_len rounds up to 32, then to 128 or 256 tokens).
PREFILL_MAX_PAD = 255


def swa_blocks_per_user(window, block_size, max_pad=PREFILL_MAX_PAD):
    """Real SWA blocks one user can hold at once (``SwaBlockTable``), for any chunk size: a later prefill chunk keeps
    the window before it (``window / block_size`` blocks: chunks start on a block boundary) and takes the blocks of
    [real end - window, padded end), at most ``ceil((window + max_pad) / block_size) + 1``. Decode needs fewer (the
    window plus the block being written)."""
    return math.ceil(window / block_size) + math.ceil((window + max_pad) / block_size) + 1


class SwaBlockTable:
    """Host-side block table of the SWA layers (shared by them: every SWA layer's cache uses the same block ids).

    Entry [slot, j] is the physical block of positions [j * block_size, (j + 1) * block_size) of that slot, indexed
    by absolute position (no ring). Block 0 is a scratch block: entries no window can read point at it, so writes
    there (a long prompt's early positions) land in it and nothing reads them back. Each user owns a pool of
    ``blocks_per_user`` real blocks: one is taken when a position in its range is about to be written, and it goes
    back once every later window starts past it. ``live`` keeps each slot's logical blocks that hold a real block,
    so the bookkeeping per step is bounded by the pool, not by the context length. ``dirty`` marks changes not yet
    copied to the device.

    ``min_blocks``: the cache has at least this many blocks (the extra ones are never mapped). ``paged_update_cache``
    rejects a page table wider than the cache, so the SWA layers' ttnn decode path (not SwaDecodeQkv) needs the pool
    to be at least as wide as the table."""

    def __init__(self, num_users, max_batch, max_blocks, block_size, window, blocks_per_user, min_blocks=0):
        self.block_size, self.window = block_size, window
        self.table = torch.zeros(max_batch, max_blocks, dtype=torch.int32)
        self.num_blocks = max(1 + num_users * blocks_per_user, min_blocks)
        self.free = {
            u: list(range(1 + (u + 1) * blocks_per_user - 1, u * blocks_per_user, -1)) for u in range(num_users)
        }
        self.live = {u: set() for u in range(num_users)}
        self.dirty = True

    def _take(self, slot, lo, hi):
        """Real blocks for positions [lo, hi) of ``slot``."""
        for j in range(lo // self.block_size, (hi - 1) // self.block_size + 1):
            if j not in self.live[slot]:
                assert self.free[slot], f"slot {slot}: no free SWA block for position {j * self.block_size}"
                self.table[slot, j] = self.free[slot].pop()
                self.live[slot].add(j)
                self.dirty = True

    def _release(self, slot, blocks):
        for j in blocks:
            self.free[slot].append(int(self.table[slot, j]))
            self.table[slot, j] = 0
            self.live[slot].discard(j)
            self.dirty = True

    def _release_below(self, slot, pos):
        """Give back the blocks of ``slot`` that end before the window of position ``pos`` starts."""
        first = max(0, pos - self.window + 1) // self.block_size
        self._release(slot, sorted(j for j in self.live[slot] if j < first))

    def release_all(self, slot):
        """Give back every block of ``slot`` (its user is done)."""
        self._release(slot, sorted(self.live[slot]))

    def prefill(self, slot, start, real_end, padded_end):
        """Before a prefill chunk of ``slot`` writes positions [start, padded_end), real up to ``real_end``. Only
        [real_end - window, padded_end) can be read after it (the next chunk's or decode step's window), so only
        those blocks become real. A chunk at ``start`` 0 begins a new user; a later chunk keeps the window before
        ``start``, which it reads (``window_blocks``). Padding past the end of the table stays in the scratch block
        (PrefillTables maps those positions there)."""
        if start == 0:
            self.release_all(slot)  # the slot's previous user
        else:
            self._release_below(slot, start)
            missing = [b for b in self.window_blocks(slot, start) if b == 0]
            assert not missing, f"slot {slot}: the window before position {start} is not in the cache"
        self._take(slot, max(0, real_end - self.window), min(padded_end, self.table.shape[1] * self.block_size))

    def window_start(self, start):
        """First position of the block-aligned keys a chunk at ``start`` reads from the cache."""
        return max(0, start - self.window) // self.block_size * self.block_size

    def window_blocks(self, slot, start):
        """Physical blocks of positions [window_start(start), start) of ``slot``, in order."""
        first, end = self.window_start(start) // self.block_size, start // self.block_size
        return [int(b) for b in self.table[slot, first:end]]

    def advance(self, slot, pos):
        """Before a decode step writes position ``pos`` of ``slot``."""
        self._release_below(slot, pos)
        self._take(slot, pos, pos + 1)

    def host_table(self, tp):
        """[tp, max_batch, max_blocks] int32 (the same table on every chip)."""
        return self.table.unsqueeze(0).repeat(tp, 1, 1)
