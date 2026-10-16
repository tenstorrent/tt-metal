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
"""

import torch

import ttnn


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


class SwaBlockTable:
    """Host-side block table of the SWA layers (shared by them: every SWA layer's cache uses the same block ids).

    Entry [slot, j] is the physical block of positions [j * block_size, (j + 1) * block_size) of that slot, indexed
    by absolute position (no ring). Block 0 is a scratch block: entries no window can read point at it, so writes
    there (a long prompt's early positions) land in it and nothing reads them back. Each user owns a pool of
    ``blocks_per_user`` real blocks: one is taken when a position in its range is about to be written, and it goes
    back once every later window starts past it. ``dirty`` marks changes not yet copied to the device."""

    def __init__(self, num_users, max_batch, max_blocks, block_size, window, blocks_per_user):
        self.block_size, self.window = block_size, window
        self.table = torch.zeros(max_batch, max_blocks, dtype=torch.int32)
        self.num_blocks = 1 + num_users * blocks_per_user
        self.free = {
            u: list(range(1 + (u + 1) * blocks_per_user - 1, u * blocks_per_user, -1)) for u in range(num_users)
        }
        self.dirty = True

    def _take(self, slot, lo, hi):
        """Real blocks for positions [lo, hi) of ``slot``."""
        for j in range(lo // self.block_size, (hi - 1) // self.block_size + 1):
            if self.table[slot, j] == 0:
                assert self.free[slot], f"slot {slot}: no free SWA block for position {j * self.block_size}"
                self.table[slot, j] = self.free[slot].pop()
                self.dirty = True

    def _release_below(self, slot, pos):
        """Give back the blocks of ``slot`` that end before the window of position ``pos`` starts."""
        first = max(0, pos - self.window + 1) // self.block_size
        for j in range(first):
            b = int(self.table[slot, j])
            if b != 0:
                self.free[slot].append(b)
                self.table[slot, j] = 0
                self.dirty = True

    def prefill(self, slot, real_len, padded_len):
        """Before a prefill of ``real_len`` tokens padded to ``padded_len``: the fill writes positions
        [0, padded_len); only [real_len - window, padded_len) can be read later (decode windows start there)."""
        self._release_below(slot, self.table.shape[1] * self.block_size)  # the slot's previous user, all of it
        self._take(slot, max(0, real_len - self.window), padded_len)

    def advance(self, slot, pos):
        """Before a decode step writes position ``pos`` of ``slot``."""
        self._release_below(slot, pos)
        self._take(slot, pos, pos + 1)

    def host_table(self, tp):
        """[tp, max_batch, max_blocks] int32 (the same table on every chip)."""
        return self.table.unsqueeze(0).repeat(tp, 1, 1)
