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
