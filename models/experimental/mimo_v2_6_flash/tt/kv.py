# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Paged-KV bookkeeping for one TP=8 replica: page tables and which chip owns which user.

Every table maps (slot, logical block) to a physical block, where logical block ``b`` holds positions
``b * block_size .. (b + 1) * block_size - 1``. The two layer groups use the two layouts of vLLM's hybrid
KV cache:

* Global layers (``FullAttentionSpec``): every logical block of a user's context is mapped. They have 4 KV
  heads over 8 chips, so chips ``2g`` and ``2g + 1`` both serve KV head ``g``. Instead of storing that
  head twice, the pair splits the users: slot ``s`` belongs to the chip with ``chip % rep == s % rep``
  (``rep = tp / num_kv_heads``). Only the owner keeps a slot's KV; the other chip's table points every
  block of that slot at one scratch block, so its writes land nowhere useful. In decode it sees position
  -1 for the slots it does not own, which the cache update and SDPA skip; in decode and prefill its
  attention output for those slots is zeroed (``owner_mask``) before the row all-reduce.
* SWA layers (``SlidingWindowSpec``): only blocks that overlap the 128-token window have to be mapped.
  vLLM frees the blocks behind the window and points their entries at its null block. Standalone, each
  user reuses its own physical blocks cyclically (``sliding_window_page_tables``). Either way the
  attention kernels never read a key outside the window, so stale or null blocks are never used.
  One KV head per chip, so the table is the same on every chip.

A cache never has fewer physical blocks than its table has entries per user: ``paged_update_cache``
rejects a table wider than the pool. vLLM keeps that bound by trimming its tables to the pool size; the
standalone SWA pool is sized up to the table width for the same reason (``layer_page_tables``).
"""

import math

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


def swa_blocks_per_user(sliding_window, max_chunk, block_size):
    """Physical blocks one user needs on a SWA layer: a prefill chunk of up to ``max_chunk`` tokens starting
    on a block boundary, plus the window of keys before it. Decode needs fewer (the window plus one block)."""
    return math.ceil(max_chunk / block_size) + math.ceil(sliding_window / block_size) + 1


def sliding_window_page_tables(num_users, max_batch, logical_blocks, physical_per_user, tp, seed=None):
    """SWA tables, int32 [tp, max_batch, logical_blocks], and blocks per chip.

    Logical block ``b`` of slot ``s`` maps to the slot's ``b % physical_per_user``-th physical block. Any
    ``physical_per_user`` consecutive logical blocks are distinct, so a chunk's writes never land on the
    window it is about to read (see ``swa_blocks_per_user``)."""
    num_blocks = num_users * physical_per_user
    order = torch.arange(num_blocks, dtype=torch.int32)
    if seed is not None:
        order = order[torch.randperm(num_blocks, generator=torch.Generator().manual_seed(seed))]
    tables = torch.zeros((tp, max_batch, logical_blocks), dtype=torch.int32)
    cyclic = torch.arange(logical_blocks) % physical_per_user
    for slot in range(num_users):
        tables[:, slot] = order[slot * physical_per_user + cyclic]
    return tables, num_blocks


def layer_page_tables(
    num_users, max_batch, max_seq_len, block_size, tp, rep, sliding_window=None, max_chunk=None, seed=None
):
    """Tables for one layer group: whole context for global layers, reused blocks for SWA layers.

    ``max_chunk``: the longest (padded) prefill chunk the caller will run. Each SWA user gets at least the
    blocks for one chunk plus the window, and at least its share of the table width, so the pool
    (users x blocks per user) is never smaller than the table is wide, as ``paged_update_cache`` requires.
    With 4 users and 4096-token chunks, that share only matters past ~17K tokens of context; at 1M it costs
    ~17 GB per chip over 39 layers, so 1M runs go through vLLM's shared pool, not the standalone model."""
    logical = math.ceil(max_seq_len / block_size)
    if sliding_window is None:
        return page_tables(num_users, max_batch, logical, tp, rep, seed)
    assert rep == 1 and max_chunk is not None
    per_user = max(swa_blocks_per_user(sliding_window, max_chunk, block_size), math.ceil(logical / num_users))
    return sliding_window_page_tables(num_users, max_batch, logical, min(per_user, logical), tp, seed)


def slot_owner_masks(tp, rep):
    """{parity: [tp, 1, 1, 1, 1]}: 1 on the chips that own slots of that parity (prefill: one user at a time)."""
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
