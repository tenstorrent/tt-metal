# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill caches for the hybrid model.

Full-attention layers (16 of 64): the canonical chunked-KV layout, copied from
``minimax_m3/tt/attention/kv_cache.py`` (the ring SDPA reads it, so it is not ours to change):

  per-chip shape      [num_users * num_attn_layers, 1, max_seq_len // sp, head_dim]
  slot packing        slot = user_id * num_attn_layers + attn_layer_ordinal   (user-major)
  DRAM memory config  NdShardSpec shard [1, 1, 32, head_dim], ROUND_ROBIN_1D over the DRAM banks
  sequence sharding   SP-sharded block-cyclic, period = the chunk the write was issued with
  allocation          zeroed, ReplicateTensorToMesh (content diverges on first write)
  write op            ttnn.experimental.deepseek_prefill.update_padded_kv_cache
  bank count          get_num_dram_banks(mesh_device)   (minimax hardcodes 8; the recipe wants the helper)

Only the 16 full-attention layers get K/V slots (``num_layers`` in the packing is the attention-layer
count): the 48 Gated-DeltaNet layers have no per-token cache at all.

Gated-DeltaNet layers (48 of 64): a fixed-size carried state per layer, which *is* this model's
"KV cache" for those layers (the golden trace stores exactly these):

  recurrent_state  fp32 [num_users * Nv_local, Dk, Dv] per chip  (12 value heads per TP column)
  conv_state       bf16 [num_users, 1, K-1, conv_dim_local] per chip (last K-1 pre-conv columns)

Both are TP-sharded (heads / channels on the columns) and identical across the SP rows, because every
SP row runs the whole chunk's recurrence (see ``tt/gdn.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

import ttnn
from models.common.utils import blockcyclic_positions
from models.demos.common.prefill.adapter import KvCaches
from models.demos.common.prefill.runners.migration import get_num_dram_banks

NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


@dataclass
class Qwen38Caches(KvCaches):
    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int  # full-attention layers (slot packing stride)
    max_seq_len: int
    sp: int
    head_dim: int
    # GDN carried state: {user_id: {model_layer_idx: {"recurrent": tensor | None, "conv": tensor | None}}}
    gdn: dict = field(default_factory=dict)

    def slot(self, user_id: int, attn_ordinal: int) -> int:
        return user_id * self.num_layers + attn_ordinal

    def gdn_state(self, user_id: int, layer_idx: int) -> dict:
        return self.gdn[user_id][layer_idx]

    def reset_gdn(self, user_id: int | None = None):
        """Drop the carried GDN state (a fresh sequence starts from zero state)."""
        users = self.gdn.keys() if user_id is None else [user_id]
        for u in users:
            for st in self.gdn[u].values():
                for key in list(st):
                    if st[key] is not None:
                        ttnn.deallocate(st[key])
                    st[key] = None


def cache_capacity(max_seq_len: int, periods) -> int:
    """Cache capacity in tokens: ``max_seq_len`` rounded up to a multiple of every block-cyclic period.

    ``update_padded_kv_cache`` requires ``seq_local % chunk_local == 0``, i.e. the capacity must be a
    whole number of chunks. The spec's max_seq_len (262144) is not a multiple of its chunk_size (5120),
    so the capacity is rounded up (to 266240 = 52 chunks); max_seq_len stays the contract bound.
    """
    step = 1
    for p in periods:
        step = step * p // math.gcd(step, p)
    return -(-max_seq_len // step) * step


def kv_memory_config(mesh_device, head_dim):
    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(get_num_dram_banks(mesh_device))
    ]
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)


def allocate_caches(
    mesh_device,
    *,
    num_attn_layers,
    gdn_layers,
    max_seq_len,
    head_dim=256,
    num_users=1,
    sp_axis=0,
    cache_dtype=ttnn.bfloat8_b,
) -> Qwen38Caches:
    """``max_seq_len`` here is the allocated capacity; callers pass ``cache_capacity(spec.max_seq_len, ...)``."""
    sp = mesh_device.shape[sp_axis]
    assert max_seq_len % (32 * sp) == 0, f"max_seq_len {max_seq_len} must be a multiple of 32*sp"
    seq_local = max_seq_len // sp
    mem = kv_memory_config(mesh_device, head_dim)

    def _alloc():
        return ttnn.from_torch(
            torch.zeros(num_users * max(num_attn_layers, 1), 1, seq_local, head_dim),
            dtype=cache_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return Qwen38Caches(
        k=_alloc(),
        v=_alloc(),
        num_users=num_users,
        num_layers=max(num_attn_layers, 1),
        max_seq_len=max_seq_len,
        sp=sp,
        head_dim=head_dim,
        gdn={u: {i: {"recurrent": None, "conv": None} for i in gdn_layers} for u in range(num_users)},
    )


def write_kv_chunk(caches: Qwen38Caches, tt_k, tt_v, *, slot_idx, layer_idx, kv_actual, sp_axis=0):
    """Write this chunk's post-RoPE K and raw V ([1, n_kv_local, s_local, head_dim] per chip)."""
    for cache, t in ((caches.k, tt_k), (caches.v, tt_v)):
        src = t if t.dtype == cache.dtype else ttnn.typecast(t, cache.dtype)
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            src,
            slot_idx=slot_idx,
            layer_idx=layer_idx,
            num_layers=caches.num_layers,
            kv_actual_global=kv_actual,
            cluster_axis=sp_axis,
        )
        if src is not t:
            ttnn.deallocate(src)


def read_attn_kv(caches: Qwen38Caches, mesh_device, *, user_id, attn_ordinal, n_tokens, period):
    """Read one layer's K/V back to host in natural order: ``[1, n_kv, n_tokens, head_dim]`` each.

    ``period`` is the block-cyclic period the chunks were written with (the chunk size; for a one-shot
    run, the one-shot length).
    """
    slot = caches.slot(user_id, attn_ordinal)
    # only the local rows the writes can have touched: whole periods covering n_tokens
    rows_local = -(-n_tokens // period) * (period // caches.sp)
    out = []
    for cache in (caches.k, caches.v):
        s = list(cache.shape)
        sl = ttnn.slice(
            cache, [slot, 0, 0, 0], [slot + 1, s[1], rows_local, s[3]], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # rows (SP) -> seq (dim 2), cols (TP) -> heads (dim 1)
        host = ttnn.to_torch(
            sl, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
        ).float()
        ttnn.deallocate(sl)
        out.append(naturalize(host, caches.sp, period, rows_local * caches.sp, n_tokens))
    return out


def naturalize(host, sp, period, max_seq_len, n_tokens):
    """Undo the block-cyclic layout of a composed cache read ``[.., sp*seq_local, D]`` -> ``[.., n_tokens, D]``.

    Row r of the composed read is chip ``r // seq_local``, local row ``lr = r % seq_local``, holding global
    position ``(lr // chunk_local) * period + c * chunk_local + lr % chunk_local`` (the writer's inverse,
    ``models.common.utils.blockcyclic_positions``). Rows past ``n_tokens`` are unwritten and dropped.
    """
    assert period % sp == 0
    if max_seq_len % period == 0:
        pos = blockcyclic_positions(sp, period, max_seq_len)
    else:  # one-shot period need not divide the capacity; same formula, computed directly
        seq_local, chunk_local = max_seq_len // sp, period // sp
        c = torch.arange(sp).repeat_interleave(seq_local)
        lr = torch.arange(seq_local).repeat(sp)
        pos = (lr // chunk_local) * period + c * chunk_local + lr % chunk_local
    keep = pos < n_tokens
    nat = torch.empty(*host.shape[:-2], n_tokens, host.shape[-1], dtype=host.dtype)
    nat[..., pos[keep], :] = host[..., keep, :]
    return nat
