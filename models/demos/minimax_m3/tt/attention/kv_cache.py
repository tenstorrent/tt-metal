# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass, field

import torch

import ttnn
from models.demos.common.prefill.adapter import KvCaches

# DRAM ND-shard geometry for the packed prefill KV cache — M3-local (decoupled from the DeepSeek substrate
# so they can diverge). The sequence is tiled into NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK-token blocks that
# round-robin across BH_NUM_DRAM_BANKS DRAM banks. The address-table builder must match both values.
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32
BH_NUM_DRAM_BANKS = 8


@dataclass
class MiniMaxKVCache(KvCaches):
    """M3's on-device prefill KV cache: three persistent, user-major packed device caches, each per-chip
    shape ``[num_users*num_layers, 1, seq_local, head_dim]`` on the DRAM ND-shard substrate, written in
    place by ``ttnn.experimental.deepseek_prefill.update_padded_kv_cache(slot_idx, layer_idx, ...)``:

      * ``k`` / ``v``  — GQA K/V. Under TP=cols each chip holds one head (heads sharded on the TP cols);
                         the sequence is SP-sharded block-cyclic on the ``sp`` rows.
      * ``index_k``    — MSA lightning-indexer key (one shared head, REPLICATED across the TP cols); only
                         the MSA layers populate it — dense layers leave their slots zeroed.

    Batch dim is user-major (``slot = user_id * num_layers + layer_idx``) so each user's layers stay
    contiguous, matching ``update_padded_kv_cache``'s indexing. The adapter allocates this once and the
    engine holds it as an opaque handle, passing it back into every runtime call that touches it.
    """

    k: ttnn.Tensor
    v: ttnn.Tensor
    index_k: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int

    # Device-valued slot metadata for request-mode tracing (populated only when num_users > 1). A captured
    # trace reads these tensors by address, so set_read_user re-targets a user's slot in place without
    # recapture; `_slot_frozen` blocks the host update during capture (a host copy inside a trace is illegal).
    #   _read_slot_start — cache-read partition-slice begin [slot,0,0,0], one per layer. `_read_slot_end` is a
    #                      constant companion (the reader ignores its value).
    #   _write_slot      — KV-write user slot (update_padded_kv_cache tensor form); one scalar, all layers.
    #   _write_kv_actual — KV-write prior-length scalar, one per distinct depth (bucket).
    _read_slot_start: dict = field(default_factory=dict, repr=False)
    _read_slot_end: object = field(default=None, repr=False)
    _write_slot: object = field(default=None, repr=False)
    _write_kv_actual: dict = field(default_factory=dict, repr=False)
    _slot_frozen: bool = field(default=False, repr=False)

    def _begin_index_tensor(self, values, mesh_device):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _meta_scalar(self, val, mesh_device):
        # 1-element uint32 replicated-DRAM scalar; update_padded_kv_cache's tensor form reads element [0].
        return ttnn.from_torch(
            torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    @staticmethod
    def _host_scalar(val):
        return ttnn.from_torch(
            torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def read_slot_start(self, layer_idx, slot, mesh_device):
        """Persistent [slot,0,0,0] begin tensor for `layer_idx`, reused across chunks/users. Re-targets to
        `slot` in place unless frozen — during capture the warm forward's value must be read as-is."""
        t = self._read_slot_start.get(layer_idx)
        if t is None:
            self._read_slot_start[layer_idx] = t = self._begin_index_tensor([slot, 0, 0, 0], mesh_device)
        elif not self._slot_frozen:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.tensor([slot, 0, 0, 0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                ),
                t,
            )
        return t

    def read_slot_end(self, max_rows, head_dim, mesh_device):
        if self._read_slot_end is None:
            self._read_slot_end = self._begin_index_tensor(
                [self.num_users * self.num_layers, 1, max_rows, head_dim], mesh_device
            )
        return self._read_slot_end

    def write_slot_tensor(self, slot_idx, mesh_device):
        """Persistent user-slot scalar for the traceable KV write. Updated in place unless frozen."""
        if self._write_slot is None:
            self._write_slot = self._meta_scalar(slot_idx, mesh_device)
        elif not self._slot_frozen:
            ttnn.copy_host_to_device_tensor(self._host_scalar(slot_idx), self._write_slot)
        return self._write_slot

    def write_kv_actual_tensor(self, kv_actual, mesh_device):
        """Persistent prior-KV-length scalar, keyed by depth (bucket), so it is never updated after creation —
        each bucket reads its own depth and is shared across users (no set_read_user)."""
        t = self._write_kv_actual.get(kv_actual)
        if t is None:
            self._write_kv_actual[kv_actual] = t = self._meta_scalar(kv_actual, mesh_device)
        return t

    def set_read_user(self, user_id):
        """Point every device-valued slot tensor at `user_id` (read begins + the KV-write slot). Call before
        replaying a captured trace for a different user (host update outside the trace). kv_actual is not
        touched — depth is a per-bucket constant shared across users."""
        for layer_idx, t in self._read_slot_start.items():
            slot = user_id * self.num_layers + layer_idx
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.tensor([slot, 0, 0, 0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                ),
                t,
            )
        if self._write_slot is not None:
            ttnn.copy_host_to_device_tensor(self._host_scalar(user_id), self._write_slot)


def allocate_kv_caches(
    mesh_device,
    *,
    num_layers,
    max_seq_len,
    sp_axis=0,
    num_users=1,
    head_dim=128,
    cache_dtype=ttnn.bfloat8_b,
) -> MiniMaxKVCache:
    """Allocate the three external prefill KV caches (K, V, index_k). See :class:`MiniMaxKVCache`.

    Deliberately NOT ``init_kvpe_cache`` (that is MLA-specific and allocates a single cache): this owns
    the M3 GQA triple and the user-major packing. It reuses the same DRAM NdShard spec (same bank grid +
    32-token contiguous shard) so ``update_padded_kv_cache`` can write into these tensors unchanged.

    Args:
        num_layers: layers per user (full model = 60). All three caches allocate all layers; only the MSA
            layers will write ``index_k`` (dense slots stay zeroed — capacity is cheap, packing stays simple).
        max_seq_len: per-user cache capacity in tokens, a multiple of ``sp``. ``seq_local = max_seq_len // sp``.
        sp_axis: mesh axis the sequence is sharded over (rows).
        num_users: independent user slots sharing the cache (1 for bring-up).
        head_dim: per-head width (128 for M3 main K/V and the index head alike).
        cache_dtype: on-device cache dtype (bf8 matches the DeepSeek substrate + the device golden check).
    """
    sp = mesh_device.shape[sp_axis]
    assert max_seq_len % sp == 0, f"max_seq_len ({max_seq_len}) must be divisible by sp ({sp})"
    seq_local = max_seq_len // sp

    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(BH_NUM_DRAM_BANKS)
    ]
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    def _alloc(dtype=cache_dtype):
        # Per-chip cache is one head ([.., 1, ..]); WHICH head a chip holds (or whether index_k is
        # replicated across cols) is decided at write time by how the input chunk is mesh-mapped, not
        # here. Allocated zeroed + ReplicateTensorToMesh: every chip gets the same empty buffer; content
        # diverges on the first update_padded_kv_cache write.
        return ttnn.from_torch(
            torch.zeros(num_users * num_layers, 1, seq_local, head_dim),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # index_k feeds the indexer's HARD top-16 block selection (not a smooth softmax like K/V), so bf8's
    # ~2-3 mantissa bits perturb the block scores enough to flip many picks -> chunked vs one-shot
    # selection diverges (~7/16 overlap) -> residual drift compounding over MSA layers. Cache it in bf16
    # (M3_INDEX_CACHE_BF16=1) to keep selection stable; it's tiny (1 head) and only the indexer reads it.
    index_dtype = ttnn.bfloat16 if os.getenv("M3_INDEX_CACHE_BF16") == "1" else cache_dtype

    return MiniMaxKVCache(
        k=_alloc(),
        v=_alloc(),
        index_k=_alloc(index_dtype),
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp=sp,
    )


def _write_one(kv_cache, cache, tensor, *, slot_idx, layer_idx, num_layers, kv_actual, sp_axis):
    """Write one SP-sharded chunk tensor into a packed cache via update_padded_kv_cache.

    The op requires TILE layout and input.dtype == cache.dtype, so cast a copy to the cache's dtype when
    needed (the original stays live for the attention op that follows). At ``kv_actual % 32 == 0`` chunk
    boundaries the per-device write offset is contiguous (block-cyclic degenerates to a reshape).

    With more than one user this takes the op's traceable tensor form: slot/kv_actual are read on-device
    from persistent scalars (kv_cache), so a captured trace re-targets the write slot per user.
    """
    src = tensor if tensor.dtype == cache.dtype else ttnn.typecast(tensor, cache.dtype)
    if kv_cache.num_users > 1:
        mesh_device = cache.device()
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            src,
            kv_cache.write_slot_tensor(slot_idx, mesh_device),
            kv_cache.write_kv_actual_tensor(kv_actual, mesh_device),
            layer_idx=layer_idx,
            num_layers=num_layers,
            cluster_axis=sp_axis,
        )
    else:
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            src,
            slot_idx=slot_idx,
            layer_idx=layer_idx,
            num_layers=num_layers,
            kv_actual_global=kv_actual,
            cluster_axis=sp_axis,
        )
    if src is not tensor:
        src.deallocate(True)


def write_kv_chunk(kv_cache: MiniMaxKVCache, tt_k, tt_v, *, slot_idx, layer_idx, kv_actual, sp_axis):
    """Write this chunk's post-RoPE K and raw V into the packed cache (every layer type).

    tt_k / tt_v are the per-device SP shards [1, n_kv_local, s_local, head_dim] (heads TP-sharded on the
    cols, sequence SP-sharded on the ``sp_axis`` rows) — exactly the per-chip cache layout, so they write
    in place. ``kv_actual`` is the cumulative valid prefix before this chunk (0 for non-chunked).
    """
    _write_one(
        kv_cache,
        kv_cache.k,
        tt_k,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=kv_cache.num_layers,
        kv_actual=kv_actual,
        sp_axis=sp_axis,
    )
    _write_one(
        kv_cache,
        kv_cache.v,
        tt_v,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=kv_cache.num_layers,
        kv_actual=kv_actual,
        sp_axis=sp_axis,
    )


def write_index_k_chunk(kv_cache: MiniMaxKVCache, tt_index_k, *, slot_idx, layer_idx, kv_actual, sp_axis):
    """Write this chunk's post-norm/post-RoPE MSA index_k (MSA layers only).

    tt_index_k is the single shared index head [1, 1, s_local, head_dim], SP-sharded on the rows and
    REPLICATED across the TP cols (so each col writes the same data into its replicated cache slot).
    """
    _write_one(
        kv_cache,
        kv_cache.index_k,
        tt_index_k,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=kv_cache.num_layers,
        kv_actual=kv_actual,
        sp_axis=sp_axis,
    )
