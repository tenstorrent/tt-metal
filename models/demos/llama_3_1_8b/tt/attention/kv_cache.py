# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The prefill KV cache: two packed device tensors on the DRAM ND-shard substrate.

The layout is **canonical by convention** across the prefill packages (recipe §5.1) because the
chunked ring SDPA reads it, so everything below is copied verbatim and then verified mechanically:

| Element | Value |
|---|---|
| Per-chip shape | ``[num_users * num_layers, n_kv_local, seq_local, head_dim]`` |
| Slot packing | ``slot = user_id * num_layers + layer_idx`` (user-major) |
| DRAM memory config | ``NdShardSpec``, shard ``[1, 1, 32, head_dim]``, ``ROUND_ROBIN_1D`` |
| Contiguous tokens per bank | 32 |
| Sequence sharding | SP-sharded block-cyclic; ``seq_local = max_seq_len // sp`` |
| Alignment | ``max_seq_len % (TILE_SIZE * sp) == 0`` |
| Allocation | zeroed, ``ReplicateTensorToMesh`` (content diverges on first write) |
| Write op | ``ttnn.experimental.deepseek_prefill.update_padded_kv_cache`` |
| Bank count | ``get_num_dram_banks(mesh_device)`` (imported, not hardcoded) |

**The one deviation, and why it is required.** Both source packages allocate ``[.., 1, ..]`` — one KV
head per chip — because GPT-OSS has 8 KV heads at TP=8 and MiniMax-M3 has 4 at TP=4. Llama-3.1-8B
has **8 KV heads at TP=4, i.e. 2 per chip**, and ``update_padded_kv_cache`` enforces
``cache_shape[1] == input_shape[1]``. So ``dim1`` is ``num_kv_heads // tp``, not the literal 1 in
the source. That is exactly the recipe's "shape-tuned, not structural" distinction: the 1 encodes
the *source's* head count, not the layout. Everything else — the shard spec, the 32-token bank
period, the slot arithmetic, the round-robin distribution — is unchanged, which is what keeps
``update_padded_kv_cache`` and the ring SDPA's cache reader working against it.

Four per-model decisions (recipe §5.2): **2** cache tensors (GQA -> ``k``, ``v``; not MLA's single
latent, not M3's third ``index_k``), ``head_dim`` **128** as-is, ``cache_dtype`` **bfloat8_b** from
the spec's ``dataformats.kv_cache.default``, and **no** auxiliary caches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.demos.common.prefill.runners.migration import get_num_dram_banks

# Must match the DRAM NdShard below and the block-cyclic address walk in
# models/common/utils.py::blockcyclic_positions, which is the inverse of the writer.
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


@dataclass
class LlamaKVCache:
    """Externally-owned packed K/V for the SP chunked-KV path.

    ``k`` holds post-RoPE keys in the device's Meta column order; ``v`` holds raw values. Both are
    per-chip ``[num_users*num_layers, n_kv_local, seq_local, head_dim]``, heads sharded across the TP
    columns at write time and the sequence SP-sharded block-cyclic on the rows.
    """

    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int
    n_kv_local: int
    head_dim: int


def allocate_kv_cache(
    mesh_device,
    *,
    num_layers: int,
    max_seq_len: int,
    num_kv_heads: int,
    tp: int,
    sp_axis: int = 0,
    num_users: int = 1,
    head_dim: int = 128,
    cache_dtype=ttnn.bfloat8_b,
) -> LlamaKVCache:
    sp = mesh_device.shape[sp_axis]
    assert max_seq_len % (ttnn.TILE_SIZE * sp) == 0, (
        f"max_seq_len ({max_seq_len}) must be a multiple of TILE_SIZE*sp ({ttnn.TILE_SIZE * sp}). This is the "
        f"block-cyclic addressing period of the KV table: a misaligned value corrupts addresses silently."
    )
    assert num_kv_heads % tp == 0, f"{num_kv_heads} kv heads do not split over tp={tp}"
    n_kv_local = num_kv_heads // tp
    seq_local = max_seq_len // sp

    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0)) for bank in range(get_num_dram_banks(mesh_device))
    ]
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    def _alloc():
        # WHICH kv heads a chip holds is decided at WRITE time by how the chunk is mesh-mapped, not
        # here: every chip is allocated the same zeroed buffer and the content diverges on the first
        # update_padded_kv_cache call.
        return ttnn.from_torch(
            torch.zeros(num_users * num_layers, n_kv_local, seq_local, head_dim),
            dtype=cache_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return LlamaKVCache(
        k=_alloc(),
        v=_alloc(),
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp=sp,
        n_kv_local=n_kv_local,
        head_dim=head_dim,
    )


def _write_one(cache, tensor, *, slot_idx, layer_idx, num_layers, kv_actual, sp_axis):
    """One SP-sharded chunk tensor into a packed cache.

    The op requires TILE layout and ``input.dtype == cache.dtype``, so a bf16 activation is cast to
    the bf8 cache dtype on a copy — the original stays live for the attention op that follows.
    """
    src = tensor if tensor.dtype == cache.dtype else ttnn.typecast(tensor, cache.dtype)
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


def write_kv_chunk(kv_cache: LlamaKVCache, tt_k, tt_v, *, slot_idx, layer_idx, kv_actual, sp_axis):
    """Write this chunk's post-RoPE K and raw V at cumulative offset ``kv_actual`` (0 = first chunk).

    ``tt_k`` / ``tt_v`` are ``[1, n_kv_local, s_local, head_dim]`` — already exactly the per-chip
    cache layout, so they land in place with no reshard.
    """
    assert tt_k.shape[1] == kv_cache.n_kv_local, (
        f"chunk carries {tt_k.shape[1]} kv heads but the cache holds {kv_cache.n_kv_local}; "
        f"update_padded_kv_cache requires the head dims to match"
    )
    for cache, tensor in ((kv_cache.k, tt_k), (kv_cache.v, tt_v)):
        _write_one(
            cache,
            tensor,
            slot_idx=slot_idx,
            layer_idx=layer_idx,
            num_layers=kv_cache.num_layers,
            kv_actual=kv_actual,
            sp_axis=sp_axis,
        )


def read_slot_kv(mesh_device, kv_cache: LlamaKVCache, slot: int, num_layers: int):
    """Read one user slot's K and V back to host in the raw on-device (block-cyclic) seq order.

    Returns ``[k, v]``, each ``[num_layers, num_kv_heads, seq_cache, head_dim]``.
    ``ConcatMesh2dToTensor(dims=(2, 1))`` concatenates SP on the sequence and TP on the heads, so one
    ``to_torch`` per cache replaces a per-chip gather. ``DRAM_MEMORY_CONFIG`` on the slice is
    required: the cache is ND-sharded ``ROUND_ROBIN_1D`` and slicing into another ND-shard
    miscomputes the DRAM core on host read-back.
    """
    start, end = slot * num_layers, slot * num_layers + num_layers
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)

    def _block(tensor):
        s = list(tensor.shape)
        sl = ttnn.slice(tensor, [start, 0, 0, 0], [end, s[1], s[2], s[3]], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        host = ttnn.to_torch(sl, mesh_composer=composer).float()
        ttnn.deallocate(sl)
        return host

    return [_block(kv_cache.k), _block(kv_cache.v)]
