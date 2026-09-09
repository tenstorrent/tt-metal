# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B on-device prefill KV cache: layout, allocation, and the chunk write.

The layout here is **not invented**. It is canonical by convention across every package on the
`common/prefill` engine, because the chunked ring SDPA reads it —
`minimax_m3/tt/attention/kv_cache.py` and `gpt_oss_d_p/tt/attention/kv_cache.py` are near-identical
for exactly that reason, and this file is the same layout again. Borrowed from `minimax_m3` (all
five coupled roles from that one package: the 32-token bank walk, slot packing, block-cyclic SP
sharding, the write op, and the bank count), measured at hidden 6144 / head_dim 128 / chunk 5120 /
sp8×tp4 on this same Blackhole Galaxy.

## The four per-model decisions (recipe §5.2)

1. **Number of cache tensors: 2** — `k` and `v`. Falls straight out of the attention family: Llama
   is GQA, so there is a real K and a real V (MLA would be 1 latent `kvpe`; MiniMax-M3's sparse
   attention needs a 3rd `index_k` for its indexer). The donor's `index_k` is dropped, and with it
   the whole `M3_INDEX_CACHE_BF16` question — there is no hard top-k block selection here to
   destabilise.
2. **`head_dim`: 128** — GQA, so the model's head dim as-is (`hidden_size / num_attention_heads`).
   No MLA-style `kv_lora_rank + qk_rope_head_dim` composition.
3. **`cache_dtype`: `bfloat8_b`** — from the spec's `dataformats.kv_cache.default`.
4. **Auxiliary caches: none.**

## The one thing no donor covers: TWO KV heads per chip

Every prefill package on this engine holds exactly **one** KV head per chip — minimax_m3 has 4 KV
heads at TP=4, gpt_oss_d_p has 8 at TP=8 — and their allocation code hardcodes a literal `1` in the
head dim while their *write* docstrings describe a parametrized `n_kv_local`. Llama-3.1-8B has 8 KV
heads at the spec's TP=4, so **`n_kv_local = 2`**, and the per-chip cache is
`[num_users*num_layers, 2, seq_local, 128]`.

Nothing raises if this is got wrong. It silently changes the per-chip cache row width, the ring
gather buffer, every host-side read-back and the migration bank walk — the ND-shard order is
`(batch, head, seq_block)`, so a walk must step the local-head dim INSIDE each (slot, layer).

`update_padded_kv_cache` is expected to take it: its only single-head constraint,
`input.padded_shape()[1] == 1`, fires solely when `tp_axis` is set (the TP-deduped cache, where each
chip stores 1/tp of the sequence for a replicated head). K/V here pass `cluster_axis=sp_axis` and no
`tp_axis`, so `tp_factor == 1` and the head dim is unconstrained. "Expected" is not "verified", so
`tests/unit/test_kv_cache_two_heads_per_chip.py` pins it at the boundary rather than assuming it.

## Capacity is rounded UP to a whole number of chunks

`max_seq_len` need not be a whole number of chunks, and the spec's is not: 131072 / 5120 = 25.6. The
cache (and the indexed rope tables) are block-cyclic with period `chunk_size`, so the ALLOCATION is
rounded up to the next whole chunk — 26 × 5120 = 133120 tokens. Allocating exactly 131072 leaves the
final partial chunk's block-cyclic addresses pointing past the end of the buffer.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.common.prefill.runners.migration import get_num_dram_banks

NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32
"""Sequence tokens per DRAM bank block. Fixed by convention — the address-table builder and the
ring cache-read op both assume exactly this, so it is not a tunable."""


def cache_capacity(max_seq_len: int, chunk_size: int) -> int:
    """Cache capacity in tokens: `max_seq_len` rounded UP to a whole number of chunks.

    See the module docstring — the block-cyclic period is `chunk_size`, so a capacity that is not a
    multiple of it lets the last chunk address past the buffer. 131072 @ chunk 5120 -> 133120.
    """
    return -(-max_seq_len // chunk_size) * chunk_size


@dataclass
class LlamaKVCache(KvCaches):
    """Two persistent packed device caches, each per-chip
    `[num_users*num_layers, n_kv_local, seq_local, head_dim]` on the DRAM ND-shard substrate,
    written in place by `ttnn.experimental.deepseek_prefill.update_padded_kv_cache`.

    * `k` — post-RoPE keys. `v` — raw values. Heads shard on the TP cols (**2 per chip** at TP=4);
      the sequence is SP-sharded block-cyclic on the `sp` rows.

    Batch dim is user-major (`slot = user_id * num_layers + layer_idx`) so each user's layers stay
    contiguous, matching `update_padded_kv_cache`'s indexing. The adapter allocates this once and
    the engine holds it as an opaque handle.
    """

    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    capacity: int
    n_kv_local: int
    head_dim: int
    sp: int


def allocate_kv_caches(
    mesh_device,
    *,
    num_layers,
    max_seq_len,
    chunk_size,
    sp_axis=0,
    tp_axis=1,
    num_users=1,
    num_kv_heads=8,
    head_dim=128,
    cache_dtype=ttnn.bfloat8_b,
) -> LlamaKVCache:
    """Allocate the K and V prefill caches. See :class:`LlamaKVCache`.

    Deliberately NOT `init_kvpe_cache` — that is MLA-specific and allocates a single latent cache.
    This owns the GQA pair and the user-major packing, while reusing the identical DRAM NdShard spec
    (same bank grid, same 32-token contiguous shard) so `update_padded_kv_cache` writes into these
    tensors unchanged.

    Args:
        num_layers: layers per user (full model = 32).
        max_seq_len: per-user capacity in tokens, before the round-up to a whole chunk.
        chunk_size: the block-cyclic addressing period. Capacity is rounded up to a multiple of it.
        num_kv_heads: GLOBAL KV head count (8). The per-chip count is this / tp — **2 at TP=4**.
        head_dim: per-head width (128).
        cache_dtype: on-device cache dtype (`bfloat8_b`, from the spec).
    """
    rows_cols = tuple(mesh_device.shape)
    sp = rows_cols[sp_axis]
    tp = rows_cols[tp_axis]

    capacity = cache_capacity(max_seq_len, chunk_size)
    tile = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    assert capacity % (tile * sp) == 0, (
        f"cache capacity ({capacity}) must be a multiple of {tile}*sp ({tile * sp}): the cache is "
        "block-cyclic on the SP axis, and a misaligned capacity corrupts addresses silently"
    )
    assert num_kv_heads % tp == 0, (
        f"num_kv_heads ({num_kv_heads}) must be divisible by tp ({tp}): a KV head cannot straddle chips"
    )
    n_kv_local = num_kv_heads // tp  # 2 at TP=4 — see the module docstring
    seq_local = capacity // sp

    num_banks = get_num_dram_banks(mesh_device)
    core_ranges = [ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(num_banks)]
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, tile, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    def _alloc():
        # Allocated zeroed and REPLICATED: every chip gets the same empty buffer, and WHICH KV heads
        # a chip ends up holding is decided at write time by how the input chunk is mesh-mapped, not
        # here. Content diverges on the first update_padded_kv_cache write.
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
        capacity=capacity,
        n_kv_local=n_kv_local,
        head_dim=head_dim,
        sp=sp,
    )


def _write_one(cache, tensor, *, slot_idx, layer_idx, num_layers, kv_actual, sp_axis):
    """Write one SP-sharded chunk tensor into a packed cache via `update_padded_kv_cache`.

    The op requires TILE layout and `input.dtype == cache.dtype`, so cast a copy to the cache's
    dtype when needed — the original stays live for the attention op that follows. At
    `kv_actual % 32 == 0` chunk boundaries the per-device write offset is contiguous (block-cyclic
    degenerates to a reshape).
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
    """Write this chunk's post-RoPE K and raw V into the packed cache.

    `tt_k` / `tt_v` are the per-device SP shards `[1, n_kv_local, s_local, head_dim]` — heads
    TP-sharded on the cols (**2 per chip**), sequence SP-sharded on the `sp_axis` rows — which is
    exactly the per-chip cache layout, so they write in place with no reshaping.

    `kv_actual` is the cumulative valid prefix BEFORE this chunk (0 for a one-shot run).
    """
    assert tt_k.shape[1] == tt_v.shape[1] == kv_cache.n_kv_local, (
        f"chunk carries {tt_k.shape[1]} local KV heads but the cache holds {kv_cache.n_kv_local}; "
        "a mismatch here writes to the wrong rows without raising"
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
