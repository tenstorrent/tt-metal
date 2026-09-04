# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The packed, block-cyclic, SP-sharded prefill KV cache — **the output of prefill**.

**HF anchor:** none — this is a serving structure, not model math. It holds
`transformers.models.llama.modeling_llama.LlamaAttention`'s **post-RoPE K** and **raw V**, which is
what every template in this tree stores (`models/demos/gpt_oss_d_p/tt/attention/prefill.py:162-165`,
`models/demos/minimax_m3/tests/unit/test_kv_cache_write_vs_ref.py:11-13`), and it is stated here
because a cache holding pre-RoPE K would read back plausibly and attend wrongly.

**Template:** `models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:48` / `:117` / `:138`, changed in
exactly two values — `head_dim` 64 -> **128** and `num_layers` 36 -> **32**
(`bringup_log/03_OUTLINE.md` §2.7). Everything structural is kept:

* **`NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32`** and the `[1, 1, 32, head_dim]` DRAM `NdShardSpec`
  (`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:27`, used at `:87`). Matching the producer's
  block geometry is what lets P10 reuse its existing packed-GQA read-back instead of writing a
  fourth reader; diverging is a `DEC` whose blast radius includes `G-MOCK-MIG`
  (`BRINGUP_RECIPE.md:1456-1458`). `head_dim = 128` is parameterised by that shard spec and is
  tile-aligned.
* **Per-chip cache is exactly ONE KV head** — the hard-coded `1` at
  `models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:95-99`. This is the equality that forces
  **TP == num_key_value_heads == 8** (`bringup_log/00_MODEL_CARD.md` §4.1): at any smaller TP the
  model emits `8/TP > 1` local KV heads and the write op aborts with
  `TT_FATAL: cache and input num-heads dim must match`
  (`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp:230`).
* **User-major batch packing**, `slot = user_id * num_layers + layer_idx`, so a user's layers stay
  contiguous and match `update_padded_kv_cache`'s own `slot_idx` / `layer_idx` indexing.

**Dtype is `bfloat8_b`** (`DEC-021`); the bf16 number is a measurement mode, and `G-KV` records the
delta rather than assuming it.

**What this file cannot be tested for on one card.** At `(1,1)`, `sp = 1` so the block-cyclic
layout degenerates to the identity, and `nkv = tp = 1` is a head count the deployment mesh's model
never emits as a whole tensor. `G-KV` therefore drives the write op **one synthetic head at a time**
and proves the cache *primitive*; the model -> cache path is `G-KV-TP8`'s (P8), and `07_RISKS.md`
R-001 carries the gap.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.common.prefill.runners.migration import get_num_dram_banks

# Must match the DRAM NdShard in `allocate_kv_cache` and P10's address-table bank walk. Kept at the
# producer's value on purpose — see the module docstring.
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


@dataclass
class LlamaKVCache(KvCaches):
    """Externally-owned packed prefill K/V caches, one KV head per chip.

    Per-chip shape `[num_users * num_layers, 1, max_seq_len // sp, head_dim]`, `bfloat8_b`, TILE,
    DRAM `NdShardSpec` round-robin over the DRAM banks. Heads are TP-sharded on the columns **at
    write time** (by how the input chunk is mesh-mapped, not by the allocation); the sequence is
    SP-sharded block-cyclic on the rows.
    """

    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int


def allocate_kv_cache(
    mesh_device,
    *,
    num_layers,
    max_seq_len,
    sp_axis=0,
    num_users=1,
    head_dim=128,
    cache_dtype=ttnn.bfloat8_b,
) -> LlamaKVCache:
    """Allocate the two persistent prefill caches (K, V). See :class:`LlamaKVCache`.

    Args:
        mesh_device: the open mesh.
        num_layers: layers per user (32 for the full model).
        max_seq_len: per-user cache capacity in tokens; a multiple of `TILE_SIZE * sp` so
            `seq_local` is tile-aligned, which the TILE layout, the 32-token DRAM shard and
            `tt/rope.py::build_indexed_rope`'s own constraint all require.
        sp_axis: mesh axis the sequence is sharded over (rows).
        num_users: independent user slots sharing the cache (1 for this bring-up).
        head_dim: per-head width, 128 for Llama.
        cache_dtype: on-device dtype, `bfloat8_b` (`DEC-021`).
    """
    sp = mesh_device.shape[sp_axis]
    assert max_seq_len % (ttnn.TILE_SIZE * sp) == 0, (
        f"max_seq_len ({max_seq_len}) must be a multiple of TILE_SIZE*sp ({ttnn.TILE_SIZE * sp}); "
        f"seq_local must be tile-aligned"
    )
    assert head_dim % ttnn.TILE_SIZE == 0, f"head_dim ({head_dim}) must be tile-aligned for the DRAM shard"
    seq_local = max_seq_len // sp

    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0))
        for bank_id in range(get_num_dram_banks(mesh_device))
    ]
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    def _alloc():
        # `1` in the head dim: the per-chip cache is one KV head. WHICH head a chip holds is decided
        # at write time by the input chunk's mesh mapping, not here — so the allocation is
        # replicated and identical on every chip, and the contents diverge on the first write.
        return ttnn.from_torch(
            torch.zeros(num_users * num_layers, 1, seq_local, head_dim),
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
    )


def _write_one(cache, tensor, *, slot_idx, layer_idx, num_layers, kv_actual, sp_axis):
    """Write one SP-sharded chunk into a packed cache via `update_padded_kv_cache`.

    The op requires TILE layout and `input.dtype == cache.dtype`, so a copy is cast when needed and
    the caller's tensor stays live in bf16 for the SDPA that follows
    (`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:117-136`).
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
    """Write this chunk's post-RoPE K and raw V into the packed cache. **One user per call.**

    `tt_k` / `tt_v` are the per-device shards `[1, n_kv_local, s_local, head_dim]` — heads
    TP-sharded on the columns, sequence SP-sharded on `sp_axis` — i.e. already the per-chip cache
    layout, so they write in place. `kv_actual` is the cumulative valid prefix **before** this
    chunk (0 for the first or only chunk).
    """
    # `update_padded_kv_cache` writes a single (slot_idx, layer_idx) and **ignores the leading
    # batch dim**, so a batch > 1 tensor would silently write only `slot_idx` and drop the rest.
    # Fail loud; multi-user prefill loops `slot_idx + b` at the call site
    # (`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:145-152`).
    assert tt_k.shape[0] == 1 and tt_v.shape[0] == 1, (
        f"write_kv_chunk writes one user per call, but got leading (batch) dim "
        f"k={tt_k.shape[0]}, v={tt_v.shape[0]}; loop over users (slot_idx + b) at the call site"
    )
    # A bad slot or layer would be a silent out-of-bounds write into another user's or layer's
    # slot; a misaligned offset breaks the block-cyclic per-device write.
    assert 0 <= slot_idx < kv_cache.num_users, f"slot_idx {slot_idx} out of range [0, {kv_cache.num_users})"
    assert 0 <= layer_idx < kv_cache.num_layers, f"layer_idx {layer_idx} out of range [0, {kv_cache.num_layers})"
    assert (
        kv_actual % ttnn.TILE_SIZE == 0
    ), f"kv_actual ({kv_actual}) must be tile-aligned (multiple of {ttnn.TILE_SIZE})"
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
