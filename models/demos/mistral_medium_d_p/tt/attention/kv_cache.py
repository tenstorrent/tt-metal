# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 chunked-prefill KV cache (GQA K/V, head_dim 128).

Ported from ``gpt_oss_d_p/tt/attention/kv_cache.py``; the layout is the DeepSeek chunked-KV
substrate that ``update_padded_kv_cache`` and the ring-joint SDPA both already speak:

  * two persistent device caches, per-chip shape ``[num_users*num_layers, n_kv_local, seq_local, head_dim]``
  * at the TP=4 target each chip holds **2 KV heads** (8 KV heads over the 4 TP cols)
  * the sequence is SP-sharded block-cyclic over the ``sp`` rows
  * user-major packing: ``slot = user_id * num_layers + layer_idx``

``n_kv_local = 2`` is the one thing here that no other GQA model in the repo does — minimax_m3 is
4 KV heads over TP=4 and gpt_oss is 8 over TP=8, both landing on exactly 1/chip. It is legal:
``update_padded_kv_cache`` only requires ``cache_shape[1] == input_shape[1]`` and block-cyclic-shards
the SEQUENCE, so heads are orthogonal to it. Unexercised, though — hence
``tests/unit/test_kv_cache_write_vs_ref.py``.

Sizing note: 88 layers x 2 (K+V) x 8 heads x 128 dim = **352 KiB/token at bf16**, 176 KiB/token at
bf8. A single 128K-token user is ~22 GiB of cache at bf8 spread over the 32 chips (~0.7 GiB/chip),
so cache capacity is not the constraint — TTFT is.
"""

from dataclasses import dataclass

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill

# Must match the DRAM NdShard below and the address-table bank walk.
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


@dataclass
class MistralKVCache(KvCaches):
    """Externally-owned, user-major packed prefill KV caches for the SP chunked-KV path."""

    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int
    n_kv_local: int


def allocate_kv_cache(
    mesh_device,
    *,
    num_layers,
    max_seq_len,
    sp_axis=0,
    num_users=1,
    head_dim=128,
    n_kv_local=2,
    cache_dtype=ttnn.bfloat8_b,
) -> MistralKVCache:
    """Allocate the two external prefill KV caches (K, V). See :class:`MistralKVCache`.

    Args:
        num_layers: layers per user (full model = 88).
        max_seq_len: per-user cache capacity in tokens; must be a multiple of ``TILE_SIZE * sp`` so
            ``seq_local`` is tile-aligned (matches build_indexed_rope and the 32-token DRAM shard).
        sp_axis: mesh axis the sequence is sharded over (rows).
        num_users: independent user slots sharing the cache (1 for bring-up).
        head_dim: per-head width (128 for Mistral-Medium-3.5).
        n_kv_local: KV heads per chip = ``num_key_value_heads // tp``. **2** at the TP=4 target;
            every other GQA model in the repo uses 1. See the module docstring.
        cache_dtype: on-device cache dtype (bf8 is required by the chunked ring SDPA path).
    """
    sp = mesh_device.shape[sp_axis]
    assert max_seq_len % (ttnn.TILE_SIZE * sp) == 0, (
        f"max_seq_len ({max_seq_len}) must be a multiple of TILE_SIZE*sp ({ttnn.TILE_SIZE * sp}); "
        "seq_local must be tile-aligned"
    )
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

    def _alloc(dtype=cache_dtype):
        """Allocate + zero ON DEVICE. Deliberately not ``from_torch(torch.zeros(...))``: at 88 layers
        x 16384 local positions x 128 the host would pack ~185M bfp8 elements per cache and copy them
        over PCIe for a buffer whose entire content is zero. ``allocate_tensor_on_device`` +
        ``DRAMZeroFill`` does it with a device kernel and no host transfer (copied from
        ``deepseek_v3_d_p/utils/kv_cache_utils.py::init_kvpe_cache``).

        WHICH KV heads a chip holds is decided at write time by how the input chunk is mesh-mapped,
        not here; every chip is allocated the same empty buffer and the contents diverge on the
        first ``update_padded_kv_cache``.
        """
        cache = ttnn.allocate_tensor_on_device(
            ttnn.Shape([num_users * num_layers, n_kv_local, seq_local, head_dim]),
            dtype,
            ttnn.TILE_LAYOUT,
            mesh_device,
            mem_config,
        )
        DRAMZeroFill.op(cache)
        # allocate_tensor_on_device assigns a default 2D fully-replicated topology; the rest of the
        # model produces replicated tensors via ReplicateTensorToMesh, which is a 1D
        # MeshShape(num_devices) with a single Replicate placement. Reproduce that exactly.
        num_devices = mesh_device.shape[0] * mesh_device.shape[1]
        cache.update_tensor_topology(
            ttnn.TensorTopology(
                ttnn.MeshShape([num_devices]),
                [ttnn.PlacementReplicate()],
                list(ttnn.MeshCoordinateRange(ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1]))),
            )
        )
        return cache

    return MistralKVCache(
        k=_alloc(),
        v=_alloc(),
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp=sp,
        n_kv_local=n_kv_local,
    )


def _write_one(cache, tensor, *, slot_idx, layer_idx, num_layers, kv_actual, sp_axis):
    """Write one SP-sharded chunk into a packed cache. The op requires TILE layout and
    ``input.dtype == cache.dtype``, so cast a copy when needed (the original stays live for the SDPA
    that follows)."""
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


def write_kv_chunk(kv_cache: MistralKVCache, tt_k, tt_v, *, slot_idx, layer_idx, kv_actual, sp_axis):
    """Write this chunk's post-RoPE K and raw V into the packed cache (every layer).

    ``tt_k`` / ``tt_v`` are the per-device shards ``[1, n_kv_local, s_local, head_dim]`` — exactly the
    per-chip cache layout (n_kv_local = 2 at TP=4), so they write in place. ``kv_actual`` is the cumulative valid prefix
    before this chunk (0 for the first chunk).
    """
    assert tt_k.shape[0] == 1 and tt_v.shape[0] == 1, (
        f"write_kv_chunk writes one user per call, got leading dim k={tt_k.shape[0]}, v={tt_v.shape[0]}; "
        "loop over users (slot_idx + b) at the call site"
    )
    assert tt_k.shape[1] == kv_cache.n_kv_local and tt_v.shape[1] == kv_cache.n_kv_local, (
        f"chunk has {tt_k.shape[1]}/{tt_v.shape[1]} KV heads but the cache was allocated for "
        f"{kv_cache.n_kv_local}; update_padded_kv_cache requires cache_shape[1] == input_shape[1]"
    )
    assert 0 <= slot_idx < kv_cache.num_users, f"slot_idx {slot_idx} out of range [0, {kv_cache.num_users})"
    assert 0 <= layer_idx < kv_cache.num_layers, f"layer_idx {layer_idx} out of range [0, {kv_cache.num_layers})"
    assert kv_actual % ttnn.TILE_SIZE == 0, f"kv_actual ({kv_actual}) must be a multiple of {ttnn.TILE_SIZE}"
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
