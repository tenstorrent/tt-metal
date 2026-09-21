# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Everything a chunk carries forward: the GQA KV cache and the Gated DeltaNet recurrent state.

A hybrid model has **two** kinds of carried state, and chunked prefill is only correct if both are
carried. They are allocated together here so a layer cannot be wired to one and forget the other.

**KV cache (16 full-attention layers).** Canonical layout, copied verbatim from the convention the
chunked ring SDPA reads (recipe section 5.1) — deviating here does not raise, it corrupts:

| element | value |
|---|---|
| per-chip shape | ``[num_users * num_kv_layers, 1, seq_local, head_dim]`` |
| slot packing | ``slot = user_id * num_kv_layers + kv_slot`` (user-major, layers contiguous) |
| DRAM memory config | ``NdShardSpec``, shard ``[1, 1, 32, head_dim]``, ``ROUND_ROBIN_1D`` |
| contiguous tokens per bank | 32 |
| sequence sharding | SP block-cyclic; ``seq_local = max_seq_len // sp`` |
| alignment | ``max_seq_len % (32 * sp) == 0`` |
| allocation | zeroed, ``ReplicateTensorToMesh`` (content diverges on first write) |
| write | ``ttnn.experimental.deepseek_prefill.update_padded_kv_cache`` |

Two of the four per-model decisions differ from the donor: **two** cache tensors (plain GQA — no
MLA latent, no sparse ``index_k``) and ``head_dim`` **256**. The third, ``cache_dtype``, comes from
the spec. The fourth, auxiliary caches, is where this model is unusual: there are none, but there
*is* a second state family.

**The layer index is not the cache slot.** Only 16 of the 64 layers own K/V, so the cache packs 16
slots per user rather than 64 — a quarter of the DRAM, and the reason ``Qwen35TextConfig.kv_slot``
exists rather than passing ``layer_idx`` through.

**GDN state (48 linear-attention layers).** A 4-tap causal conv needs the previous chunk's last 3
pre-conv tokens, and the delta-rule scan needs its ``[k_dim, v_dim]`` matrix state per head. Both
are replicated across the SP rows (every row runs the same full-chunk scan — see
``gdn/prefill.py``) and sharded across TP by value head.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

import ttnn
from models.demos.common.prefill.runners.migration import get_num_dram_banks

from ..config import MeshConfig
from ..reference.config import Qwen35TextConfig

NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


@dataclass
class KVCache:
    """The two persistent, user-major packed device caches for the full-attention layers."""

    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_kv_layers: int
    max_seq_len: int
    sp: int
    head_dim: int

    def slot(self, user_id: int, kv_slot: int) -> int:
        return user_id * self.num_kv_layers + kv_slot


@dataclass
class GdnState:
    """One Gated DeltaNet layer's carried state.

    ``conv_state``  ``[1, kernel-1, conv_dim/tp]`` bf16 ROW_MAJOR — the last pre-conv tokens, in
                    the layout ``qkv_causal_conv1d_silu`` takes as its ``history`` argument.
    ``recurrent``   ``[1, num_v_heads/tp, head_k_dim, head_v_dim]`` fp32 — the delta-rule matrix
                    state, in the layout ``chunk_gated_delta_rule`` takes as ``initial_state``.

    Mutable on purpose: a chunk replaces both in place on the object the layer holds, so the
    runtime threads state through chunks without rebuilding the model.
    """

    conv_state: ttnn.Tensor
    recurrent: ttnn.Tensor
    seeded: bool = False  # False until the first chunk writes real values


@dataclass
class PrefillCaches:
    """Both state families, allocated together and passed to every layer as one handle."""

    kv: KVCache
    gdn: dict[int, GdnState] = field(default_factory=dict)


def _nd_shard_memory_config(mesh_device, head_dim: int) -> ttnn.MemoryConfig:
    banks = get_num_dram_banks(mesh_device)
    core_ranges = [ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)]
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)


def allocate_kv_cache(
    mesh_device,
    cfg: Qwen35TextConfig,
    *,
    mesh_config: MeshConfig,
    max_seq_len: int,
    num_users: int = 1,
    num_kv_layers: int | None = None,
    cache_dtype=ttnn.bfloat8_b,
) -> KVCache:
    """Allocate the K and V caches for the full-attention layers.

    Deliberately NOT ``init_kvpe_cache``: that is MLA-specific and allocates a single latent cache.
    This owns the GQA pair and the user-major packing, but reuses the same DRAM NdShard spec so
    ``update_padded_kv_cache`` writes into these tensors unchanged.
    """
    sp = mesh_config.sp
    num_kv_layers = num_kv_layers if num_kv_layers is not None else len(cfg.full_attention_layers)
    align = ttnn.TILE_SIZE * sp
    assert max_seq_len % align == 0, (
        f"max_seq_len {max_seq_len} must be a multiple of TILE_SIZE*sp ({align}); the block-cyclic "
        f"addressing period is silent about a misaligned value and corrupts addresses rather than failing"
    )
    seq_local = max_seq_len // sp
    mem_config = _nd_shard_memory_config(mesh_device, cfg.head_dim)

    def _alloc() -> ttnn.Tensor:
        # Which head a chip holds is decided at WRITE time by the mesh mapping, not here: every
        # chip is allocated the same zeroed buffer and the content diverges on the first write.
        return ttnn.from_torch(
            torch.zeros(num_users * num_kv_layers, 1, seq_local, cfg.head_dim),
            dtype=cache_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=mesh_config.replicate(mesh_device),
        )

    return KVCache(
        k=_alloc(),
        v=_alloc(),
        num_users=num_users,
        num_kv_layers=num_kv_layers,
        max_seq_len=max_seq_len,
        sp=sp,
        head_dim=cfg.head_dim,
    )


def write_kv_chunk(
    kv_cache: KVCache,
    tt_k: ttnn.Tensor,
    tt_v: ttnn.Tensor,
    *,
    user_id: int,
    kv_slot: int,
    cached_len: int,
    sp_axis: int,
) -> None:
    """Write this chunk's post-RoPE K and raw V into the packed cache at the chunk's offset.

    ``tt_k`` / ``tt_v`` are the per-device shards ``[1, n_kv_local, s_local, head_dim]`` — heads on
    the TP cols, sequence on the SP rows — which is exactly the per-chip cache layout, so they land
    in place. ``cached_len`` is the cumulative valid prefix BEFORE this chunk (0 when one-shot).
    """
    for cache, tensor in ((kv_cache.k, tt_k), (kv_cache.v, tt_v)):
        src = tensor if tensor.dtype == cache.dtype else ttnn.typecast(tensor, cache.dtype)
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            src,
            slot_idx=user_id,
            layer_idx=kv_slot,
            num_layers=kv_cache.num_kv_layers,
            kv_actual_global=cached_len,
            cluster_axis=sp_axis,
        )
        if src is not tensor:
            src.deallocate(True)


def allocate_gdn_state(
    mesh_device,
    cfg: Qwen35TextConfig,
    *,
    mesh_config: MeshConfig,
) -> GdnState:
    """Zeroed conv + recurrent state for one Gated DeltaNet layer.

    Zero is the correct *initial* value for both: an all-zero conv history is exactly the left
    zero-padding a one-shot forward's ``padding=kernel-1`` produces, and an all-zero recurrent
    state is upstream's ``initial_state=None``. So chunk 0 needs no special case.
    """
    conv_state = ttnn.from_torch(
        torch.zeros(1, cfg.linear_conv_kernel_dim - 1, cfg.gdn_conv_dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.shard_mapper(mesh_device, tensor_dim=-1),
    )
    hv_dims: list[int | None] = [None, None]
    hv_dims[mesh_config.tp_axis] = 1  # value heads across the TP cols
    recurrent = ttnn.from_torch(
        torch.zeros(
            1, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim, dtype=torch.float32
        ),
        dtype=ttnn.float32,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=tuple(hv_dims)),
    )
    return GdnState(conv_state=conv_state, recurrent=recurrent)


def allocate_prefill_caches(
    mesh_device,
    cfg: Qwen35TextConfig,
    *,
    mesh_config: MeshConfig,
    max_seq_len: int,
    num_users: int = 1,
    layer_indices: list[int] | None = None,
    cache_dtype=ttnn.bfloat8_b,
) -> PrefillCaches:
    """Both state families for the layers this instance owns.

    ``layer_indices`` are GLOBAL layer indices; the KV cache is sized to how many of them are
    full-attention layers, and one :class:`GdnState` is allocated per linear-attention layer.
    """
    layers = list(layer_indices if layer_indices is not None else range(cfg.num_hidden_layers))
    kv_layers = [i for i in layers if cfg.is_full_attention(i)]
    gdn_layers = [i for i in layers if not cfg.is_full_attention(i)]
    return PrefillCaches(
        kv=allocate_kv_cache(
            mesh_device,
            cfg,
            mesh_config=mesh_config,
            max_seq_len=max_seq_len,
            num_users=num_users,
            num_kv_layers=max(1, len(kv_layers)),
            cache_dtype=cache_dtype,
        ),
        gdn={i: allocate_gdn_state(mesh_device, cfg, mesh_config=mesh_config) for i in gdn_layers},
    )
