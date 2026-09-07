# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Externally owned, zero-copy Gemma 4 prefill KV caches."""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.attention.ring_prefill import init_packed_ring_kv_cache, init_ring_kv_cache


@dataclass
class Gemma4KvCaches(KvCaches):
    """One durable migration-ready ring cache per semantic model layer."""

    layers: list
    layer_types: tuple[str, ...]
    num_users: int
    max_seq_len: int
    sp: int
    tp: int

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, layer_idx):
        return self.layers[layer_idx]

    @property
    def global_layers(self):
        return tuple(i for i, layer_type in enumerate(self.layer_types) if layer_type == "full_attention")

    @property
    def sliding_layers(self):
        return tuple(i for i, layer_type in enumerate(self.layer_types) if layer_type == "sliding_attention")


def allocate_ring_kv_caches(
    mesh_device,
    hf_config,
    mesh_config,
    *,
    num_users: int,
    max_seq_len: int,
    num_layers: int | None = None,
    cache_dtype=ttnn.bfloat8_b,
    first_layer_idx: int = 0,
) -> Gemma4KvCaches:
    """Allocate the sole compute+migration cache family for a CP prefill model.

    A pipeline rank owns GLOBAL layers ``[first_layer_idx, first_layer_idx + num_layers)``
    and allocates one cache per layer it owns, in that order -- so ``caches[j]`` is global
    layer ``first_layer_idx + j``. ``layer_types`` must be sliced by that window, not
    truncated to a prefix, or a later rank builds sliding caches for its global layers.
    """
    num_layers = num_layers or hf_config.num_hidden_layers
    if num_users <= 0 or num_layers <= 0:
        raise ValueError(f"num_users and num_layers must be positive, got {num_users}, {num_layers}")
    if mesh_config.prefill.sp <= 1:
        raise ValueError("migration-ready Gemma 4 caches require context parallel prefill")
    if first_layer_idx + num_layers > len(hf_config.layer_types):
        raise ValueError(
            f"layer window [{first_layer_idx}, {first_layer_idx + num_layers}) exceeds the model's "
            f"{len(hf_config.layer_types)} layers"
        )
    layer_types = tuple(hf_config.layer_types[first_layer_idx : first_layer_idx + num_layers])
    caches = []
    for local_idx, layer_type in enumerate(layer_types):
        layer_idx = first_layer_idx + local_idx
        config = Gemma4AttentionConfig(hf_config, layer_idx)
        # Both branches divide by TP. A global layer has num_global_key_value_heads (4 on
        # 31B) rather than num_key_value_heads (16), which is what made a hardcoded 1
        # correct at TP=4 and wrong everywhere else -- notably at the TP=1 of a [8,1]
        # pipeline stage, where a global layer holds all 4 heads.
        local_heads = config.num_key_value_heads // mesh_config.tp
        if local_heads < 1:
            raise ValueError(
                f"layer {layer_idx} ({layer_type}) has {config.num_key_value_heads} KV heads, "
                f"which does not divide TP={mesh_config.tp}"
            )
        if layer_type == "full_attention":
            cache = init_packed_ring_kv_cache(
                mesh_device,
                mesh_config,
                local_heads,
                max_seq_len,
                num_users=num_users,
                cache_dtype=cache_dtype,
            )
        elif layer_type == "sliding_attention":
            cache = init_ring_kv_cache(
                mesh_device,
                mesh_config,
                local_heads,
                config.head_dim,
                max_seq_len,
                num_users=num_users,
                cache_dtype=cache_dtype,
            )
        else:
            raise ValueError(f"unsupported Gemma 4 layer type {layer_type!r} at layer {layer_idx}")
        caches.append(cache)
    return Gemma4KvCaches(
        layers=caches,
        layer_types=layer_types,
        num_users=num_users,
        max_seq_len=max_seq_len,
        sp=mesh_config.prefill.sp,
        tp=mesh_config.tp,
    )
