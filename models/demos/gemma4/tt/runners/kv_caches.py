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
) -> Gemma4KvCaches:
    """Allocate the sole compute+migration cache family for a CP prefill model."""
    num_layers = num_layers or hf_config.num_hidden_layers
    if num_users <= 0 or num_layers <= 0:
        raise ValueError(f"num_users and num_layers must be positive, got {num_users}, {num_layers}")
    if mesh_config.prefill.sp <= 1:
        raise ValueError("migration-ready Gemma 4 caches require context parallel prefill")
    layer_types = tuple(hf_config.layer_types[:num_layers])
    caches = []
    for layer_idx, layer_type in enumerate(layer_types):
        config = Gemma4AttentionConfig(hf_config, layer_idx)
        local_heads = 1 if layer_type == "full_attention" else config.num_key_value_heads // mesh_config.tp
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
