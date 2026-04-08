# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Per-layer configuration proxy for Gemma 4's hybrid attention architecture.

Gemma 4 uses two types of attention layers with different dimensions:
- Sliding attention: head_dim=256, n_kv_heads=16, with v_proj
- Full attention: head_dim=512, n_kv_heads=4, K=V sharing (no v_proj)

This proxy wraps the ModelArgs and overrides attention-related attributes
per layer, allowing the standard Attention class to work without modification.
"""

import ttnn


class Gemma4LayerConfig:
    """Proxy that delegates to ModelArgs but overrides per-layer attention attributes."""

    def __init__(self, base_args, layer_num):
        object.__setattr__(self, "_base", base_args)
        object.__setattr__(self, "_layer_num", layer_num)
        object.__setattr__(self, "_overrides", {})

        is_sliding = base_args.is_layer_sliding(layer_num)

        if is_sliding:
            hd = base_args.head_dim  # 256
            nkv = base_args.n_kv_heads  # 16
        else:
            hd = base_args.global_head_dim  # 512
            nkv = base_args.num_global_kv_heads  # 4

        overrides = object.__getattribute__(self, "_overrides")
        overrides["head_dim"] = hd
        overrides["n_kv_heads"] = nkv
        overrides["qkv_size"] = hd * (2 * nkv + base_args.n_heads)

        # Recompute derived values
        cluster_width = base_args.cluster_shape[1]
        n_local_kv = nkv // cluster_width
        overrides["min_kv_prefill_shard_seqlen"] = (ttnn.TILE_SIZE * 8 * 8) / n_local_kv if n_local_kv > 0 else 0

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        base = object.__getattribute__(self, "_base")
        return getattr(base, name)

    def __setattr__(self, name, value):
        overrides = object.__getattribute__(self, "_overrides")
        overrides[name] = value
