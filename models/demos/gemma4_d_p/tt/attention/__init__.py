# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 context-parallel prefill attention with local and packed global ring caches."""

import ttnn

from .weights import load_attention_weights
from .prefill import prefill_forward
from .ring_prefill import init_packed_ring_kv_cache, init_ring_kv_cache


class Gemma4AttentionConfig:
    """Configuration for a single attention layer, derived from HF config + layer type."""

    def __init__(self, hf_config, layer_idx):
        self.layer_type = hf_config.layer_types[layer_idx]
        self.hidden_size = hf_config.hidden_size
        self.num_attention_heads = hf_config.num_attention_heads
        self.rms_norm_eps = hf_config.rms_norm_eps

        self.is_sliding = self.layer_type == "sliding_attention"
        self.use_kv_tying = getattr(hf_config, "attention_k_eq_v", False) and not self.is_sliding

        if self.is_sliding:
            self.num_key_value_heads = hf_config.num_key_value_heads
            self.head_dim = hf_config.head_dim
            self.sliding_window = hf_config.sliding_window
            self.rope_theta = hf_config.rope_theta
            self.partial_rotary_factor = 1.0
        else:
            # Global KV heads: use num_global_key_value_heads if set, else fall back to sliding
            global_kv = getattr(hf_config, "num_global_key_value_heads", None)
            self.num_key_value_heads = global_kv if global_kv else hf_config.num_key_value_heads
            self.head_dim = getattr(hf_config, "global_head_dim", hf_config.head_dim)
            self.sliding_window = None
            self.rope_theta = hf_config.global_rope_theta
            self.partial_rotary_factor = hf_config.partial_rotary_factor

        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads


class Gemma4Attention:
    def __init__(
        self,
        mesh_device,
        config,
        state_dict,
        ccl_manager,
        mesh_config,
        layer_idx,
        tensor_cache_path=None,
        max_batch_size=1,
        max_seq_len=131072,
        weight_dtype=ttnn.bfloat16,
        ring_kv_cache=None,
        ring_layer_idx=0,
        ring_num_layers=1,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config
        self.layer_idx = layer_idx

        if ring_kv_cache is not None:
            cache = ring_kv_cache.kv if hasattr(ring_kv_cache, "kv") else ring_kv_cache[0]
            if cache.shape[-2] * mesh_config.prefill.sp < max_seq_len:
                raise ValueError("External ring cache is too small for the configured prefill capacity")

        self.weights = load_attention_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            mesh_config=mesh_config,
            tensor_cache_path=tensor_cache_path,
            weight_dtype=weight_dtype,
        )

        self.ring_kv_cache = ring_kv_cache
        self.ring_layer_idx = ring_layer_idx
        self.ring_num_layers = ring_num_layers
        self.ring_max_seq_len = cache.shape[-2] * mesh_config.prefill.sp if ring_kv_cache is not None else None
        if self.ring_kv_cache is None:
            num_local_kv_heads = 1 if self.weights.kv_replicated else config.num_key_value_heads // mesh_config.tp
            if self.weights.is_global:
                self.ring_kv_cache = init_packed_ring_kv_cache(
                    mesh_device=mesh_device,
                    mesh_config=mesh_config,
                    num_local_kv_heads=num_local_kv_heads,
                    max_seq_len=max_seq_len,
                    num_layers=1,
                    num_users=max_batch_size,
                )
            else:
                self.ring_kv_cache = init_ring_kv_cache(
                    mesh_device=mesh_device,
                    mesh_config=mesh_config,
                    num_local_kv_heads=num_local_kv_heads,
                    head_dim=config.head_dim,
                    max_seq_len=max_seq_len,
                    num_layers=1,
                    num_users=max_batch_size,
                )
            self.ring_max_seq_len = max_seq_len

    def __call__(
        self,
        hidden_states,
        rope_mats,
        chunk_start_idx=0,
        packed_global_rope=None,
        packed_sliding_rope=None,
    ):
        """Run one chunk of ring attention."""
        tt_out = prefill_forward(
            hidden_states=hidden_states,
            cos_cache=rope_mats[0],
            sin_cache=rope_mats[1],
            weights=self.weights,
            config=self.config,
            mesh_config=self.mesh_config,
            ccl_manager=self.ccl_manager,
            chunk_start_idx=chunk_start_idx,
            ring_kv_cache=self.ring_kv_cache,
            ring_max_seq_len=self.ring_max_seq_len,
            ring_layer_idx=self.ring_layer_idx,
            ring_num_layers=self.ring_num_layers,
            packed_global_rope=packed_global_rope,
            packed_sliding_rope=packed_sliding_rope,
        )
        return tt_out
