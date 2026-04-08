# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Attention module.

Uses HF-style ttnn.experimental.rotary_embedding — no Meta-format weight conversion,
no transformation matrices. Cos/sin caches are passed directly.

Supports two layer types:
- sliding_attention: head_dim=256, 8 KV heads, separate K/V, full RoPE, window=1024
- full_attention: head_dim=512, 2 KV heads, K=V tying, partial RoPE (0.25), full context
"""

import ttnn
from models.demos.gemma4.config import MeshConfig, Mode

from .weights import AttentionWeights, load_attention_weights
from .kv_cache import init_kv_cache
from .decode import decode_forward
from .prefill import prefill_forward


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
        program_config,
        layer_idx,
        tensor_cache_path=None,
        create_kv_cache=False,
        max_batch_size=1,
        max_seq_len=131072,
        # Legacy parameter — ignored (no longer needed with HF-style RoPE)
        transformation_mats=None,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config
        self.layer_idx = layer_idx

        self.weights = load_attention_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            mesh_config=mesh_config,
            tensor_cache_path=tensor_cache_path,
        )

        if create_kv_cache:
            self.kv_cache = init_kv_cache(
                mesh_device=mesh_device,
                config=config,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                tensor_cache_path=tensor_cache_path,
            )
        else:
            self.kv_cache = None

    def __call__(
        self,
        hidden_states,
        rope_mats=None,
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=True,
        token_index=None,
        shared_kv=None,
        keep_kv=False,
        is_kv_shared=False,
    ):
        """
        Attention forward pass — dispatches to on-device decode or prefill.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device
            rope_mats: (cos_cache, sin_cache) TT tensors, shape [1, 1, max_seq_len, head_dim]
            position_idx: position tensor for KV cache update (decode only)
            page_table: paged attention page table
            kv_cache: [k_cache, v_cache] or None
            is_decode: True for decode mode
            token_index: int position for decode RoPE slicing (decode only)
            shared_kv: optional (tt_k, tt_v) from source layer for KV sharing (prefill only)
            keep_kv: if True, keep K/V alive for sharing with later layers (prefill only)
            is_kv_shared: if True, this layer shares KV from source (skip K/V proj + cache update)
        """
        cache = kv_cache or self.kv_cache
        cos_cache, sin_cache = rope_mats

        if is_decode:
            return decode_forward(
                hidden_states=hidden_states,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                weights=self.weights,
                kv_cache=cache,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                position_idx=position_idx,
                token_index=token_index,
                page_table=page_table,
                ccl_manager=self.ccl_manager,
                is_kv_shared=is_kv_shared,
            )
        else:
            tt_out, kept_kv = prefill_forward(
                hidden_states=hidden_states,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                weights=self.weights,
                kv_cache=cache,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                page_table=page_table,
                ccl_manager=self.ccl_manager,
                shared_kv=shared_kv,
                keep_kv=keep_kv,
            )
            self._last_kv = kept_kv
            return tt_out
