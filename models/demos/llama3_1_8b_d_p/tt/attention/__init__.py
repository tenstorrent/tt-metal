# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B prefill attention module."""

import ttnn

from models.demos.llama3_1_8b_d_p.tt.config import MeshConfig

from .config import AttentionConfig, ProgramConfig
from .kv_cache import LlamaKVCache, allocate_kv_cache, write_kv_chunk
from .prefill import attention_forward
from .weights import AttentionWeights, load_attention_weights

__all__ = [
    "Attention",
    "AttentionConfig",
    "ProgramConfig",
    "AttentionWeights",
    "LlamaKVCache",
    "allocate_kv_cache",
    "write_kv_chunk",
    "load_attention_weights",
    "attention_forward",
]


class Attention:
    """Llama 3.1 8B prefill attention layer.

    Builds weights and dispatches the chunked-prefill forward. Every Llama layer is identical
    full-causal GQA, so — unlike gpt-oss — there is no per-layer type dispatch and no per-layer
    config copy: one ``AttentionConfig`` is shared by all 32 layers unmodified.
    """

    def __init__(
        self,
        mesh_device,
        config: AttentionConfig,
        state_dict,
        ccl_manager,
        mesh_config: MeshConfig,
        program_config: ProgramConfig,
        layer_idx,
        transformation_mats=None,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        """
        Args:
            mesh_device: TTNN mesh device
            config: AttentionConfig, shared across layers (never mutated here)
            state_dict: ``self_attn.*`` substate; empty dict => cache-only load
            ccl_manager: CCLManager (unused when TP == 1 and SP == 1)
            mesh_config: MeshConfig
            program_config: SDPA / compute program configs
            layer_idx: this layer's index — selects the KV cache slot row
            transformation_mats: optional ``{"prefill": tensor}`` RoPE transformation matrices
            weight_dtype: on-device weight dtype (see load_attention_weights)
            tensor_cache_path: optional weight-cache dir
        """
        self.mesh_config = mesh_config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.program_config = program_config
        self.layer_idx = layer_idx
        self.transformation_mats = transformation_mats
        self.config = config

        self.weights = load_attention_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            mesh_config=mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = config.scaling

    def __call__(
        self,
        hidden_states,
        rope_mats,
        kv_cache=None,
        user_id=0,
        batch_size=1,
        cached_len=0,
        indexed_rope=False,
    ):
        """Prefill attention forward. See ``prefill.attention_forward`` for argument semantics."""
        transformation_mat = self.transformation_mats["prefill"] if self.transformation_mats else None
        return attention_forward(
            hidden_states=hidden_states,
            rope_mats=rope_mats,
            weights=self.weights,
            kv_cache=kv_cache,
            config=self.config,
            mesh_config=self.mesh_config,
            mesh_device=self.mesh_device,
            program_config=self.program_config,
            transformation_mat=transformation_mat,
            ccl_manager=self.ccl_manager,
            user_id=user_id,
            batch_size=batch_size,
            layer_idx=self.layer_idx,
            cached_len=cached_len,
            indexed_rope=indexed_rope,
        )
