# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B prefill attention module."""

import dataclasses

import ttnn

from models.demos.llama_3_1_8b.tt.config import MeshConfig

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
]


class Attention:
    """
    Llama 3.1 8B prefill attention layer.

    Builds config + weights and dispatches the chunked-prefill forward. All layers are full-causal
    (no sliding window). No decode path in this bring-up module.
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
        Initialize attention layer.

        Args:
            mesh_device: TTNN mesh device
            config: Attention configuration
            state_dict: State dictionary containing weights
            ccl_manager: Communication manager
            mesh_config: Mesh parallelization config
            program_config: Model-specific program configurations
            layer_idx: Layer index
            transformation_mats: Optional {"prefill": tensor} RoPE transformation matrices
            weight_dtype: Data type for weights (default: bfloat8_b)
            tensor_cache_path: Optional path for weight caching
        """
        self.mesh_config = mesh_config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.program_config = program_config
        self.layer_idx = layer_idx
        self.transformation_mats = transformation_mats

        # All Llama 3.1 layers are full-causal — no sliding window, no per-layer type switching.
        self.config = config

        # Load weights (no bias, no sinks)
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
        position_idx=None,
        kv_cache=None,
        user_id=0,
        batch_size=1,
        cached_len=0,
        indexed_rope=False,
    ):
        """
        Prefill attention forward.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            rope_mats: Tuple/list of (cos, sin) matrices for RoPE
            position_idx: Position indices (unused in prefill)
            kv_cache: Optional LlamaKVCache
            user_id: User/batch index; also the cache slot index
            batch_size: number of users packed on the sequence dim
            cached_len: valid prefix already in the cache before this chunk
            indexed_rope: use the on-device indexed RoPE

        Returns:
            Attention output [batch, seq_len, hidden_size]
        """
        transformation_mat = self.transformation_mats["prefill"] if self.transformation_mats else None

        return attention_forward(
            hidden_states=hidden_states,
            rope_mats=rope_mats,
            user_id=user_id,
            weights=self.weights,
            kv_cache=kv_cache,
            config=self.config,
            mesh_config=self.mesh_config,
            mesh_device=self.mesh_device,
            program_config=self.program_config,
            transformation_mat=transformation_mat,
            position_idx=position_idx,
            ccl_manager=self.ccl_manager,
            batch_size=batch_size,
            layer_idx=self.layer_idx,
            cached_len=cached_len,
            indexed_rope=indexed_rope,
        )
