# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS prefill attention module. Mirrors ``minimax_m3/tt/attention/__init__.py``."""

import dataclasses

import ttnn

from models.demos.gpt_oss_d_p.tt.config import MeshConfig

from .config import AttentionConfig, ProgramConfig
from .prefill import attention_forward
from .weights import AttentionWeights, load_attention_weights

__all__ = ["Attention", "AttentionConfig", "ProgramConfig", "AttentionWeights"]


class Attention:
    """
    GPT-OSS prefill attention layer.

    Builds config + weights and dispatches the chunked-prefill forward. Selects sliding-window
    vs full-causal masking per layer from ``hf_config.layer_types`` (or the caller-provided
    ``layer_types``): "sliding_attention" keeps ``config.sliding_window`` (128 for gpt-oss),
    "full_attention" nulls it. No decode path in this P1 bring-up module.
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
        layer_types=None,
        transformation_mats=None,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        """
        Initialize attention layer.

        Args:
            mesh_device: TTNN mesh device
            config: Attention configuration (``config.sliding_window`` holds the sliding window
                *size*, e.g. 128; it is nulled here for full-attention layers)
            state_dict: State dictionary containing weights
            ccl_manager: Communication manager (unused when TP == 1 and SP == 1)
            mesh_config: Mesh parallelization config
            program_config: Model-specific program configurations
            layer_idx: Layer index
            layer_types: Optional per-layer type list (from hf_config.layer_types); entry
                layer_idx selects sliding vs full. Falls back to even=sliding / odd=full when None.
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

        # Sliding vs full for this layer (gpt-oss alternates per hf_config.layer_types; even=sliding).
        if layer_types is not None:
            self.is_sliding = layer_types[layer_idx] == "sliding_attention"
        else:
            self.is_sliding = (layer_idx % 2) == 0
        # Per-layer config copy so full layers get sliding_window=None — never mutate the caller's
        # config, which may be shared across layers.
        self.config = dataclasses.replace(config, sliding_window=config.sliding_window if self.is_sliding else None)

        # Load weights
        self.weights = load_attention_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            mesh_config=mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )

        # Store references for convenience
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
    ):
        """
        Prefill attention forward.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            rope_mats: Tuple/list of (cos, sin) matrices for RoPE (YaRN baked in)
            position_idx: Position indices (unused in prefill)
            kv_cache: Optional [k_cache, v_cache] pair; may be None
            user_id: User/batch index; also the cache slot index (default: 0)
            batch_size: number of users packed on the sequence dim

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
        )
