# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 prefill attention module."""

from models.demos.mistral_3_5_d_p.tt.config import MeshConfig

from .config import AttentionConfig, ProgramConfig
from .kv_cache import MistralKVCache, allocate_kv_cache, write_kv_chunk
from .prefill import attention_forward
from .weights import AttentionWeights, load_attention_weights

__all__ = [
    "Attention",
    "AttentionConfig",
    "ProgramConfig",
    "AttentionWeights",
    "MistralKVCache",
    "allocate_kv_cache",
    "write_kv_chunk",
    "load_attention_weights",
    "attention_forward",
]


class Attention:
    """
    Mistral-Medium-3.5 prefill attention layer.

    Builds config + weights and dispatches the chunked-prefill forward. Every layer is full-causal
    (``sliding_window`` is null in the config), so — unlike the gpt-oss donor — there is no per-layer
    ``layer_types`` dispatch and no per-layer config copy: one config serves all 88 layers. No decode
    path in this prefill bring-up module.
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
        weight_dtype=None,
        tensor_cache_path=None,
    ):
        """
        Args:
            mesh_device: TTNN mesh device
            config: Attention configuration
            state_dict: State dict with ``{q,k,v,o}_proj.weight`` (q/k Meta-swizzled). Empty ->
                cache-only load.
            ccl_manager: Communication manager (unused when TP == 1 and SP == 1)
            mesh_config: Mesh parallelization config
            program_config: Model-specific program configurations
            layer_idx: Layer index (selects the KV-cache slot's layer offset)
            transformation_mats: Optional ``{"prefill": tensor}`` RoPE transformation matrices
            weight_dtype: Weight dtype; defaults to the spec's attention weight dataformat
            tensor_cache_path: Optional path for weight caching
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

        # Convenience mirrors of the config, matching the donor's surface.
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
            hidden_states: Input tensor [1, 1, batch*seq_len, hidden_size]
            rope_mats: (cos, sin) with YaRN baked in. Whole-cache block-cyclic SP cos/sin when
                ``indexed_rope`` is set (see ``tt/rope.build_indexed_rope``).
            position_idx: Position indices (unused in prefill)
            kv_cache: Optional MistralKVCache (packed K/V); may be None on the unit-test path
            user_id: User/batch index; also the cache slot index
            batch_size: number of users packed on the sequence dim
            cached_len: valid prefix already in the cache before this chunk
            indexed_rope: use the on-device indexed RoPE

        Returns:
            Attention output [1, 1, batch*seq_len, hidden_size]
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
