# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.demos.minimax_m3.config import MeshConfig

from .config import AttentionConfig, ProgramConfig
from .kv_cache import MiniMaxKVCache, allocate_kv_caches
from .prefill import attention_forward
from .weights import load_attention_weights

__all__ = ["Attention", "AttentionConfig", "ProgramConfig", "MiniMaxKVCache", "allocate_kv_caches"]


class Attention:
    """
    Generic Attention implementation with automatic decode/prefill dispatch.

    This class provides a clean interface for attention layers. Models provide
    their own ProgramConfig implementations to customize behavior.
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
            layer_idx: Layer index (for sliding window)
            transformation_mats: Optional transformation matrices for RoPE
            weight_dtype: Data type for weights (default: bfloat8_b)
            tensor_cache_path: Optional path for weight caching
        """
        self.config = config
        self.mesh_config = mesh_config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.program_config = program_config
        self.layer_idx = layer_idx
        self.transformation_mats = transformation_mats

        # MiniMax-M3 has no sliding-window attention (dense layers full causal; sparse layers block-sparse MSA).
        object.__setattr__(config, "sliding_window", None)

        # Load weights
        self.weights = load_attention_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            mesh_config=mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )

        # Store references for backward compatibility
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
            rope_mats: Tuple of (cos, sin) matrices for RoPE
            kv_cache: Externally-owned :class:`MiniMaxKVCache` (packed K/V/index_k). ``user_id`` is the
                cache slot, ``self.layer_idx`` the layer, ``cached_len`` the prior-prefix length.
            user_id: User/batch index; also the cache slot index (default: 0).

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
