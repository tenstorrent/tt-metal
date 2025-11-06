# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.demos.gpt_oss.config import MeshConfig

from .config import AttentionConfig, ProgramConfig
from .decode import decode_forward
from .kv_cache import get_kv_memory_config, init_kv_cache
from .prefill import prefill_forward
from .weights import load_attention_weights

__all__ = ["Attention", "AttentionConfig", "ProgramConfig"]


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
        paged_attention_config=None,
        transformation_mats=None,
        weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        create_kv_cache=True,
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
            paged_attention_config: Optional paged attention configuration
            transformation_mats: Optional transformation matrices for RoPE
            weight_dtype: Data type for weights (default: bfloat8_b)
            activation_dtype: Data type for activations (default: bfloat16)
            tensor_cache_path: Optional path for weight caching
            create_kv_cache: Whether to create KV cache (default: True)
        """
        self.config = config
        self.mesh_config = mesh_config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.program_config = program_config
        self.activation_dtype = activation_dtype
        self.layer_idx = layer_idx
        self.transformation_mats = transformation_mats
        self.paged_attention_config = paged_attention_config

        # Determine sliding window based on layer index
        self.use_sliding_window = self.layer_idx % 2 == 0
        if self.use_sliding_window:
            # Update config's sliding window based on layer
            object.__setattr__(config, "sliding_window", config.sliding_window)
        else:
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

        # Initialize KV cache
        if create_kv_cache:
            self.kv_cache = init_kv_cache(
                mesh_device=mesh_device,
                config=config,
                mesh_config=mesh_config,
                paged_attention_config=paged_attention_config,
                tensor_cache_path=tensor_cache_path,
            )
            self.layer_past = self.kv_cache  # For tt-transformers compatibility

        # Get KV memory config for decode mode
        self.kv_mem_cfg = get_kv_memory_config(
            mesh_device,
            mesh_config.shard_size(config.num_kv_heads),
            config.head_dim,
        )

        # Store references for backward compatibility
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = config.scaling

    def __call__(self, hidden_states, rope_mats, position_idx=None, page_table=None, kv_cache=None):
        """
        Forward pass - automatically dispatches to decode or prefill.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            rope_mats: Tuple of (cos, sin) matrices for RoPE
            position_idx: Position index for KV cache update
            page_table: Page table for paged attention (optional)
            kv_cache: External KV cache (optional, uses internal if not provided)

        Returns:
            Attention output [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Determine mode based on sequence length
        is_decode = seq_len == 1

        # Use provided kv_cache or internal cache
        cache = kv_cache if kv_cache is not None else self.kv_cache

        # Get transformation matrix for the mode
        mode = "decode" if is_decode else "prefill"
        transformation_mat = self.transformation_mats[mode] if self.transformation_mats else None

        if is_decode:
            return decode_forward(
                hidden_states=hidden_states,
                rope_mats=rope_mats,
                weights=self.weights,
                kv_cache=cache,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                program_config=self.program_config,
                transformation_mat=transformation_mat,
                kv_mem_cfg=self.kv_mem_cfg,
                position_idx=position_idx,
                page_table=page_table,
                ccl_manager=self.ccl_manager,
                activation_dtype=self.activation_dtype,
            )
        else:
            return prefill_forward(
                hidden_states=hidden_states,
                rope_mats=rope_mats,
                weights=self.weights,
                kv_cache=cache,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                program_config=self.program_config,
                transformation_mat=transformation_mat,
                position_idx=position_idx,
                page_table=page_table,
                ccl_manager=self.ccl_manager,
                activation_dtype=self.activation_dtype,
            )
