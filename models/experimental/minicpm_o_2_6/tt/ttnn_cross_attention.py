# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Cross-Attention layer for MiniCPM-o-2_6.

Cross-attention layers are added to specific Qwen2 transformer layers (8, 16, 24)
to attend to multimodal embeddings (vision/audio features).
"""

import ttnn
from typing import Optional
from loguru import logger

try:
    pass
except ImportError:
    pass


class TtnnCrossAttention:
    """
    TTNN implementation of Cross-Attention layer for multimodal fusion.

    This layer performs cross-attention between LLM hidden states (queries)
    and multimodal embeddings (keys/values) that have been processed through
    the resampler layers.

    Uses tt_transformers-style compute configs and memory layouts for consistency.

    Architecture:
        - Query projection: from LLM hidden states
        - Key/Value projections: from multimodal embeddings
        - Multi-head cross-attention computation
        - Output projection

    Args:
        device: TTNN device (mesh device)
        hidden_size: LLM hidden dimension (default 3584 for Qwen2.5)
        num_attention_heads: Number of attention heads (default 28)
        num_key_value_heads: Number of key/value heads for GQA (default 4)
        layer_idx: Layer index (for cross-attention layer identification)
        model_args: ModelArgs from tt_transformers (for proper configs)
    """

    def __init__(
        self,
        device: ttnn.Device,
        hidden_size: int = 3584,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        layer_idx: int = 0,
        model_args=None,
    ):
        self.device = device
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.layer_idx = layer_idx
        self.model_args = model_args

        # Derived dimensions
        self.head_dim = hidden_size // num_attention_heads
        self.num_local_heads = num_attention_heads  # For single device
        self.num_local_kv_heads = num_key_value_heads  # For single device

        # Get compute configs from model_args if available, otherwise use defaults
        if model_args and hasattr(model_args, "get_model_config"):
            model_config = model_args.get_model_config()
            # Use the same compute configs as tt_transformers for consistency
            self.compute_kernel_config = model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
                decoder_id=layer_idx, op=ttnn.transformer.OpGroup.LI_QKV, configuration=model_args
            )
            self.sdpa_compute_kernel_config = model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
                decoder_id=layer_idx, op=ttnn.transformer.OpGroup.SDPA_PREFILL, configuration=model_args
            )
            self.activation_dtype = model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
                decoder_id=layer_idx, tensor=ttnn.transformer.TensorGroup.ACTIVATION
            )
        else:
            # Fallback to HiFi4 configs (matching tt_transformers)
            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4)
            self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4)
            self.activation_dtype = ttnn.bfloat8_b

        # Scale factor for attention (following tt_transformers)
        self.scale = self.head_dim**-0.5

        # Default dtype (following TTNN LLM patterns)
        self.dtype = ttnn.bfloat16

        # Weights will be loaded later (following TTNN LLM naming)
        self.wq = None
        self.wk = None
        self.wv = None
        self.wo = None

        # RMS norms (following TTNN LLM pattern)
        self.q_norm = None
        self.k_norm = None

        logger.info(
            f"TtnnCrossAttention initialized: hidden_size={hidden_size}, "
            f"num_heads={num_attention_heads}, num_kv_heads={num_key_value_heads}, "
            f"layer_idx={layer_idx}, compute_config={self.compute_kernel_config}"
        )

    def load_weights(self, weights_dict: dict):
        """
        Load weights for the cross-attention layer following TTNN LLM patterns.

        Args:
            weights_dict: Dictionary containing weight tensors
        """
        # Import RMSNorm from TTNN common
        from models.common.rmsnorm import RMSNorm

        # Query projection weights (following TTNN LLM naming)
        self.wq = ttnn.as_tensor(
            weights_dict["q_proj.weight"].transpose(-2, -1),
            device=self.device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=-1),
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        # Key projection weights
        self.wk = ttnn.as_tensor(
            weights_dict["k_proj.weight"].transpose(-2, -1),
            device=self.device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=-1),
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        # Value projection weights
        self.wv = ttnn.as_tensor(
            weights_dict["v_proj.weight"].transpose(-2, -1),
            device=self.device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=-1),
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        # Output projection weights
        self.wo = ttnn.as_tensor(
            weights_dict["o_proj.weight"].transpose(-2, -1),
            device=self.device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=-2),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
        )

        # RMS norms (following TTNN LLM pattern)
        # RMS norms (following TTNN LLM pattern)
        # Do NOT synthesize or invent missing weights — respect the checkpoint.
        if "q_norm.weight" in weights_dict or "q_norm" in weights_dict:
            self.q_norm = RMSNorm(
                device=self.device,
                dim=self.head_dim,
                state_dict=weights_dict,
                state_dict_prefix="",
                weight_cache_path=None,
                weight_key="q_norm",
                eps=1e-6,
            )
        else:
            self.q_norm = None
            logger.info("q_norm.weight not present in weights_dict; skipping Q RMSNorm (no fake weights)")

        if "k_norm.weight" in weights_dict or "k_norm" in weights_dict:
            self.k_norm = RMSNorm(
                device=self.device,
                dim=self.head_dim,
                state_dict=weights_dict,
                state_dict_prefix="",
                weight_cache_path=None,
                weight_key="k_norm",
                eps=1e-6,
            )
        else:
            self.k_norm = None
            logger.info("k_norm.weight not present in weights_dict; skipping K RMSNorm (no fake weights)")

        logger.info("✅ Cross-Attention weights loaded with TTNN optimizations")

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        multimodal_embeds: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Forward pass of cross-attention layer using TTNN optimized operations.

        Args:
            hidden_states: LLM hidden states [batch_size, seq_len, hidden_size]
            multimodal_embeds: Multimodal embeddings [batch_size, multimodal_seq_len, multimodal_dim]
            attention_mask: Optional attention mask [batch_size, seq_len, multimodal_seq_len]

        Returns:
            ttnn.Tensor: Cross-attended output [batch_size, seq_len, hidden_size]
        """
        seq_len = hidden_states.shape[-2]
        multimodal_seq_len = multimodal_embeds.shape[-2]

        # Query projection (following tt_transformers patterns)
        xq = ttnn.linear(
            hidden_states,
            self.wq,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Key/Value projections from multimodal embeddings
        xk = ttnn.linear(
            multimodal_embeds,
            self.wk,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        xv = ttnn.linear(
            multimodal_embeds,
            self.wv,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Get batch size
        batch_size = hidden_states.shape[0]

        # Reshape for multi-head attention (following standard attention format)
        # xq: [batch, seq_len, hidden_size] -> [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        xq = ttnn.to_layout(xq, layout=ttnn.ROW_MAJOR_LAYOUT)
        xq = ttnn.reshape(xq, (batch_size, seq_len, self.num_attention_heads, self.head_dim))
        xq = ttnn.to_layout(xq, layout=ttnn.TILE_LAYOUT)
        xq = ttnn.permute(xq, (0, 2, 1, 3))  # [batch, num_heads, seq_len, head_dim]

        # Apply Q RMS norm after head creation (if present in checkpoint)
        if self.q_norm is not None:
            xq = self.q_norm(xq, mode="prefill")

        # xk: [batch, multimodal_seq_len, hidden_size] -> [batch, multimodal_seq_len, num_kv_heads, head_dim] -> [batch, num_kv_heads, multimodal_seq_len, head_dim]
        xk = ttnn.to_layout(xk, layout=ttnn.ROW_MAJOR_LAYOUT)
        xk = ttnn.reshape(xk, (batch_size, multimodal_seq_len, self.num_key_value_heads, self.head_dim))
        xk = ttnn.to_layout(xk, layout=ttnn.TILE_LAYOUT)
        xk = ttnn.permute(xk, (0, 2, 1, 3))  # [batch, num_kv_heads, multimodal_seq_len, head_dim]

        # xv: same as xk
        xv = ttnn.to_layout(xv, layout=ttnn.ROW_MAJOR_LAYOUT)
        xv = ttnn.reshape(xv, (batch_size, multimodal_seq_len, self.num_key_value_heads, self.head_dim))
        xv = ttnn.to_layout(xv, layout=ttnn.TILE_LAYOUT)
        xv = ttnn.permute(xv, (0, 2, 1, 3))  # [batch, num_kv_heads, multimodal_seq_len, head_dim]

        # Apply K RMS norm after head creation (if present in checkpoint)
        if self.k_norm is not None:
            xk = self.k_norm(xk, mode="prefill")

        # Get SDPA program config following tt_transformers pattern
        if self.model_args and hasattr(self.model_args, "get_model_config"):
            model_config = self.model_args.get_model_config()
            program_config = model_config["SDPA_PROGCFG"](seq_len)
        else:
            # Fallback program config
            program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=128,
                k_chunk_size=128,
                exp_approx_mode=False,
            )

        # Perform attention using TTNN's optimized SDPA (following tt_transformers)
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            is_causal=False,  # Cross-attention is not causal
            attn_mask=attention_mask,
            scale=self.scale,
            program_config=program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )

        # Concatenate heads using TTNN's optimized function
        attn_output = ttnn.experimental.nlp_concat_heads(attn_output)

        # Output projection (following tt_transformers patterns)
        output = ttnn.matmul(
            attn_output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Reshape back to original format
        output = ttnn.to_layout(output, layout=ttnn.ROW_MAJOR_LAYOUT)
        output = ttnn.reshape(output, (batch_size, seq_len, self.hidden_size))
        output = ttnn.to_layout(output, layout=ttnn.TILE_LAYOUT)

        return output
