# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Transformer Decoder Block for Molmo2 Text Model.

Implements a pre-norm decoder block:
    x = x + attention(attn_norm(x))
    x = x + mlp(ff_norm(x))

With RMSNorm (no bias) following Llama architecture.
"""

from typing import Dict, List, Optional, Tuple

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.molmo2.tt.text_attention import TextAttention
from models.demos.molmo2.tt.text_mlp import TextMLP
from models.demos.molmo2.tt.text_rmsnorm import TextRMSNorm


class TextBlock(LightweightModule):
    """
    Single transformer decoder block for Molmo2 text model.

    Pre-norm architecture with RMSNorm, GQA attention with QK-norm,
    and SwiGLU MLP.
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        layer_num: int,
        hidden_dim: int = 4096,
        intermediate_dim: int = 12288,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        max_seq_len: int = 8192,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-5,
        weight_cache_path=None,
        state_dict_prefix: str = "model.transformer.blocks",
        dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize TextBlock.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Model state dict containing weights
            layer_num: Layer number (0-35)
            hidden_dim: Hidden dimension (4096)
            intermediate_dim: MLP intermediate dimension (11008)
            num_heads: Number of query heads (32)
            num_kv_heads: Number of KV heads (8)
            head_dim: Dimension per head (128)
            max_seq_len: Maximum sequence length (8192)
            rope_theta: RoPE theta (1,000,000)
            rms_norm_eps: Epsilon for RMSNorm
            weight_cache_path: Path to cache weights
            state_dict_prefix: Prefix for state dict keys
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.layer_num = layer_num

        # Layer prefix
        prefix = f"{state_dict_prefix}.{layer_num}"

        # Attention normalization
        self.attn_norm = TextRMSNorm(
            mesh_device=mesh_device,
            state_dict=state_dict,
            hidden_dim=hidden_dim,
            eps=rms_norm_eps,
            weight_cache_path=weight_cache_path,
            state_dict_prefix=f"{prefix}.attn_norm",
        )

        # Self-attention with GQA and QK-norm
        self.self_attn = TextAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            layer_num=layer_num,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            weight_cache_path=weight_cache_path,
            state_dict_prefix=state_dict_prefix,
            dtype=dtype,
        )

        # Feed-forward normalization
        self.ff_norm = TextRMSNorm(
            mesh_device=mesh_device,
            state_dict=state_dict,
            hidden_dim=hidden_dim,
            eps=rms_norm_eps,
            weight_cache_path=weight_cache_path,
            state_dict_prefix=f"{prefix}.ff_norm",
        )

        # Feed-forward network (SwiGLU)
        self.mlp = TextMLP(
            mesh_device=mesh_device,
            state_dict=state_dict,
            layer_num=layer_num,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            weight_cache_path=weight_cache_path,
            state_dict_prefix=state_dict_prefix,
            dtype=dtype,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        rot_mats: List[ttnn.Tensor],
        transformation_mats: Dict[str, ttnn.Tensor],
        attn_mask: Optional[ttnn.Tensor] = None,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Forward pass through decoder block.

        Args:
            x: Input tensor of shape [1, 1, seq_len, hidden_dim]
            rot_mats: List of [cos, sin] rotation matrices
            transformation_mats: Dict with 'decode' and 'prefill' transformation matrices
            attn_mask: Optional attention mask
            start_pos: Starting position for KV cache
            kv_cache: Optional (k_cache, v_cache) tuple

        Returns:
            Tuple of (output, updated_kv_cache)
        """
        # Attention block with residual
        residual = x
        x = self.attn_norm(x)
        attn_out, new_kv_cache = self.self_attn(x, rot_mats, transformation_mats, attn_mask, start_pos, kv_cache)
        x = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        # MLP block with residual
        residual = x
        x = self.ff_norm(x)
        mlp_out = self.mlp(x)
        x = ttnn.add(residual, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_out)

        return x, new_kv_cache

    def forward_decode(
        self,
        x: ttnn.Tensor,
        rot_mats: List[ttnn.Tensor],
        transformation_mat: ttnn.Tensor,
        kv_cache: Tuple[ttnn.Tensor, ttnn.Tensor],
        current_pos: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Decode-mode forward pass (single token at a time).

        Args:
            x: Input tensor of shape [1, 1, 1, hidden_dim]
            rot_mats: List of [cos, sin] rotation matrices
            transformation_mat: RoPE transformation matrix for decode
            kv_cache: (k_cache, v_cache) pre-allocated tensors
            current_pos: Current decode position tensor

        Returns:
            Output tensor
        """
        # Attention block with residual
        residual = x
        x = self.attn_norm(x)
        attn_out = self.self_attn.forward_decode(x, rot_mats, transformation_mat, kv_cache, current_pos)
        x = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        # MLP block with residual
        residual = x
        x = self.ff_norm(x)
        mlp_out = self.mlp(x)
        x = ttnn.add(residual, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_out)

        return x
