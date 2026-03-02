# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Vision Transformer block for Molmo2.

Each block consists of:
    1. LayerNorm (attention_norm) -> Attention -> Residual
    2. LayerNorm (ffn_norm) -> MLP -> Residual

Pre-norm architecture with residual connections.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.molmo2.tt.vision_attention import VisionAttention
from models.demos.molmo2.tt.vision_layernorm import VisionLayerNorm
from models.demos.molmo2.tt.vision_mlp import VisionMLP


class VisionBlock(LightweightModule):
    """
    Single transformer block for Molmo2 Vision Transformer.

    Architecture (pre-norm):
        x = x + attention(attention_norm(x))
        x = x + feed_forward(ffn_norm(x))
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        layer_num: int,
        hidden_dim: int = 1152,
        intermediate_dim: int = 4304,
        num_heads: int = 16,
        head_dim: int = 72,
        layer_norm_eps: float = 1e-6,
        weight_cache_path=None,
        state_dict_prefix: str = None,
        dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize VisionBlock.

        Args:
            mesh_device: TTNN mesh device
            state_dict: Model state dict containing weights
            layer_num: Layer index (0-indexed)
            hidden_dim: Model hidden dimension (1152 for Molmo2 ViT)
            intermediate_dim: MLP intermediate dimension
            num_heads: Number of attention heads (16 for Molmo2 ViT)
            head_dim: Dimension per head (72 for Molmo2 ViT)
            layer_norm_eps: Epsilon for LayerNorm
            weight_cache_path: Path to cache weights
            state_dict_prefix: Override prefix for state dict keys
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num

        # Determine state dict prefix (full path including model.vision_backbone...)
        if state_dict_prefix is None:
            state_dict_prefix = f"model.vision_backbone.image_vit.transformer.resblocks.{layer_num}"
        # Note: state_dict_prefix should be the full key path, not just the relative path

        # Pre-attention LayerNorm
        self.attention_norm = VisionLayerNorm(
            device=mesh_device,
            dim=hidden_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}.attention_norm",
            weight_cache_path=weight_cache_path,
            eps=layer_norm_eps,
        )

        # Attention
        self.attention = VisionAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}.attention",
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

        # Pre-MLP LayerNorm
        self.ffn_norm = VisionLayerNorm(
            device=mesh_device,
            dim=hidden_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}.ffn_norm",
            weight_cache_path=weight_cache_path,
            eps=layer_norm_eps,
        )

        # MLP (feed_forward)
        self.feed_forward = VisionMLP(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}.feed_forward",
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass through the vision block.

        Args:
            x: Input tensor of shape [1, 1, seq_len, hidden_dim]

        Returns:
            Output tensor of shape [1, 1, seq_len, hidden_dim]

        Note: Does NOT deallocate the input tensor to allow multi-scale feature
        extraction (hidden states from multiple layers are needed).
        Following tt_transformers' pattern for traceable ViT.
        """
        # Pre-norm attention with residual
        # Note: We don't deallocate input x - caller manages lifetime
        attn_out = self.attention_norm(x)
        attn_out = self.attention(attn_out)
        res = ttnn.add(x, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        # Pre-norm MLP with residual
        mlp_out = self.ffn_norm(res)
        mlp_out = self.feed_forward(mlp_out)
        out = ttnn.add(res, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(res)

        return out
