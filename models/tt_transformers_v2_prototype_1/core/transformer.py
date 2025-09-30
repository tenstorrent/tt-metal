# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Transformer block module - combines attention and MLP"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

import ttnn

from .attention import Attention, AttentionConfig
from .mlp import MLP, MLPConfig
from .norm import LayerNorm, NormConfig, RMSNorm


@dataclass
class TransformerBlockConfig:
    """Configuration for TransformerBlock module"""

    hidden_size: int
    num_heads: int
    num_kv_heads: Optional[int] = None
    intermediate_size: int = None
    norm_type: str = "rmsnorm"  # Options: "rmsnorm", "layernorm"
    norm_eps: float = 1e-6
    activation: str = "silu"
    use_parallel_residual: bool = False  # Parallel attention/MLP like GPT-J
    dropout: float = 0.0

    def __post_init__(self):
        if self.intermediate_size is None:
            # Default to 4x hidden size
            self.intermediate_size = 4 * self.hidden_size


class TransformerBlock(torch.nn.Module):
    """
    Transformer block combining self-attention and MLP.

    Supports both sequential (standard) and parallel residual connections.
    """

    def __init__(
        self,
        config: TransformerBlockConfig,
        device: ttnn.Device,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.layer_idx = layer_idx

        # Create attention module
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
        )
        self.attention = Attention(attn_config, device, layer_idx)

        # Create MLP module
        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=config.activation,
            dropout=config.dropout,
        )
        self.mlp = MLP(mlp_config, device)

        # Create normalization modules
        norm_config = NormConfig(
            normalized_shape=config.hidden_size,
            eps=config.norm_eps,
        )

        if config.norm_type == "rmsnorm":
            self.norm1 = RMSNorm(norm_config, device)
            self.norm2 = RMSNorm(norm_config, device)
        elif config.norm_type == "layernorm":
            self.norm1 = LayerNorm(norm_config, device)
            self.norm2 = LayerNorm(norm_config, device)
        else:
            raise ValueError(f"Unknown norm type: {config.norm_type}")

    def setup_weights(self, weights: Dict[str, ttnn.Tensor]):
        """Setup all weights from a dictionary"""
        # Attention weights
        self.attention.setup_weights(
            wq=weights.get(f"attention.wq"),
            wk=weights.get(f"attention.wk"),
            wv=weights.get(f"attention.wv"),
            wo=weights.get(f"attention.wo"),
        )

        # MLP weights
        self.mlp.setup_weights(
            w1=weights.get(f"mlp.w1"),
            w2=weights.get(f"mlp.w2"),
            w3=weights.get(f"mlp.w3"),  # Optional for gated variants
        )

        # Norm weights
        if isinstance(self.norm1, RMSNorm):
            self.norm1.setup_weight(weights.get(f"norm1.weight"))
            self.norm2.setup_weight(weights.get(f"norm2.weight"))
        else:
            self.norm1.setup_parameters(
                weight=weights.get(f"norm1.weight"),
                bias=weights.get(f"norm1.bias"),
            )
            self.norm2.setup_parameters(
                weight=weights.get(f"norm2.weight"),
                bias=weights.get(f"norm2.bias"),
            )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        rotary_embeddings: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Forward pass of transformer block.

        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            rotary_embeddings: Optional rotary embeddings (cos, sin)
            kv_cache: Optional KV cache for autoregressive generation
            use_cache: Whether to return updated KV cache

        Returns:
            hidden_states: Output tensor
            kv_cache: Updated KV cache if use_cache is True
        """
        if self.config.use_parallel_residual:
            # Parallel residual (GPT-J style)
            return self._forward_parallel(
                hidden_states,
                attention_mask,
                position_ids,
                rotary_embeddings,
                kv_cache,
                use_cache,
            )
        else:
            # Sequential residual (standard transformer)
            return self._forward_sequential(
                hidden_states,
                attention_mask,
                position_ids,
                rotary_embeddings,
                kv_cache,
                use_cache,
            )

    def _forward_sequential(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor],
        position_ids: Optional[ttnn.Tensor],
        rotary_embeddings: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]],
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]],
        use_cache: bool,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """Sequential residual connection (standard transformer)"""
        residual = hidden_states

        # Self-attention
        hidden_states = self.norm1(hidden_states)
        attn_output, new_kv_cache = self.attention(
            hidden_states,
            attention_mask,
            position_ids,
            rotary_embeddings,
            kv_cache,
            use_cache,
        )
        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, new_kv_cache

    def _forward_parallel(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor],
        position_ids: Optional[ttnn.Tensor],
        rotary_embeddings: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]],
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]],
        use_cache: bool,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """Parallel residual connection (GPT-J style)"""
        residual = hidden_states

        # Normalize input for both paths
        norm_hidden_states = self.norm1(hidden_states)

        # Self-attention path
        attn_output, new_kv_cache = self.attention(
            norm_hidden_states,
            attention_mask,
            position_ids,
            rotary_embeddings,
            kv_cache,
            use_cache,
        )

        # MLP path (using same normalized input)
        mlp_output = self.mlp(norm_hidden_states)

        # Combine residual + attention + mlp
        hidden_states = residual + attn_output + mlp_output

        return hidden_states, new_kv_cache


class CrossAttentionTransformerBlock(torch.nn.Module):
    """
    Transformer block with cross-attention for encoder-decoder models.

    Includes self-attention, cross-attention, and MLP.
    """

    def __init__(
        self,
        config: TransformerBlockConfig,
        device: ttnn.Device,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.layer_idx = layer_idx

        # Create modules
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
        )

        # Self-attention
        self.self_attention = Attention(attn_config, device, layer_idx)

        # Cross-attention
        self.cross_attention = Attention(attn_config, device, layer_idx)

        # MLP
        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=config.activation,
        )
        self.mlp = MLP(mlp_config, device)

        # Normalization
        norm_config = NormConfig(
            normalized_shape=config.hidden_size,
            eps=config.norm_eps,
        )

        if config.norm_type == "rmsnorm":
            self.norm1 = RMSNorm(norm_config, device)
            self.norm2 = RMSNorm(norm_config, device)
            self.norm3 = RMSNorm(norm_config, device)
        else:
            self.norm1 = LayerNorm(norm_config, device)
            self.norm2 = LayerNorm(norm_config, device)
            self.norm3 = LayerNorm(norm_config, device)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        encoder_attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        rotary_embeddings: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        self_attn_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        cross_attn_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[Tuple[ttnn.Tensor, ttnn.Tensor], Tuple[ttnn.Tensor, ttnn.Tensor]]]]:
        """
        Forward pass with cross-attention.

        Returns:
            hidden_states: Output tensor
            caches: Tuple of (self_attn_cache, cross_attn_cache) if use_cache is True
        """
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        self_attn_output, new_self_cache = self.self_attention(
            hidden_states,
            attention_mask,
            position_ids,
            rotary_embeddings,
            self_attn_cache,
            use_cache,
        )
        hidden_states = residual + self_attn_output

        # Cross-attention
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        # For cross-attention, Q comes from decoder, K/V from encoder
        cross_attn_output, new_cross_cache = self.cross_attention(
            hidden_states,  # Query from decoder
            encoder_attention_mask,
            None,  # No position IDs for cross-attention
            None,  # No rotary embeddings for cross-attention
            cross_attn_cache,
            use_cache,
        )
        hidden_states = residual + cross_attn_output

        # MLP
        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        # Return caches if requested
        if use_cache:
            return hidden_states, (new_self_cache, new_cross_cache)
        return hidden_states, None
