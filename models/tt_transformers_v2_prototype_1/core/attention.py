# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Attention module - core component of transformer architecture"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

import ttnn


@dataclass
class AttentionConfig:
    """Configuration for Attention module"""

    hidden_size: int
    num_heads: int
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_seq_len: int = 2048
    use_rotary_embeddings: bool = True
    attention_dropout: float = 0.0

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"


class Attention(torch.nn.Module):
    """
    Multi-head attention module with minimal dependencies.

    This module implements the core attention mechanism used in transformer models.
    It supports both multi-head attention (MHA) and grouped-query attention (GQA).
    """

    def __init__(
        self,
        config: AttentionConfig,
        device: ttnn.Device,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.layer_idx = layer_idx

        # Calculate local heads for distributed execution
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        # Initialize weights (will be loaded from state dict)
        self.wq = None
        self.wk = None
        self.wv = None
        self.wo = None

        # KV cache for autoregressive generation
        self.kv_cache = None

    def setup_weights(self, wq: ttnn.Tensor, wk: ttnn.Tensor, wv: ttnn.Tensor, wo: ttnn.Tensor):
        """Set pre-loaded weights for the attention module"""
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def forward(
        self,
        x: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        rotary_embeddings: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Forward pass of attention module.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask for attention scores
            position_ids: Optional position IDs for rotary embeddings
            rotary_embeddings: Optional pre-computed rotary embeddings
            kv_cache: Optional KV cache for autoregressive generation
            use_cache: Whether to return updated KV cache

        Returns:
            output: Attention output of shape [batch_size, seq_len, hidden_size]
            kv_cache: Updated KV cache if use_cache is True
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V projections
        q = ttnn.matmul(x, self.wq)
        k = ttnn.matmul(x, self.wk)
        v = ttnn.matmul(x, self.wv)

        # Reshape for multi-head attention
        q = self._reshape_for_attention(q, self.num_heads)
        k = self._reshape_for_attention(k, self.num_kv_heads)
        v = self._reshape_for_attention(v, self.num_kv_heads)

        # Apply rotary embeddings if enabled
        if self.config.use_rotary_embeddings and rotary_embeddings is not None:
            cos, sin = rotary_embeddings
            q, k = self._apply_rotary_embeddings(q, k, cos, sin)

        # Handle KV cache for autoregressive generation
        if kv_cache is not None:
            k, v = self._update_kv_cache(k, v, kv_cache)

        # Compute attention scores
        attn_output = self._compute_attention(q, k, v, attention_mask)

        # Reshape back to hidden size
        attn_output = self._reshape_from_attention(attn_output)

        # Output projection
        output = ttnn.matmul(attn_output, self.wo)

        # Return output and optionally updated cache
        if use_cache:
            return output, (k, v)
        return output, None

    def _reshape_for_attention(self, x: ttnn.Tensor, num_heads: int) -> ttnn.Tensor:
        """Reshape tensor for multi-head attention"""
        batch_size, seq_len, hidden_size = x.shape
        head_dim = hidden_size // num_heads

        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        x = ttnn.reshape(x, [batch_size, seq_len, num_heads, head_dim])

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        return ttnn.transpose(x, 1, 2)

    def _reshape_from_attention(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Reshape tensor from multi-head attention format back to hidden size"""
        batch_size, num_heads, seq_len, head_dim = x.shape

        # Transpose back to [batch_size, seq_len, num_heads, head_dim]
        x = ttnn.transpose(x, 1, 2)

        # Reshape to [batch_size, seq_len, hidden_size]
        return ttnn.reshape(x, [batch_size, seq_len, num_heads * head_dim])

    def _apply_rotary_embeddings(
        self, q: ttnn.Tensor, k: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Apply rotary position embeddings to Q and K"""
        # This is a simplified implementation
        # Real implementation would use optimized RoPE kernels
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot

    def _rotate_half(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Rotate half the hidden dims of the input for RoPE"""
        # Simplified - actual implementation would be optimized
        x1, x2 = ttnn.split(x, x.shape[-1] // 2, dim=-1)
        return ttnn.concat([-x2, x1], dim=-1)

    def _compute_attention(
        self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, attention_mask: Optional[ttnn.Tensor] = None
    ) -> ttnn.Tensor:
        """Compute scaled dot-product attention"""
        # Scale factor
        scale = 1.0 / (self.head_dim**0.5)

        # Compute attention scores
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1)) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax
        attn_weights = ttnn.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = ttnn.matmul(attn_weights, v)

        return attn_output

    def _update_kv_cache(
        self, k: ttnn.Tensor, v: ttnn.Tensor, kv_cache: Tuple[ttnn.Tensor, ttnn.Tensor]
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Update KV cache for autoregressive generation"""
        k_cache, v_cache = kv_cache

        # Concatenate along sequence dimension
        k = ttnn.concat([k_cache, k], dim=2)
        v = ttnn.concat([v_cache, v], dim=2)

        return k, v
