# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Core transformer building blocks using TTNN APIs.

These components implement the transformer layers used by the CosyVoice LLM
backbone. Each block is implemented as a standalone TTNN operation graph.
"""

from typing import Optional, Tuple

import torch
import ttnn
from loguru import logger


class TtLayerNorm:
    """Layer normalization using TTNN API."""

    def __init__(
        self,
        device: ttnn.Device,
        hidden_size: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.hidden_size = hidden_size
        self.eps = eps

        # Convert weights to TTNN tensors
        self.weight = ttnn.as_tensor(
            weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )
        if bias is not None:
            self.bias = ttnn.as_tensor(
                bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
                dtype=dtype,
            )
        else:
            self.bias = None

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Normalized tensor
        """
        if self.bias is not None:
            return ttnn.layer_norm(x, weight=self.weight, bias=self.bias, epsilon=self.eps)
        return ttnn.layer_norm(x, weight=self.weight, epsilon=self.eps)


class TtMultiHeadAttention:
    """Multi-head attention using TTNN API."""

    def __init__(
        self,
        device: ttnn.Device,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        o_weight: torch.Tensor,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Projection weights
        self.q_weight = ttnn.as_tensor(
            q_weight.unsqueeze(0).unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )
        self.k_weight = ttnn.as_tensor(
            k_weight.unsqueeze(0).unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )
        self.v_weight = ttnn.as_tensor(
            v_weight.unsqueeze(0).unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )
        self.o_weight = ttnn.as_tensor(
            o_weight.unsqueeze(0).unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )

    def __call__(
        self,
        x: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
    ) -> ttnn.Tensor:
        """Apply multi-head attention.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional causal mask
            kv_cache: Optional (key_cache, value_cache) tuple for autoregressive decoding

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # Project to Q, K, V
        q = ttnn.linear(x, self.q_weight)
        k = ttnn.linear(x, self.k_weight)
        v = ttnn.linear(x, self.v_weight)

        # Reshape for multi-head attention
        # [batch, seq_len, hidden_size] -> [batch, seq_len, num_heads, head_dim]
        B, S, D = x.shape[0], x.shape[1], x.shape[2]
        q = ttnn.reshape(q, (B, S, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (B, S, self.num_heads, self.head_dim))
        v = ttnn.reshape(v, (B, S, self.num_heads, self.head_dim))

        # Update KV cache if provided
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Concatenate new K/V with cache
            k = ttnn.concat((k_cache, k), dim=1)
            v = ttnn.concat((v_cache, v), dim=1)

        # Scaled dot-product attention
        # Use TTNN's built-in attention operations
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q, k, v,
            is_causal=(attention_mask is not None),
        )

        # Reshape back to [batch, seq_len, hidden_size]
        attn_output = ttnn.reshape(attn_output, (B, S, self.hidden_size))

        # Output projection
        output = ttnn.linear(attn_output, self.o_weight)

        return output


class TtFeedForward:
    """Feed-forward network using TTNN API."""

    def __init__(
        self,
        device: ttnn.Device,
        hidden_size: int,
        intermediate_size: int,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # SwiGLU variant: gate_proj, up_proj, down_proj
        self.gate_weight = ttnn.as_tensor(
            gate_weight.unsqueeze(0).unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )
        self.up_weight = ttnn.as_tensor(
            up_weight.unsqueeze(0).unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )
        self.down_weight = ttnn.as_tensor(
            down_weight.unsqueeze(0).unsqueeze(0),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            dtype=dtype,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Apply feed-forward network (SwiGLU).

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        # SwiGLU: down_proj(gate_proj(x) * silu(up_proj(x)))
        gate = ttnn.linear(x, self.gate_weight)
        up = ttnn.linear(x, self.up_weight)

        # Apply SiLU activation and gate
        gate = ttnn.silu(gate)
        hidden = ttnn.mul(gate, up)

        # Down projection
        output = ttnn.linear(hidden, self.down_weight)

        return output


class TtTransformerBlock:
    """Single transformer decoder block using TTNN API."""

    def __init__(
        self,
        device: ttnn.Device,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
        attention_params: dict,
        ffn_params: dict,
        norm_weight: torch.Tensor,
        norm_bias: Optional[torch.Tensor] = None,
        post_attention_norm_weight: Optional[torch.Tensor] = None,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size

        # Pre-attention layer norm
        self.input_layernorm = TtLayerNorm(
            device, hidden_size, norm_weight, norm_bias,
            memory_config=memory_config, dtype=dtype,
        )

        # Multi-head attention
        self.attention = TtMultiHeadAttention(
            device, hidden_size, num_heads, head_dim,
            attention_params["q_weight"],
            attention_params["k_weight"],
            attention_params["v_weight"],
            attention_params["o_weight"],
            memory_config=memory_config, dtype=dtype,
        )

        # Post-attention layer norm
        if post_attention_norm_weight is not None:
            self.post_attention_layernorm = TtLayerNorm(
                device, hidden_size, post_attention_norm_weight, None,
                memory_config=memory_config, dtype=dtype,
            )
        else:
            self.post_attention_layernorm = None

        # Feed-forward network
        self.feed_forward = TtFeedForward(
            device, hidden_size, intermediate_size,
            ffn_params["gate_weight"],
            ffn_params["up_weight"],
            ffn_params["down_weight"],
            memory_config=memory_config, dtype=dtype,
        )

    def __call__(
        self,
        x: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
    ) -> ttnn.Tensor:
        """Apply transformer block.

        Args:
            x: Input tensor
            attention_mask: Optional causal mask
            kv_cache: Optional KV cache for autoregressive decoding

        Returns:
            Output tensor
        """
        # Pre-attention norm
        residual = x
        x = self.input_layernorm(x)

        # Attention
        x = self.attention(x, attention_mask, kv_cache)

        # Residual connection
        x = ttnn.add(x, residual)

        # Post-attention norm (if using pre-norm with post-norm)
        if self.post_attention_layernorm is not None:
            residual = x
            x = self.post_attention_layernorm(x)
        else:
            residual = x

        # Feed-forward
        x = self.feed_forward(x)

        # Residual connection
        x = ttnn.add(x, residual)

        return x
