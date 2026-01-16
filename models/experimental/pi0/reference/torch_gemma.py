# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Gemma transformer blocks - PyTorch Reference Implementation.

This module implements Gemma 2B style transformer layers with:
    - RMSNorm (pre-normalization)
    - Multi-Query Attention (MQA) with num_kv_heads=1
    - GeGLU MLP (gated GELU activation)
    - Rotary Position Embeddings (RoPE)

Architecture configurations:
    - Gemma 2B (VLM): width=2048, depth=18, mlp_dim=16384, heads=8, kv_heads=1
    - Gemma 300M (Expert): width=1024, depth=18, mlp_dim=4096, heads=8, kv_heads=1
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from models.experimental.pi0.common.configs import GemmaConfig


# ============================================================================
# RMSNorm
# ============================================================================


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    RMSNorm with Gemma-style weight offset.

    Gemma uses (weight + 1) instead of just weight.

    Args:
        x: Input tensor (batch_size, seq_len, hidden_dim)
        weight: Learnable weight (hidden_dim,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor of same shape as input
    """
    # Compute RMS
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normalized = x * torch.rsqrt(variance + eps)

    # Apply Gemma-style weight (weight + 1)
    return x_normalized * (weight + 1.0)


# ============================================================================
# Rotary Position Embeddings
# ============================================================================


def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos and sin for rotary embeddings.

    Args:
        head_dim: Dimension per head (must be even)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation
        dtype: Output dtype
        device: Device to create tensors on

    Returns:
        Tuple of (cos, sin) each of shape (max_seq_len, head_dim // 2)
    """
    # Compute inverse frequencies
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))

    # Compute positions
    t = torch.arange(max_seq_len, device=device, dtype=dtype)

    # Outer product
    freqs_outer = torch.outer(t, freqs)

    return torch.cos(freqs_outer), torch.sin(freqs_outer)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Q and K.

    Args:
        q: Query tensor (batch, num_heads, seq_len, head_dim)
        k: Key tensor (batch, num_kv_heads, seq_len, head_dim)
        cos: Cosine frequencies (max_seq_len, head_dim // 2)
        sin: Sine frequencies (max_seq_len, head_dim // 2)
        position_ids: Position indices for each position (batch, seq_len)

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    seq_len = q.shape[2]
    head_dim = q.shape[-1]

    # Select cos/sin based on position IDs or use sequential
    if position_ids is not None:
        cos = cos[position_ids]  # (batch, seq_len, head_dim // 2)
        sin = sin[position_ids]
        # Add head dimension
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim // 2)
        sin = sin.unsqueeze(1)
    else:
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim // 2)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # Duplicate for full head_dim
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

    # Apply rotation
    # Split and interleave for rotation
    q1, q2 = q[..., : head_dim // 2], q[..., head_dim // 2 :]
    k1, k2 = k[..., : head_dim // 2], k[..., head_dim // 2 :]

    q_rotated = torch.cat(
        [
            q1 * cos[..., : head_dim // 2] - q2 * sin[..., : head_dim // 2],
            q1 * sin[..., head_dim // 2 :] + q2 * cos[..., head_dim // 2 :],
        ],
        dim=-1,
    )
    k_rotated = torch.cat(
        [
            k1 * cos[..., : head_dim // 2] - k2 * sin[..., : head_dim // 2],
            k1 * sin[..., head_dim // 2 :] + k2 * cos[..., head_dim // 2 :],
        ],
        dim=-1,
    )

    return q_rotated, k_rotated


# ============================================================================
# Multi-Query Attention
# ============================================================================


class GemmaAttention:
    """
    Gemma Multi-Query Attention (PyTorch).

    Uses 8 query heads but only 1 key-value head, which is broadcast
    across all query heads during attention computation.
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ):
        """
        Initialize attention layer.

        Args:
            config: Gemma configuration
            weights: Layer weights with keys:
                - self_attn.q_proj.weight
                - self_attn.k_proj.weight
                - self_attn.v_proj.weight
                - self_attn.o_proj.weight
            layer_idx: Layer index for caching
        """
        self.config = config
        self.layer_idx = layer_idx

        self.q_proj = weights["self_attn.q_proj.weight"]
        self.k_proj = weights["self_attn.k_proj.weight"]
        self.v_proj = weights["self_attn.v_proj.weight"]
        self.o_proj = weights["self_attn.o_proj.weight"]

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for attention.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            cos, sin: RoPE embeddings
            attention_mask: (batch, 1, seq_len, kv_len) additive mask
            position_ids: Position indices
            past_key_value: Cached (K, V) from previous forward
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (output, optional_new_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V (ensure dtype compatibility)
        q_proj = self.q_proj.to(hidden_states.dtype)
        k_proj = self.k_proj.to(hidden_states.dtype)
        v_proj = self.v_proj.to(hidden_states.dtype)
        q = F.linear(hidden_states, q_proj)
        k = F.linear(hidden_states, k_proj)
        v = F.linear(hidden_states, v_proj)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, cos, sin, position_ids)

        # Handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # Expand K, V for broadcast with Q
        # K, V: (batch, 1, kv_len, head_dim) -> (batch, num_heads, kv_len, head_dim)
        kv_len = k.shape[2]
        k_expanded = k.expand(batch_size, self.num_heads, kv_len, self.head_dim)
        v_expanded = v.expand(batch_size, self.num_heads, kv_len, self.head_dim)

        # Compute attention scores
        attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.scale

        # Apply mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v_expanded)

        # Reshape and project output (ensure dtype compatibility)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        o_proj = self.o_proj.to(attn_output.dtype)
        output = F.linear(attn_output, o_proj)

        return output, new_cache


# ============================================================================
# GeGLU MLP
# ============================================================================


class GemmaMLP:
    """
    Gemma MLP with GeGLU activation (PyTorch).

    Uses gated GELU: output = down_proj(gelu(gate_proj(x)) * up_proj(x))
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """
        Initialize MLP.

        Args:
            config: Gemma configuration
            weights: Layer weights with keys:
                - mlp.gate_proj.weight
                - mlp.up_proj.weight
                - mlp.down_proj.weight
        """
        self.config = config
        self.gate_proj = weights["mlp.gate_proj.weight"]
        self.up_proj = weights["mlp.up_proj.weight"]
        self.down_proj = weights["mlp.down_proj.weight"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)

        Returns:
            Output tensor of same shape
        """
        # Ensure dtype compatibility
        gate_proj = self.gate_proj.to(x.dtype)
        up_proj = self.up_proj.to(x.dtype)
        down_proj = self.down_proj.to(x.dtype)
        gate = F.linear(x, gate_proj)
        up = F.linear(x, up_proj)
        hidden = F.gelu(gate, approximate="tanh") * up
        return F.linear(hidden, down_proj)


# ============================================================================
# Full Transformer Block
# ============================================================================


class GemmaBlock:
    """
    Complete Gemma transformer block (PyTorch).

    Architecture: Pre-LN with residual connections
        x -> RMSNorm -> Attention -> + -> RMSNorm -> MLP -> +
        |______________________________|___________________|
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ):
        """
        Initialize transformer block.

        Args:
            config: Gemma configuration
            weights: All weights for this layer
            layer_idx: Layer index
        """
        self.config = config
        self.layer_idx = layer_idx

        self.input_layernorm_weight = weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = weights["post_attention_layernorm.weight"]

        self.attention = GemmaAttention(config, weights, layer_idx)
        self.mlp = GemmaMLP(config, weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through transformer block.

        Args:
            hidden_states: Input tensor
            cos, sin: RoPE embeddings
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached KV
            use_cache: Whether to return cache

        Returns:
            Tuple of (output, optional_cache)
        """
        # Pre-attention norm
        normed = rms_norm(
            hidden_states,
            self.input_layernorm_weight,
            self.config.rms_norm_eps,
        )

        # Attention with residual
        attn_output, new_cache = self.attention.forward(
            normed,
            cos,
            sin,
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
        )
        hidden_states = hidden_states + attn_output

        # Pre-MLP norm
        normed = rms_norm(
            hidden_states,
            self.post_attention_layernorm_weight,
            self.config.rms_norm_eps,
        )

        # MLP with residual
        mlp_output = self.mlp.forward(normed)
        hidden_states = hidden_states + mlp_output

        return hidden_states, new_cache
