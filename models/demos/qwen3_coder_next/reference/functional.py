# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Functional reference implementations for Qwen3-Coder-Next submodules.

These functions extract and run individual components of the HF model
for PCC comparison against the TT implementation.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def reference_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference RMSNorm implementation.

    Args:
        x: Input tensor (..., hidden_size).
        weight: Learnable scale parameter (hidden_size,).
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor of same shape as x.
    """
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(x.dtype)


def reference_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    partial_rotary_factor: float = 0.25,
) -> torch.Tensor:
    """Reference partial RoPE: only rotate a fraction of head dimensions.

    Args:
        x: Input tensor (batch, seq_len, num_heads, head_dim).
        cos: Cosine frequencies (seq_len, rotary_dim).
        sin: Sine frequencies (seq_len, rotary_dim).
        partial_rotary_factor: Fraction of head_dim to rotate (0.25 = 64 of 256).

    Returns:
        Tensor with partial rotary embeddings applied.
    """
    head_dim = x.shape[-1]
    rotary_dim = int(head_dim * partial_rotary_factor)

    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    # Standard RoPE on the rotary portion
    x_rot_half1 = x_rot[..., : rotary_dim // 2]
    x_rot_half2 = x_rot[..., rotary_dim // 2 :]

    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, rotary_dim/2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    x_rot_out = torch.cat(
        [
            x_rot_half1 * cos - x_rot_half2 * sin,
            x_rot_half2 * cos + x_rot_half1 * sin,
        ],
        dim=-1,
    )

    return torch.cat([x_rot_out, x_pass], dim=-1)


def reference_gqa_attention(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    num_heads: int = 16,
    num_kv_heads: int = 2,
    head_dim: int = 256,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    partial_rotary_factor: float = 0.25,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference GQA attention (for every 4th layer).

    Args:
        hidden_states: (batch, seq_len, hidden_size).
        q/k/v/o_proj_weight: Projection weights.
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads (GQA).
        head_dim: Dimension per head.
        cos, sin: RoPE frequencies.
        partial_rotary_factor: Fraction of dims to rotate.
        attention_mask: Optional causal mask.

    Returns:
        Output tensor (batch, seq_len, hidden_size).
    """
    batch, seq_len, hidden_size = hidden_states.shape

    q = hidden_states @ q_proj_weight.T
    k = hidden_states @ k_proj_weight.T
    v = hidden_states @ v_proj_weight.T

    q = q.view(batch, seq_len, num_heads, head_dim)
    k = k.view(batch, seq_len, num_kv_heads, head_dim)
    v = v.view(batch, seq_len, num_kv_heads, head_dim)

    # Apply partial RoPE
    if cos is not None and sin is not None:
        q = reference_partial_rope(q, cos, sin, partial_rotary_factor)
        k = reference_partial_rope(k, cos, sin, partial_rotary_factor)

    # GQA: repeat KV heads to match Q heads
    num_groups = num_heads // num_kv_heads
    k = k.repeat_interleave(num_groups, dim=2)
    v = v.repeat_interleave(num_groups, dim=2)

    # Transpose to (batch, heads, seq, dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Scaled dot-product attention
    scale = head_dim**-0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)

    # Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    return attn_output @ o_proj_weight.T


def reference_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: SiLU(gate) * up."""
    return F.silu(gate) * up


def reference_expert_mlp(
    x: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """Reference single expert MLP (SwiGLU).

    Args:
        x: Input (batch*seq, hidden_size).
        gate/up/down_proj_weight: Expert weights.

    Returns:
        Output (batch*seq, hidden_size).
    """
    gate = x @ gate_proj_weight.T
    up = x @ up_proj_weight.T
    return reference_silu_mul(gate, up) @ down_proj_weight.T


def reference_moe_gate(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    num_experts_per_tok: int = 10,
    norm_topk_prob: bool = True,
) -> tuple:
    """Reference MoE routing gate (top-k).

    Args:
        hidden_states: (batch*seq, hidden_size).
        router_weight: (num_experts, hidden_size).
        num_experts_per_tok: Number of experts to route to.
        norm_topk_prob: Whether to normalize top-k probabilities.

    Returns:
        Tuple of (routing_weights, selected_experts) where:
            routing_weights: (batch*seq, num_experts_per_tok) normalized probs
            selected_experts: (batch*seq, num_experts_per_tok) expert indices
    """
    router_logits = hidden_states @ router_weight.T
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

    topk_weights, topk_indices = torch.topk(routing_weights, num_experts_per_tok, dim=-1)

    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(hidden_states.dtype), topk_indices
