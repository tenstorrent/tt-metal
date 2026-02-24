"""
Functional torch implementation of Gated Attention.

Extracted from HuggingFace Transformers Qwen3-Next:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py

Gated Attention is standard Scaled Dot-Product Attention with a query-dependent
sigmoid gate applied to the output. The Q projection is 2x wide -- the second
half is used as the gate signal.

Architecture:
  Input -> q_proj (2x width) -> split into [Q, Gate]
        -> k_proj, v_proj
        -> Q/K norm -> RoPE
        -> SDPA(Q, K, V)
        -> output * sigmoid(Gate)
        -> o_proj -> Output
"""

import torch
import torch.nn.functional as F


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to query and key tensors.

    Args:
        q: [B, H, T, D] query
        k: [B, H, T, D] key
        cos: [B, T, D] or [1, T, D] cosine embeddings
        sin: [B, T, D] or [1, T, D] sine embeddings
        unsqueeze_dim: dimension to unsqueeze cos/sin for broadcasting

    Returns:
        q_embed, k_embed: rotated query and key tensors
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def rms_norm_zero_centered(x, weight, eps=1e-6):
    """
    Zero-centered RMSNorm: output = x * rsqrt(mean(x^2) + eps) * (1 + weight).
    Used by Qwen3-Next for Q/K normalization.
    """
    input_dtype = x.dtype
    x_f32 = x.float()
    output = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)
    output = output * (1.0 + weight.float())
    return output.to(input_dtype)


def repeat_kv(hidden_states, n_rep):
    """
    Expand KV heads to match Q heads for GQA.
    [B, H_kv, T, D] -> [B, H_q, T, D]
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def gated_attention_forward(
    hidden_states,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    q_norm_weight,
    k_norm_weight,
    cos,
    sin,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    attention_mask=None,
    norm_eps=1e-6,
    attention_dropout=0.0,
    training=False,
    past_key_value_k=None,
    past_key_value_v=None,
    output_kv_cache=False,
):
    """
    Functional forward pass for Gated Attention.

    The Q projection is 2x wide: first half is the actual query, second half
    is the gate (applied as sigmoid after attention).

    Args:
        hidden_states: [B, T, hidden_size]
        q_proj_weight: [num_heads * head_dim * 2, hidden_size]
        k_proj_weight: [num_kv_heads * head_dim, hidden_size]
        v_proj_weight: [num_kv_heads * head_dim, hidden_size]
        o_proj_weight: [hidden_size, num_heads * head_dim]
        q_norm_weight: [head_dim] for Q RMSNorm
        k_norm_weight: [head_dim] for K RMSNorm
        cos: [B, T, head_dim] or [1, T, head_dim] rotary cos
        sin: [B, T, head_dim] or [1, T, head_dim] rotary sin
        num_attention_heads: number of query heads
        num_key_value_heads: number of KV heads (for GQA)
        head_dim: dimension per head
        attention_mask: [B, 1, T, S] causal mask (additive, -inf for masked)
        norm_eps: epsilon for Q/K RMSNorm
        attention_dropout: dropout rate
        training: training mode flag
        past_key_value_k: [B, H_kv, S_past, D] cached keys
        past_key_value_v: [B, H_kv, S_past, D] cached values
        output_kv_cache: whether to return updated KV cache

    Returns:
        output: [B, T, hidden_size]
        new_k_cache: [B, H_kv, S, D] or None
        new_v_cache: [B, H_kv, S, D] or None
    """
    B, T, _ = hidden_states.shape
    num_kv_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim**-0.5

    # Q projection: 2x wide, split into query + gate
    qg = F.linear(hidden_states, q_proj_weight)  # [B, T, num_heads * head_dim * 2]
    qg = qg.view(B, T, num_attention_heads, head_dim * 2)
    query_states, gate = torch.chunk(qg, 2, dim=-1)  # each [B, T, H, D]
    gate = gate.reshape(B, T, -1)  # [B, T, num_heads * head_dim]

    # Q/K normalization (zero-centered RMSNorm, applied per-head)
    query_states = rms_norm_zero_centered(query_states, q_norm_weight, eps=norm_eps)
    query_states = query_states.transpose(1, 2)  # [B, H, T, D]

    key_states = F.linear(hidden_states, k_proj_weight)
    key_states = key_states.view(B, T, num_key_value_heads, head_dim)
    key_states = rms_norm_zero_centered(key_states, k_norm_weight, eps=norm_eps)
    key_states = key_states.transpose(1, 2)  # [B, H_kv, T, D]

    value_states = F.linear(hidden_states, v_proj_weight)
    value_states = value_states.view(B, T, num_key_value_heads, head_dim)
    value_states = value_states.transpose(1, 2)  # [B, H_kv, T, D]

    # Apply RoPE
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # KV cache handling
    if past_key_value_k is not None:
        key_states = torch.cat([past_key_value_k, key_states], dim=2)
        value_states = torch.cat([past_key_value_v, value_states], dim=2)

    new_k = key_states if output_kv_cache else None
    new_v = value_states if output_kv_cache else None

    # Expand KV heads for GQA
    key_states = repeat_kv(key_states, num_kv_groups)
    value_states = repeat_kv(value_states, num_kv_groups)

    # Scaled dot-product attention
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if training and attention_dropout > 0:
        attn_weights = F.dropout(attn_weights, p=attention_dropout, training=True)

    attn_output = torch.matmul(attn_weights, value_states)  # [B, H, T, D]
    attn_output = attn_output.transpose(1, 2).reshape(B, T, -1)  # [B, T, H*D]

    # Apply sigmoid gate
    attn_output = attn_output * torch.sigmoid(gate)

    # Output projection
    attn_output = F.linear(attn_output, o_proj_weight)

    return attn_output, new_k, new_v
