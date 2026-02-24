"""
TTNN implementation of Gated Attention.
"""

import ttnn


def rotate_half_ttnn(x):
    """Rotates half the hidden dims of the input."""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def apply_rotary_pos_emb_ttnn(q, k, cos, sin):
    """Apply RoPE to query and key tensors using TTNN ops."""
    cos = ttnn.unsqueeze(cos, 1)  # [B, 1, T, D]
    sin = ttnn.unsqueeze(sin, 1)

    rotary_dim = cos.shape[-1]
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_embed = ttnn.add(
        ttnn.multiply(q_rot, cos),
        ttnn.multiply(rotate_half_ttnn(q_rot), sin),
    )
    k_embed = ttnn.add(
        ttnn.multiply(k_rot, cos),
        ttnn.multiply(rotate_half_ttnn(k_rot), sin),
    )

    q_embed = ttnn.concat([q_embed, q_pass], dim=-1)
    k_embed = ttnn.concat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def rms_norm_zero_centered_ttnn(x, weight, eps=1e-6):
    """
    Zero-centered RMSNorm using TTNN: x * rsqrt(mean(x^2) + eps) * (1 + weight).
    """
    x_sq = ttnn.multiply(x, x)
    variance = ttnn.mean(x_sq, dim=-1, keepdim=True)
    inv_rms = ttnn.rsqrt(ttnn.add(variance, eps))
    x_normed = ttnn.multiply(x, inv_rms)
    scale = ttnn.add(weight, 1.0)
    return ttnn.multiply(x_normed, scale)


def repeat_kv_ttnn(hidden_states, n_rep):
    """Expand KV heads to match Q heads for GQA."""
    if n_rep == 1:
        return hidden_states
    return ttnn.repeat_interleave(hidden_states, n_rep, dim=1)


def gated_attention_forward_ttnn(
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
    device,
    attention_mask=None,
    norm_eps=1e-6,
):
    """
    TTNN forward pass for Gated Attention.

    All operations use TTNN equivalents for execution on Tenstorrent hardware.

    Args:
        hidden_states: ttnn.Tensor [B, T, hidden_size]
        *_proj_weight: ttnn.Tensor weight matrices in [in_features, out_features] format
                       (transposed from PyTorch convention)
        q_norm_weight, k_norm_weight: ttnn.Tensor [head_dim]
        cos, sin: ttnn.Tensor [B, T, head_dim] rotary embeddings
        num_attention_heads: number of Q heads
        num_key_value_heads: number of KV heads
        head_dim: dimension per head
        device: ttnn device
        attention_mask: ttnn.Tensor [B, 1, T, S] or None
        norm_eps: RMSNorm epsilon

    Returns:
        output: ttnn.Tensor [B, T, hidden_size]
    """
    B = hidden_states.shape[0]
    T = hidden_states.shape[1]
    num_kv_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim**-0.5

    # Q projection: 2x wide
    qg = ttnn.linear(hidden_states, q_proj_weight)
    qg = ttnn.reshape(qg, [B, T, num_attention_heads, head_dim * 2])
    # Split into query and gate
    query_states, gate = ttnn.chunk(qg, 2, dim=-1)
    gate = ttnn.reshape(gate, [B, T, num_attention_heads * head_dim])

    # Q norm + transpose to [B, H, T, D]
    query_states = rms_norm_zero_centered_ttnn(query_states, q_norm_weight, eps=norm_eps)
    query_states = ttnn.transpose(query_states, 1, 2)  # [B, H, T, D]

    # K projection + norm + transpose
    key_states = ttnn.linear(hidden_states, k_proj_weight)
    key_states = ttnn.reshape(key_states, [B, T, num_key_value_heads, head_dim])
    key_states = rms_norm_zero_centered_ttnn(key_states, k_norm_weight, eps=norm_eps)
    key_states = ttnn.transpose(key_states, 1, 2)

    # V projection + transpose
    value_states = ttnn.linear(hidden_states, v_proj_weight)
    value_states = ttnn.reshape(value_states, [B, T, num_key_value_heads, head_dim])
    value_states = ttnn.transpose(value_states, 1, 2)

    # RoPE
    query_states, key_states = apply_rotary_pos_emb_ttnn(query_states, key_states, cos, sin)

    # Expand KV for GQA
    key_states = repeat_kv_ttnn(key_states, num_kv_groups)
    value_states = repeat_kv_ttnn(value_states, num_kv_groups)

    # Attention: Q @ K^T * scale
    key_t = ttnn.transpose(key_states, 2, 3)
    attn_weights = ttnn.matmul(query_states, key_t)
    attn_weights = ttnn.multiply(attn_weights, scaling)

    if attention_mask is not None:
        attn_weights = ttnn.add(attn_weights, attention_mask)

    # Softmax
    attn_weights = ttnn.softmax(attn_weights, dim=-1)

    # Attention output: weights @ V
    attn_output = ttnn.matmul(attn_weights, value_states)  # [B, H, T, D]

    # Transpose back and reshape: [B, T, H*D]
    attn_output = ttnn.transpose(attn_output, 1, 2)
    attn_output = ttnn.reshape(attn_output, [B, T, num_attention_heads * head_dim])

    # Apply sigmoid gate
    gate = ttnn.sigmoid(gate)
    attn_output = ttnn.multiply(attn_output, gate)

    # Output projection
    attn_output = ttnn.linear(attn_output, o_proj_weight)

    return attn_output
