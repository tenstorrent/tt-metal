# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

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
    full_dim = q.shape[-1]

    q_rot = q[..., :rotary_dim]
    k_rot = k[..., :rotary_dim]

    q_embed = ttnn.add(
        ttnn.multiply(q_rot, cos),
        ttnn.multiply(rotate_half_ttnn(q_rot), sin),
    )
    k_embed = ttnn.add(
        ttnn.multiply(k_rot, cos),
        ttnn.multiply(rotate_half_ttnn(k_rot), sin),
    )

    if rotary_dim < full_dim:
        q_pass = q[..., rotary_dim:]
        k_pass = k[..., rotary_dim:]
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


def _get_sdpa_program_config(device, seq_len):
    """Build SDPAProgramConfig with chunk sizes tuned to sequence length.

    Follows the same pattern as tt_transformers model_config.py:
      - T >= 2048 -> q/k chunk = 256
      - T < 2048  -> q/k chunk = 64
    """
    grid_size = device.compute_with_storage_grid_size()
    q_chunk = 256 if seq_len >= 2048 else 64
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=q_chunk,
        k_chunk_size=q_chunk,
        exp_approx_mode=False,
    )


def _get_sdpa_compute_kernel_config():
    """WormholeComputeKernelConfig for SDPA -- HiFi4 with fp32 accumulation."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


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
    norm_eps=1e-6,
    use_optimized_concat=False,
):
    """
    TTNN forward pass for Gated Attention.

    Uses ttnn.transformer.scaled_dot_product_attention (FlashAttention-2 kernel)
    with SDPAProgramConfig for tiling and WormholeComputeKernelConfig for precision.
    The fused kernel handles causal masking and GQA (num_q_heads != num_kv_heads)
    internally, avoiding explicit repeat_kv and O(T^2) attention-matrix allocation.

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
        norm_eps: RMSNorm epsilon
        use_optimized_concat: if True, use ttnn.transformer.concatenate_heads
                              instead of ttnn.transpose + ttnn.reshape

    Returns:
        output: ttnn.Tensor [B, T, hidden_size]
    """
    B = hidden_states.shape[0]
    T = hidden_states.shape[1]
    scaling = head_dim**-0.5

    # Q projection: 2x wide
    qg = ttnn.linear(hidden_states, q_proj_weight)
    qg = ttnn.reshape(qg, [B, T, num_attention_heads, head_dim * 2])
    # Split into query and gate
    query_states, gate = ttnn.chunk(qg, 2, dim=-1)
    gate = ttnn.reshape(gate, [B, T, num_attention_heads * head_dim])

    # Q norm + transpose to [B, H_q, T, D]
    query_states = rms_norm_zero_centered_ttnn(query_states, q_norm_weight, eps=norm_eps)
    query_states = ttnn.transpose(query_states, 1, 2)

    # K projection + norm + transpose to [B, H_kv, T, D]
    key_states = ttnn.linear(hidden_states, k_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    key_states = ttnn.reshape(key_states, [B, T, num_key_value_heads, head_dim])
    key_states = rms_norm_zero_centered_ttnn(key_states, k_norm_weight, eps=norm_eps)
    key_states = ttnn.transpose(key_states, 1, 2)

    # V projection + transpose to [B, H_kv, T, D]
    value_states = ttnn.linear(hidden_states, v_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    value_states = ttnn.reshape(value_states, [B, T, num_key_value_heads, head_dim])
    value_states = ttnn.transpose(value_states, 1, 2)

    # RoPE
    query_states, key_states = apply_rotary_pos_emb_ttnn(query_states, key_states, cos, sin)

    # Fused scaled dot-product attention
    # SDPAProgramConfig sets chunk tiling; WormholeComputeKernelConfig sets fp32 accum.
    # The kernel handles GQA (num_q_heads != num_kv_heads) and causal masking internally.
    attn_output = ttnn.transformer.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        is_causal=True,
        scale=scaling,
        program_config=_get_sdpa_program_config(device, T),
        compute_kernel_config=_get_sdpa_compute_kernel_config(),
    )

    # Convert from [B, H, T, D] back to [B, T, H*D]
    if use_optimized_concat:
        attn_output = ttnn.transformer.concatenate_heads(attn_output)
    else:
        attn_output = ttnn.transpose(attn_output, 1, 2)
        attn_output = ttnn.reshape(attn_output, [B, T, num_attention_heads * head_dim])

    # Apply sigmoid gate
    gate = ttnn.sigmoid(gate)
    attn_output = ttnn.multiply(attn_output, gate)

    # Output projection
    attn_output = ttnn.linear(attn_output, o_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)

    return attn_output
