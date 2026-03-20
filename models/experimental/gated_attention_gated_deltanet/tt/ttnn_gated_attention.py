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


def rms_norm_zero_centered_ttnn(x, weight, eps=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG):
    """
    Zero-centered RMSNorm using TTNN: x * rsqrt(mean(x^2) + eps) * (1 + weight).
    """
    x_sq = ttnn.multiply(x, x, memory_config=memory_config)
    variance = ttnn.mean(x_sq, dim=-1, keepdim=True, memory_config=memory_config)
    inv_rms = ttnn.rsqrt(ttnn.add(variance, eps, memory_config=memory_config), memory_config=memory_config)
    x_normed = ttnn.multiply(x, inv_rms, memory_config=memory_config)
    scale = ttnn.add(weight, 1.0, memory_config=memory_config)
    return ttnn.multiply(x_normed, scale, memory_config=memory_config)


def _get_sdpa_program_config(device, seq_len, exp_approx_mode=False):
    """Build SDPAProgramConfig with chunk sizes tuned to sequence length.

    Chunk sizes: T >= 2048 -> 256; 256 <= T < 2048 -> 128; else 64.
    """
    grid_size = device.compute_with_storage_grid_size()
    if seq_len >= 2048:
        q_chunk = 256
    elif seq_len >= 256:
        q_chunk = 128
    else:
        q_chunk = 64
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=q_chunk,
        k_chunk_size=q_chunk,
        exp_approx_mode=exp_approx_mode,
    )


def _get_sdpa_compute_kernel_config(fast_sdpa=False):
    """Return Wormhole compute config for SDPA (default accuracy vs optional fast path)."""
    if fast_sdpa:
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
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
    use_optimized_concat=True,
    fast_sdpa=False,
    attn_linear_dram_threshold=128,
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
        use_optimized_concat: if True (default), use ttnn.transformer.concatenate_heads when
            head_dim is tile-aligned (multiple of 32); otherwise transpose+reshape.
        fast_sdpa: if True, use LoFi + softmax exp approx + L1 packer (faster; re-check PCC).
        attn_linear_dram_threshold: use DRAM for Q/K/V linears and RMS norms when T exceeds this (0 = always L1).

    Returns:
        output: ttnn.Tensor [B, T, hidden_size]
    """
    B = hidden_states.shape[0]
    T = hidden_states.shape[1]
    scaling = head_dim**-0.5

    attn_mem = (
        ttnn.DRAM_MEMORY_CONFIG
        if attn_linear_dram_threshold > 0 and T > attn_linear_dram_threshold
        else ttnn.L1_MEMORY_CONFIG
    )

    # Q projection: 2x wide
    qg = ttnn.linear(hidden_states, q_proj_weight, memory_config=attn_mem)
    qg = ttnn.reshape(qg, [B, T, num_attention_heads, head_dim * 2], memory_config=attn_mem)
    # Split into query and gate
    query_states, gate = ttnn.chunk(qg, 2, dim=-1)
    gate = ttnn.reshape(gate, [B, T, num_attention_heads * head_dim], memory_config=attn_mem)

    # Q norm + transpose to [B, H_q, T, D]
    query_states = rms_norm_zero_centered_ttnn(query_states, q_norm_weight, eps=norm_eps, memory_config=attn_mem)
    query_states = ttnn.transpose(query_states, 1, 2, memory_config=attn_mem)

    # K projection + norm + transpose to [B, H_kv, T, D]
    key_states = ttnn.linear(hidden_states, k_proj_weight, memory_config=attn_mem)
    key_states = ttnn.reshape(key_states, [B, T, num_key_value_heads, head_dim], memory_config=attn_mem)
    key_states = rms_norm_zero_centered_ttnn(key_states, k_norm_weight, eps=norm_eps, memory_config=attn_mem)
    key_states = ttnn.transpose(key_states, 1, 2, memory_config=attn_mem)

    # V projection + transpose to [B, H_kv, T, D]
    value_states = ttnn.linear(hidden_states, v_proj_weight, memory_config=attn_mem)
    value_states = ttnn.reshape(value_states, [B, T, num_key_value_heads, head_dim], memory_config=attn_mem)
    value_states = ttnn.transpose(value_states, 1, 2, memory_config=attn_mem)

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
        program_config=_get_sdpa_program_config(device, T, exp_approx_mode=fast_sdpa),
        compute_kernel_config=_get_sdpa_compute_kernel_config(fast_sdpa=fast_sdpa),
    )

    ttnn.deallocate(query_states)
    ttnn.deallocate(key_states)
    ttnn.deallocate(value_states)

    # [B, H, T, D] -> [B, T, H*D]; concatenate_heads avoids an extra transpose+reshape when valid.
    if use_optimized_concat and head_dim % 32 == 0:
        attn_output = ttnn.transformer.concatenate_heads(attn_output)
    else:
        attn_output = ttnn.transpose(attn_output, 1, 2, memory_config=attn_mem)
        attn_output = ttnn.reshape(attn_output, [B, T, num_attention_heads * head_dim], memory_config=attn_mem)

    # Apply sigmoid gate
    gate = ttnn.sigmoid(gate, memory_config=attn_mem)
    attn_output = ttnn.multiply(attn_output, gate, memory_config=attn_mem)

    # Output projection
    attn_output = ttnn.linear(attn_output, o_proj_weight, memory_config=attn_mem)

    return attn_output
