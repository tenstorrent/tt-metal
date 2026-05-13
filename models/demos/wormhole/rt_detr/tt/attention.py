# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn


def multihead_attention(query, key, value, parameters, device, num_heads=8):
    q = ttnn.linear(query, parameters.q_proj.weight, bias=parameters.q_proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.linear(key, parameters.k_proj.weight, bias=parameters.k_proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.linear(value, parameters.v_proj.weight, bias=parameters.v_proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG)

    # ttnn linears come back as (B, 1, seq, hidden) in TILE layout.
    # Q and K/V can have different sequence lengths in cross-attention.
    if len(q.shape) == 4:
        b, _, seq_q, hidden = q.shape
    else:
        b, seq_q, hidden = q.shape

    if len(k.shape) == 4:
        _, _, seq_kv, _ = k.shape
    else:
        _, seq_kv, _ = k.shape

    head_dim = hidden // num_heads

    q = ttnn.reshape(q, (b, seq_q,  num_heads, head_dim))
    k = ttnn.reshape(k, (b, seq_kv, num_heads, head_dim))
    v = ttnn.reshape(v, (b, seq_kv, num_heads, head_dim))

    # (B, seq, H, D) -> (B, H, seq, D)
    q = ttnn.transpose(q, 1, 2)
    k = ttnn.transpose(k, 1, 2)
    v = ttnn.transpose(v, 1, 2)

    attn_out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)

    # (B, H, seq_q, D) -> (B, seq_q, H, D) -> (B, 1, seq_q, hidden)
    attn_out = ttnn.transpose(attn_out, 1, 2)
    attn_out = ttnn.reshape(attn_out, (b, 1, seq_q, hidden))

    return ttnn.linear(
        attn_out,
        parameters.out_proj.weight,
        bias=parameters.out_proj.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def self_attention(x, parameters, device, num_heads=8):
    # query, key, value all sourced from the same tensor
    return multihead_attention(x, x, x, parameters, device, num_heads)


def cross_attention(query, encoder_out, parameters, device, num_heads=8):
    # query from the decoder, key/value from encoder output
    return multihead_attention(query, encoder_out, encoder_out, parameters, device, num_heads)