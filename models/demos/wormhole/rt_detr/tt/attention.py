# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn


def multihead_attention(query, key, value, parameters, device, num_heads=8):
    q = ttnn.linear(query, parameters.q_proj.weight, bias=parameters.q_proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.linear(key, parameters.k_proj.weight, bias=parameters.k_proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.linear(value, parameters.v_proj.weight, bias=parameters.v_proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Handle 4D shapes
    if len(q.shape) == 4:
        b = q.shape[0]
        seq = q.shape[-2]
        hidden = q.shape[-1]
    else:
        b, seq, hidden = q.shape

    head_dim = hidden // num_heads

    q = ttnn.reshape(q, (b, seq, num_heads, head_dim))
    k = ttnn.reshape(k, (b, seq, num_heads, head_dim))
    v = ttnn.reshape(v, (b, seq, num_heads, head_dim))

    q = ttnn.transpose(q, 1, 2)
    k = ttnn.transpose(k, 1, 2)
    v = ttnn.transpose(v, 1, 2)

    attn_out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)

    attn_out = ttnn.transpose(attn_out, 1, 2)

    # Restore to 4D (Batch, 1, Seq, Hidden) to match typical TILE layout flow
    attn_out = ttnn.reshape(attn_out, (b, 1, seq, hidden))

    return ttnn.linear(
        attn_out, parameters.out_proj.weight, bias=parameters.out_proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG
    )


def self_attention(x, parameters, device, num_heads=8):
    # query, key, value all come from the same input
    return multihead_attention(x, x, x, parameters, device, num_heads)


def cross_attention(query, encoder_out, parameters, device, num_heads=8):
    # query from decoder, key/value from encoder
    return multihead_attention(query, encoder_out, encoder_out, parameters, device, num_heads)
