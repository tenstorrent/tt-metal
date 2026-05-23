# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn

def multihead_attention(query, key, value, parameters, device, num_heads=8):
    # 1. Compute Q, K, V projections
    q = ttnn.linear(query, parameters.q_proj.weight, bias=parameters.q_proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.linear(key, parameters.k_proj.weight, bias=parameters.k_proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.linear(value, parameters.v_proj.weight, bias=parameters.v_proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Robustly extract shapes (handling both 3D and 4D inputs)
    shape_q = q.shape
    shape_k = k.shape
    b = shape_q[0] if len(shape_q) == 4 else shape_q[0]
    seq_q = shape_q[2] if len(shape_q) == 4 else shape_q[1]
    seq_kv = shape_k[2] if len(shape_k) == 4 else shape_k[1]
    hidden = shape_q[-1]
    
    head_dim = hidden // num_heads

    # 2. Reshape to split heads
    q = ttnn.reshape(q, (b, seq_q, num_heads, head_dim))
    k = ttnn.reshape(k, (b, seq_kv, num_heads, head_dim))
    v = ttnn.reshape(v, (b, seq_kv, num_heads, head_dim))

    # 3. Transpose to (B, H, seq, D)
    q = ttnn.transpose(q, 1, 2)
    k = ttnn.transpose(k, 1, 2)
    v = ttnn.transpose(v, 1, 2)

    # 4. Scaled Dot Product Attention
    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, 
        is_causal=False
    )

    # Deallocate Q, K, V immediately after SDPA.
    # This prevents L1 fragmentation and drastically speeds up the next operations.
    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    # 5. Concatenate heads and restore shape
    # (B, H, seq_q, D) -> (B, seq_q, H, D) -> (B, 1, seq_q, hidden)
    attn_out = ttnn.transpose(attn_out, 1, 2)
    attn_out = ttnn.reshape(attn_out, (b, 1, seq_q, hidden))
    
    # 6. Final output projection
    out = ttnn.linear(
        attn_out,
        parameters.out_proj.weight,
        bias=parameters.out_proj.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    
    # Deallocate intermediate attention output
    ttnn.deallocate(attn_out)
    
    return out


def self_attention(x, parameters, device, num_heads=8):
    # query, key, value all sourced from the same tensor
    return multihead_attention(x, x, x, parameters, device, num_heads)


def cross_attention(query, encoder_out, parameters, device, num_heads=8):
    # query from the decoder, key/value from encoder output
    return multihead_attention(query, encoder_out, encoder_out, parameters, device, num_heads)