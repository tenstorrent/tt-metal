# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RT-DETR encoder: AIFI (Attention-based Intra-scale Feature Interaction).


import ttnn


def _layer_norm(x, norm_params, eps=1e-5):
    return ttnn.layer_norm(
        x,
        epsilon=eps,
        weight=norm_params.weight,
        bias=norm_params.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _split_heads(x, num_heads, batch, seq_len, hidden):
    head_dim = hidden // num_heads
    x = ttnn.reshape(x, (batch, seq_len, num_heads, head_dim))
    return ttnn.transpose(x, 1, 2)  # (B, H, seq, D)


def encoder_layer(x, p, device, num_heads=8, pos_embed=None):
    """Single AIFI transformer encoder layer.

    pos_embed, when provided, is added to Q and K only. V is always the raw x.
    """
    residual = x

    # self-attention projections 
    # Q and K get positional embedding; V does not.
    x_with_pos = x if pos_embed is None else ttnn.add(x, pos_embed, memory_config=ttnn.L1_MEMORY_CONFIG)

    q = ttnn.linear(x_with_pos, p.self_attn.q.weight, bias=p.self_attn.q.bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.linear(x_with_pos, p.self_attn.k.weight, bias=p.self_attn.k.bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.linear(x, p.self_attn.v.weight, bias=p.self_attn.v.bias, memory_config=ttnn.L1_MEMORY_CONFIG)

    if len(q.shape) == 4:
        batch, _, seq_len, hidden = q.shape
    else:
        batch, seq_len, hidden = q.shape

    q = _split_heads(q, num_heads, batch, seq_len, hidden)
    k = _split_heads(k, num_heads, batch, seq_len, hidden)
    v = _split_heads(v, num_heads, batch, seq_len, hidden)

    attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
    attn = ttnn.transpose(attn, 1, 2)
    attn = ttnn.reshape(attn, (batch, 1, seq_len, hidden))
    attn = ttnn.linear(
        attn, p.self_attn.out_proj.weight, bias=p.self_attn.out_proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    x = ttnn.add(residual, attn, memory_config=ttnn.L1_MEMORY_CONFIG)
    x = _layer_norm(x, p.norm1)

    # feed-forward network 
    residual = x
    ffn = ttnn.linear(x, p.linear1.weight, bias=p.linear1.bias, activation="gelu", memory_config=ttnn.L1_MEMORY_CONFIG)
    ffn = ttnn.linear(ffn, p.linear2.weight, bias=p.linear2.bias, memory_config=ttnn.L1_MEMORY_CONFIG)

    x = ttnn.add(residual, ffn, memory_config=ttnn.L1_MEMORY_CONFIG)
    x = _layer_norm(x, p.norm2)

    return x


def run_aifi(x, layer_params, device, pos_embed=None):
    for p in layer_params:
        x = encoder_layer(x, p, device, pos_embed=pos_embed)
    return x