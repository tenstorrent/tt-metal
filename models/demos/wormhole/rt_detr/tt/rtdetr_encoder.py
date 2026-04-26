# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

import ttnn


def _layer_norm(x, norm_params, eps=1e-5):
    return ttnn.layer_norm(
        x, epsilon=eps, weight=norm_params.weight, bias=norm_params.bias, memory_config=ttnn.L1_MEMORY_CONFIG
    )


def encoder_layer(x, p, device, num_heads=8):
    residual = x

    # 1. Self Attention
    q = ttnn.linear(x, p.self_attn.q.weight, bias=p.self_attn.q.bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.linear(x, p.self_attn.k.weight, bias=p.self_attn.k.bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.linear(x, p.self_attn.v.weight, bias=p.self_attn.v.bias, memory_config=ttnn.L1_MEMORY_CONFIG)

    if len(q.shape) == 4:
        batch, _, seq_len, hidden = q.shape
    else:
        batch, seq_len, hidden = q.shape

    head_dim = hidden // num_heads

    q = ttnn.reshape(q, (batch, seq_len, num_heads, head_dim))
    k = ttnn.reshape(k, (batch, seq_len, num_heads, head_dim))
    v = ttnn.reshape(v, (batch, seq_len, num_heads, head_dim))

    q = ttnn.transpose(q, 1, 2)
    k = ttnn.transpose(k, 1, 2)
    v = ttnn.transpose(v, 1, 2)

    attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
    attn = ttnn.transpose(attn, 1, 2)
    attn = ttnn.reshape(attn, (batch, 1, seq_len, hidden))
    attn = ttnn.linear(
        attn, p.self_attn.out_proj.weight, bias=p.self_attn.out_proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    x = ttnn.add(residual, attn, memory_config=ttnn.L1_MEMORY_CONFIG)
    x = _layer_norm(x, p.norm1)

    # 2. Feed Forward Network
    residual = x

    ffn = ttnn.linear(x, p.linear1.weight, bias=p.linear1.bias, activation="gelu", memory_config=ttnn.L1_MEMORY_CONFIG)
    ffn = ttnn.linear(ffn, p.linear2.weight, bias=p.linear2.bias, memory_config=ttnn.L1_MEMORY_CONFIG)

    x = ttnn.add(residual, ffn, memory_config=ttnn.L1_MEMORY_CONFIG)
    x = _layer_norm(x, p.norm2)

    return x


def run_encoder(x, layer_params, device, pos_embed=None):
    if isinstance(x, torch.Tensor):
        x = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

    if pos_embed is not None:
        if isinstance(pos_embed, torch.Tensor):
            pos_embed = ttnn.from_torch(
                pos_embed,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        x = ttnn.add(x, pos_embed, memory_config=ttnn.L1_MEMORY_CONFIG)

    for p in layer_params:
        x = encoder_layer(x, p, device)

    return ttnn.to_torch(x)
