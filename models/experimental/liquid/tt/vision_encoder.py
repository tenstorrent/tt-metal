# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def siglip2_patch_embeddings(pixel_values, parameters):
    batch, img_h, img_w, img_c = pixel_values.shape
    patch_size = 14
    patch_count = img_h // patch_size
    patch_count_all = patch_count * patch_count
    stride_h, stride_w = patch_size, 1

    pixel_values = ttnn.reshape(pixel_values, (batch, img_h, img_w // patch_size, 4 * patch_size))
    pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT)

    out = ttnn.linear(
        pixel_values,
        parameters.projection.weight,
        bias=parameters.projection.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )
    ttnn.deallocate(pixel_values)
    out = ttnn.to_layout(out, layout=ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.reshape(out, (batch, patch_count_all, -1))
    return out


def siglip2_attention(hidden_states, attention_mask, parameters):
    num_heads = 16
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    qkv = ttnn.linear(
        hidden_states,
        parameters.query_key_value.weight,
        bias=parameters.query_key_value.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )
    ttnn.deallocate(hidden_states)

    qkv = ttnn.to_layout(qkv, layout=ttnn.ROW_MAJOR_LAYOUT)
    query, key, value = ttnn.split(qkv, 3, dim=-1)
    batch, seq_len, _ = query.shape

    query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
    key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
    value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

    query = ttnn.reshape(query, (batch, seq_len, num_heads, head_size))
    key = ttnn.reshape(key, (batch, seq_len, num_heads, head_size))
    value = ttnn.reshape(value, (batch, seq_len, num_heads, head_size))

    query = ttnn.permute(query, (0, 2, 1, 3))
    key = ttnn.permute(key, (0, 2, 1, 3))
    value = ttnn.permute(value, (0, 2, 1, 3))

    attn_weights = ttnn.matmul(query, ttnn.permute(key, (0, 1, 3, 2)))
    attn_weights = attn_weights / (head_size ** 0.5)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = ttnn.softmax(attn_weights, dim=-1)

    attn_output = ttnn.matmul(attn_weights, value)
    attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
    attn_output = ttnn.reshape(attn_output, (batch, seq_len, hidden_size))

    out = ttnn.linear(
        attn_output,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )
    ttnn.deallocate(attn_output)
    return out


def siglip2_ffn(hidden_states, parameters):
    out = ttnn.linear(
        hidden_states,
        parameters.intermediate.dense.weight,
        bias=parameters.intermediate.dense.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )
    ttnn.deallocate(hidden_states)
    out = ttnn.gelu(out)
    out = ttnn.linear(
        out,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )
    return out


def siglip2_encoder_layer(hidden_states, attention_mask, parameters):
    residual = hidden_states
    hidden_states = ttnn.layer_norm(hidden_states, weight=parameters.layer_norm.weight, bias=parameters.layer_norm.bias)
    hidden_states = siglip2_attention(hidden_states, attention_mask, parameters.attention)
    hidden_states = ttnn.add(hidden_states, residual)
    ttnn.deallocate(residual)

    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states, weight=parameters.post_attention_layer_norm.weight, bias=parameters.post_attention_layer_norm.bias
    )
    hidden_states = siglip2_ffn(hidden_states, parameters.ffn)
    hidden_states = ttnn.add(hidden_states, residual)
    ttnn.deallocate(residual)
    return hidden_states


def siglip2_encoder(pixel_values, parameters):
    hidden_states = siglip2_patch_embeddings(pixel_values, parameters.embeddings)
    hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)

    for layer in parameters.encoder.layers:
        hidden_states = siglip2_encoder_layer(hidden_states, None, layer)

    pooled = ttnn.layer_norm(
        hidden_states, weight=parameters.post_layer_norm.weight, bias=parameters.post_layer_norm.bias
    )
    return hidden_states, pooled
