# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def _shape(tensor, seq_len, bsz, num_heads, head_dim, device):
    output_tensor = ttnn.from_device(tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.reshape(output_tensor, (bsz, seq_len, num_heads, head_dim))
    output_tensor = ttnn.to_device(output_tensor, device)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.permute(output_tensor, (0, 2, 1, 3))
    return output_tensor


def trocr_attention(
    hidden_states,
    key_value_states=None,
    past_key_value=None,
    attention_mask=None,
    layer_head_mask=None,
    output_attentions=None,
    config=None,
    embed_dim=None,
    num_heads=None,
    kdim=None,
    vdim=None,
    is_decoder=False,
    bias=True,
    is_cross_attention=False,
    *,
    parameters,
    device,
):
    head_dim = embed_dim // num_heads
    scaling = head_dim**-0.5

    is_cross_attention = key_value_states is not None
    bsz, tgt_len, embed_dim = hidden_states.shape

    query_states = ttnn.linear(hidden_states, parameters.q_proj.weight, bias=parameters.q_proj.bias)
    query_states = query_states * scaling

    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]

    elif is_cross_attention:
        key_value_states = ttnn.to_device(key_value_states, device)
        key_states = ttnn.linear(key_value_states, parameters.k_proj.weight, bias=parameters.k_proj.bias)
        key_states = _shape(key_states, -1, bsz, num_heads, head_dim, device)

        value_states = ttnn.linear(key_value_states, parameters.v_proj.weight, bias=parameters.v_proj.bias)
        value_states = _shape(value_states, -1, bsz, num_heads, head_dim, device)

    elif past_key_value is not None:
        key_states = ttnn.linear(hidden_states, parameters.k_proj.weight, bias=parameters.k_proj.bias)
        key_states = _shape(key_states, -1, bsz, num_heads, head_dim, device)

        value_states = ttnn.linear(hidden_states, parameters.v_proj.weight, bias=parameters.v_proj.bias)
        value_states = _shape(value_states, -1, bsz, num_heads, head_dim, device)

        key_states = ttnn.concat(past_key_value[0], key_states, dim=2)
        value_states = ttnn.concat(past_key_value[1], value_states, dim=2)

    else:
        key_states = ttnn.linear(hidden_states, parameters.k_proj.weight, bias=parameters.k_proj.bias)
        key_states = _shape(key_states, -1, bsz, num_heads, head_dim, device)

        value_states = ttnn.linear(hidden_states, parameters.v_proj.weight, bias=parameters.v_proj.bias)
        value_states = _shape(value_states, -1, bsz, num_heads, head_dim, device)

    if is_decoder:
        past_key_value = (key_states, value_states)

    query_states = _shape(query_states, tgt_len, bsz, num_heads, head_dim, device)
    query_states = ttnn.to_layout(query_states, layout=ttnn.ROW_MAJOR_LAYOUT)
    query_states = ttnn.reshape(query_states, (bsz * num_heads, -1, head_dim))
    query_states = ttnn.to_layout(query_states, ttnn.TILE_LAYOUT)

    key_states = ttnn.to_layout(key_states, layout=ttnn.ROW_MAJOR_LAYOUT)
    key_states = ttnn.reshape(key_states, (bsz * num_heads, -1, head_dim))
    key_states = ttnn.to_layout(key_states, ttnn.TILE_LAYOUT)

    value_states = ttnn.to_layout(value_states, layout=ttnn.ROW_MAJOR_LAYOUT)
    value_states = ttnn.reshape(value_states, (bsz * num_heads, -1, head_dim))
    value_states = ttnn.to_layout(value_states, ttnn.TILE_LAYOUT)

    src_len = key_states.shape[1]

    key_states = ttnn.permute(key_states, (0, 2, 1))
    attn_weights = ttnn.matmul(query_states, key_states)

    if attention_mask is not None:
        attention_mask = ttnn.from_torch(attention_mask, device=None, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        attn_weights = ttnn.from_device(attn_weights)
        attn_weights = ttnn.to_layout(attn_weights, ttnn.ROW_MAJOR_LAYOUT)
        attn_weights = ttnn.reshape(attn_weights, (bsz, num_heads, tgt_len, src_len))

        attn_weights = torch.add(ttnn.to_torch(attn_weights), ttnn.to_torch(attention_mask))  # broadcast issue

        attn_weights = ttnn.from_torch(attn_weights, device=None, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        attn_weights = ttnn.reshape(attn_weights, (bsz * num_heads, tgt_len, src_len))

    attn_weights = ttnn.to_layout(attn_weights, ttnn.TILE_LAYOUT)
    attn_weights = ttnn.to_device(attn_weights, device)
    attn_weights = ttnn.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        attn_weights = ttnn.reshape(attn_weights, (bsz, num_heads, tgt_len, src_len))
        layer_head_mask = ttnn.reshape(layer_head_mask, (1, -1, 1, 1))
        attn_weights = layer_head_mask * attn_weights

    if output_attentions:
        attn_weights_reshaped = ttnn.reshape(attn_weights, (bsz, num_heads, tgt_len, src_len))
        attn_weights = ttnn.reshape(attn_weights_reshaped, (bsz * num_heads, tgt_len, src_len))
    else:
        attn_weights_reshaped = None

    attn_output = ttnn.matmul(attn_weights, value_states)

    attn_output = ttnn.to_layout(attn_output, ttnn.ROW_MAJOR_LAYOUT)
    attn_output = ttnn.reshape(attn_output, (bsz, num_heads, tgt_len, head_dim))
    attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
    attn_output = ttnn.from_device(attn_output)
    attn_output = ttnn.reshape(attn_output, (bsz, tgt_len, embed_dim))
    attn_output = ttnn.to_device(attn_output, device)
    attn_output = ttnn.to_layout(attn_output, ttnn.TILE_LAYOUT)

    attn_output = ttnn.linear(attn_output, parameters.out_proj.weight, bias=parameters.out_proj.bias)

    return attn_output, attn_weights_reshaped, past_key_value
