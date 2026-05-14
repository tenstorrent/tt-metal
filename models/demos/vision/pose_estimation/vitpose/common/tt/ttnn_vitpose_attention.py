# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def vitpose_attention(hidden_states, *, parameters, num_heads=12, compute_kernel_config=None):
    batch_size, seq_len, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_heads

    mm_kwargs = {}
    if compute_kernel_config is not None:
        mm_kwargs["compute_kernel_config"] = compute_kernel_config

    query = ttnn.matmul(hidden_states, parameters["attention.query.weight"], **mm_kwargs)
    query = query + parameters["attention.query.bias"]
    query = ttnn.to_layout(query, layout=ttnn.ROW_MAJOR_LAYOUT)
    query = ttnn.reshape(query, (batch_size, seq_len, num_heads, head_dim))
    query = ttnn.to_layout(query, layout=ttnn.TILE_LAYOUT)
    query = ttnn.permute(query, (0, 2, 1, 3))

    key = ttnn.matmul(hidden_states, parameters["attention.key.weight"], **mm_kwargs)
    key = key + parameters["attention.key.bias"]
    key = ttnn.to_layout(key, layout=ttnn.ROW_MAJOR_LAYOUT)
    key = ttnn.reshape(key, (batch_size, seq_len, num_heads, head_dim))
    key = ttnn.to_layout(key, layout=ttnn.TILE_LAYOUT)
    key = ttnn.permute(key, (0, 2, 3, 1))

    value = ttnn.matmul(hidden_states, parameters["attention.value.weight"], **mm_kwargs)
    value = value + parameters["attention.value.bias"]
    value = ttnn.to_layout(value, layout=ttnn.ROW_MAJOR_LAYOUT)
    value = ttnn.reshape(value, (batch_size, seq_len, num_heads, head_dim))
    value = ttnn.to_layout(value, layout=ttnn.TILE_LAYOUT)
    value = ttnn.permute(value, (0, 2, 1, 3))

    attention_scores = ttnn.matmul(query, key, **mm_kwargs)
    attention_scores = attention_scores * (1.0 / (head_dim**0.5))
    attention_probs = ttnn.softmax(attention_scores, dim=-1)

    context = ttnn.matmul(attention_probs, value, **mm_kwargs)
    context = ttnn.permute(context, (0, 2, 1, 3))
    context = ttnn.to_layout(context, ttnn.ROW_MAJOR_LAYOUT)
    context = ttnn.reshape(context, (batch_size, seq_len, hidden_size))
    context = ttnn.to_layout(context, ttnn.TILE_LAYOUT)

    output = ttnn.matmul(context, parameters["output.dense.weight"], **mm_kwargs)
    output = output + parameters["output.dense.bias"]
    return output


def preprocess_attention_parameters(state_dict, layer_idx, *, dtype=ttnn.bfloat16):
    """
    Preprocess attention parameters for a single layer from HuggingFace state dict.

    Returns dict with transposed linear weights and reshaped biases.
    """
    prefix = f"backbone.encoder.layer.{layer_idx}.attention"
    params = {}

    for name in ["query", "key", "value"]:
        w = state_dict[f"{prefix}.attention.{name}.weight"].T.contiguous()
        b = state_dict[f"{prefix}.attention.{name}.bias"].reshape(1, -1)
        params[f"attention.{name}.weight"] = ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        params[f"attention.{name}.bias"] = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    w = state_dict[f"{prefix}.output.dense.weight"].T.contiguous()
    b = state_dict[f"{prefix}.output.dense.bias"].reshape(1, -1)
    params["output.dense.weight"] = ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    params["output.dense.bias"] = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    return params
