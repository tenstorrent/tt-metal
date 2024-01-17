# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch


import ttnn
from ttnn.model_preprocessing import (
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.experimental.functional_bloom.reference import torch_functional_bloom
from models.experimental.functional_bloom.tt import ttnn_functional_bloom, ttnn_optimized_functional_bloom
from models.utility_functions import torch_random, skip_for_wormhole_b0

from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_wormhole_b0()
def test_merge_heads(device):
    torch.manual_seed(0)
    torch_tensor = torch_random((1, 16, 64, 64), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output = torch_functional_bloom.merge_heads(torch_tensor)
    tt_tensor = ttnn.from_torch(torch_tensor)
    tt_tensor = ttnn.to_device(tt_tensor, device)
    tt_output = ttnn_functional_bloom.merge_heads(tt_tensor)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@skip_for_wormhole_b0()
def test_optimized_merge_heads(device):
    torch.manual_seed(0)
    torch_tensor = torch_random((8, 16, 384, 64), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output = torch_functional_bloom.merge_heads(torch_tensor)
    tt_tensor = ttnn.from_torch(torch_tensor)
    tt_tensor = ttnn.to_device(tt_tensor, device)
    tt_tensor = ttnn.to_layout(tt_tensor, ttnn.TILE_LAYOUT)
    tt_output = ttnn_optimized_functional_bloom.merge_heads(tt_tensor)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.9999)


@skip_for_wormhole_b0()
def test_create_query_key_value(device):
    batch_size = 1
    sequence_size = 64
    num_heads = 16
    head_size = 64
    hidden_size = num_heads * head_size

    torch.manual_seed(0)
    torch_hidden = torch_random((batch_size, sequence_size, hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    torch_weight = torch_random((hidden_size, hidden_size * 3), -0.1, 0.1, dtype=torch.bfloat16)
    torch_bias = torch_random((hidden_size * 3), -0.1, 0.1, dtype=torch.bfloat16)
    (torch_query_layer, torch_key_layer, torch_value_layer) = torch_functional_bloom.create_query_key_value(
        torch_hidden, torch_weight, torch_bias, num_heads
    )
    tt_hidden = ttnn.from_torch(torch_hidden)
    tt_hidden = ttnn.to_device(tt_hidden, device)
    tt_weight = ttnn.from_torch(torch_weight)
    tt_weight = ttnn.to_device(tt_weight, device)
    tt_bias = ttnn.from_torch(torch_bias)
    tt_bias = ttnn.to_device(tt_bias, device)
    (tt_query_layer, tt_key_layer, tt_value_layer) = ttnn_functional_bloom.create_query_key_value(
        tt_hidden, tt_weight, tt_bias, num_heads, use_core_grid=False
    )
    tt_query_layer = ttnn.to_torch(tt_query_layer)
    assert_with_pcc(torch_query_layer, tt_query_layer, 0.9991)

    tt_key_layer = ttnn.to_torch(tt_key_layer)
    assert_with_pcc(torch_key_layer, tt_key_layer, 0.9991)

    tt_value_layer = ttnn.to_torch(tt_value_layer)
    assert_with_pcc(torch_value_layer, tt_value_layer, 0.9991)


@skip_for_wormhole_b0()
def test_optimized_create_query_key_value(device):
    batch_size = 8
    sequence_size = 384
    num_heads = 16
    head_size = 64
    hidden_size = num_heads * head_size

    torch.manual_seed(0)
    torch_hidden = torch_random((batch_size, sequence_size, hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    torch_weight = torch_random((hidden_size, hidden_size * 3), -0.1, 0.1, dtype=torch.bfloat16)
    torch_bias = torch_random((hidden_size * 3), -0.1, 0.1, dtype=torch.bfloat16)
    (torch_query_layer, torch_key_layer, torch_value_layer) = torch_functional_bloom.create_query_key_value(
        torch_hidden, torch_weight, torch_bias, num_heads
    )
    tt_hidden = ttnn.from_torch(torch_hidden)
    tt_hidden = ttnn.to_device(tt_hidden, device)
    tt_hidden = ttnn.to_layout(tt_hidden, ttnn.TILE_LAYOUT)
    tt_weight = preprocess_linear_weight(torch_weight.T, dtype=ttnn.bfloat16)
    tt_weight = ttnn.to_device(tt_weight, device)
    tt_bias = preprocess_linear_bias(torch_bias, dtype=ttnn.bfloat16)
    tt_bias = ttnn.to_device(tt_bias, device)
    (tt_query_layer, tt_key_layer, tt_value_layer) = ttnn_optimized_functional_bloom.create_query_key_value(
        tt_hidden, tt_weight, tt_bias, num_heads=num_heads
    )

    tt_query_layer = ttnn.to_torch(tt_query_layer)
    assert_with_pcc(torch_query_layer, tt_query_layer, 0.9986)

    tt_key_layer = ttnn.to_torch(tt_key_layer)
    assert_with_pcc(torch_key_layer, tt_key_layer, 0.9986)

    tt_value_layer = ttnn.to_torch(tt_value_layer)
    assert_with_pcc(torch_value_layer, tt_value_layer, 0.9985)


@skip_for_wormhole_b0()
def test_compute_attention_scores(device):
    torch.manual_seed(0)
    torch_query = torch_random((16, 1, 64, 64), -0.1, 0.1, dtype=torch.bfloat16)
    torch_key = torch_random((16, 1, 64, 64), -0.1, 0.1, dtype=torch.bfloat16)
    torch_alibi = torch_random((16, 1, 64, 64), -0.1, 0.1, dtype=torch.bfloat16)
    torch_attention_scores = torch_functional_bloom.compute_attention_scores(torch_query, torch_key, torch_alibi)
    tt_query = ttnn.from_torch(torch_query)
    tt_query = ttnn.to_device(tt_query, device)
    tt_key = ttnn.from_torch(torch_key)
    tt_key = ttnn.to_device(tt_key, device)
    tt_alibi = ttnn.from_torch(torch_alibi)
    tt_alibi = ttnn.to_device(tt_alibi, device)
    tt_attention_scores = ttnn_functional_bloom.compute_attention_scores(tt_query, tt_key, tt_alibi)
    tt_attention_scores = ttnn.from_device(tt_attention_scores)
    tt_attention_scores = ttnn.to_layout(tt_attention_scores, ttnn.ROW_MAJOR_LAYOUT)
    tt_attention_scores = ttnn.to_torch(tt_attention_scores)
    assert_with_pcc(torch_attention_scores, tt_attention_scores, 0.9999)


@skip_for_wormhole_b0()
def test_optimized_compute_attention_scores(device):
    torch.manual_seed(0)
    torch_query = torch_random((16, 1, 64, 64), -0.1, 0.1, dtype=torch.bfloat16)
    torch_key = torch_random((16, 1, 64, 64), -0.1, 0.1, dtype=torch.bfloat16)
    torch_alibi = torch_random((16, 1, 64, 64), -0.1, 0.1, dtype=torch.bfloat16)
    torch_attention_scores = torch_functional_bloom.compute_attention_scores(torch_query, torch_key, torch_alibi)
    tt_query = ttnn.from_torch(torch_query)
    tt_query = ttnn.to_device(tt_query, device)
    tt_query = ttnn.to_layout(tt_query, ttnn.TILE_LAYOUT)
    tt_key = ttnn.from_torch(torch_key)
    tt_key = ttnn.to_device(tt_key, device)
    tt_key = ttnn.to_layout(tt_key, ttnn.TILE_LAYOUT)
    tt_alibi = ttnn.from_torch(torch_alibi)
    tt_alibi = ttnn.to_device(tt_alibi, device)
    tt_attention_scores = ttnn_optimized_functional_bloom.compute_attention_scores(tt_query, tt_key, tt_alibi)
    tt_attention_scores = ttnn.from_device(tt_attention_scores)
    tt_attention_scores = ttnn.to_layout(tt_attention_scores, ttnn.ROW_MAJOR_LAYOUT)
    tt_attention_scores = ttnn.to_torch(tt_attention_scores)
    assert_with_pcc(torch_attention_scores, tt_attention_scores, 0.9999)


@skip_for_wormhole_b0()
def test_compute_attention_probs(device):
    torch.manual_seed(0)
    torch_attention_scores = torch_random((1, 2, 4, 4), -0.1, 0.1, dtype=torch.bfloat16)
    torch_attention_mask = torch_random((1, 2, 4, 4), -0.1, 0.1, dtype=torch.bfloat16)
    torch_attention_probes = torch_functional_bloom.compute_attention_probs(
        torch_attention_scores, torch_attention_mask
    )
    tt_attention_scores = ttnn.from_torch(torch_attention_scores)
    tt_attention_scores = ttnn.to_device(tt_attention_scores, device)
    tt_attention_mask = ttnn.from_torch(torch_attention_mask * -100)
    tt_attention_mask = ttnn.to_device(tt_attention_mask, device)
    tt_attention_probes = ttnn_functional_bloom.compute_attention_probs(tt_attention_scores, tt_attention_mask)
    tt_attention_probes = ttnn.from_device(tt_attention_probes)
    tt_attention_probes = ttnn.to_layout(tt_attention_probes, ttnn.ROW_MAJOR_LAYOUT)
    tt_attention_probes = ttnn.to_torch(tt_attention_probes)
    assert_with_pcc(torch_attention_probes, tt_attention_probes, 0.9999)


@skip_for_wormhole_b0()
def test_compute_context_layer(device):
    torch.manual_seed(0)
    torch_attention_probes = torch_random((1, 16, 64, 64), -0.1, 0.1, dtype=torch.bfloat16)
    torch_values_layer = torch_random((1, 16, 64, 64), -0.1, 0.1, dtype=torch.bfloat16)
    torch_context_layer = torch_functional_bloom.compute_context_layer(torch_attention_probes, torch_values_layer)
    tt_attention_probes = ttnn.from_torch(torch_attention_probes)
    tt_attention_probes = ttnn.to_device(tt_attention_probes, device)
    tt_values_layer = ttnn.from_torch(torch_values_layer)
    tt_values_layer = ttnn.to_device(tt_values_layer, device)
    tt_context_layer = ttnn_functional_bloom.compute_context_layer(tt_attention_probes, tt_values_layer)
    tt_context_layer = ttnn.from_device(tt_context_layer)
    tt_context_layer = ttnn.to_layout(tt_context_layer, ttnn.ROW_MAJOR_LAYOUT)
    tt_context_layer = ttnn.to_torch(tt_context_layer)
    assert_with_pcc(torch_context_layer, tt_context_layer, 0.9999)


@skip_for_wormhole_b0()
def test_optimized_compute_context_layer(device):
    torch.manual_seed(0)
    torch_attention_probes = torch_random((8, 16, 384, 384), -0.1, 0.1, dtype=torch.bfloat16)
    torch_values_layer = torch_random((8, 16, 384, 64), -0.1, 0.1, dtype=torch.bfloat16)
    torch_context_layer = torch_functional_bloom.compute_context_layer(torch_attention_probes, torch_values_layer)
    tt_attention_probes = ttnn.from_torch(torch_attention_probes)
    tt_attention_probes = ttnn.to_device(tt_attention_probes, device)
    tt_attention_probes = ttnn.to_layout(tt_attention_probes, ttnn.TILE_LAYOUT)
    tt_values_layer = ttnn.from_torch(torch_values_layer)
    tt_values_layer = ttnn.to_device(tt_values_layer, device)
    tt_values_layer = ttnn.to_layout(tt_values_layer, ttnn.TILE_LAYOUT)
    tt_context_layer = ttnn_optimized_functional_bloom.compute_context_layer(tt_attention_probes, tt_values_layer)
    tt_context_layer = ttnn.from_device(tt_context_layer)
    tt_context_layer = ttnn.to_layout(tt_context_layer, ttnn.ROW_MAJOR_LAYOUT)
    tt_context_layer = ttnn.to_torch(tt_context_layer)
    assert_with_pcc(torch_context_layer, tt_context_layer, 0.9985)


@skip_for_wormhole_b0()
def test_finalize_output(device):
    torch.manual_seed(0)
    torch_context_layer = torch_random((1, 64, 1024), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_weight = torch_random((1024, 1024), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_bias = torch_random((1024), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = torch_functional_bloom.finalize_output(
        torch_context_layer, torch_output_weight, torch_output_bias
    )
    tt_context_layer = ttnn.from_torch(torch_context_layer)
    tt_context_layer = ttnn.to_device(tt_context_layer, device)
    tt_output_weight = ttnn.from_torch(torch_output_weight)
    tt_output_weight = ttnn.to_device(tt_output_weight, device)
    tt_output_bias = ttnn.from_torch(torch_output_bias)
    tt_output_bias = ttnn.to_device(tt_output_bias, device)
    tt_output_tensor = ttnn_functional_bloom.finalize_output(tt_context_layer, tt_output_weight, tt_output_bias)
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_layout(tt_output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.99918)


@skip_for_wormhole_b0()
def test_mlp(device):
    torch.manual_seed(0)
    torch_layernorm_output = torch_random((1, 64, 1024), -0.1, 0.1, dtype=torch.bfloat16)
    torch_h_to_4h_weight = torch_random((1024, 4096), -0.1, 0.1, dtype=torch.bfloat16)
    torch_h_to_4h_bias = torch_random((4096), -0.1, 0.1, dtype=torch.bfloat16)
    torch_dense_h_to_4h_weight = torch_random((4096, 1024), -0.1, 0.1, dtype=torch.bfloat16)
    torch_dense_h_to_4h_bias = torch_random((1024), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = torch_functional_bloom.mlp(
        torch_layernorm_output,
        torch_h_to_4h_weight,
        torch_h_to_4h_bias,
        torch_dense_h_to_4h_weight,
        torch_dense_h_to_4h_bias,
    )
    tt_layernorm_output = ttnn.from_torch(torch_layernorm_output)
    tt_layernorm_output = ttnn.to_device(tt_layernorm_output, device)
    tt_h_to_4h_weight = ttnn.from_torch(torch_h_to_4h_weight)
    tt_h_to_4h_weight = ttnn.to_device(tt_h_to_4h_weight, device)
    tt_h_to_4h_bias = ttnn.from_torch(torch_h_to_4h_bias)
    tt_h_to_4h_bias = ttnn.to_device(tt_h_to_4h_bias, device)
    tt_dense_h_to_4h_weight = ttnn.from_torch(torch_dense_h_to_4h_weight)
    tt_dense_h_to_4h_weight = ttnn.to_device(tt_dense_h_to_4h_weight, device)
    tt_dense_h_to_4h_bias = ttnn.from_torch(torch_dense_h_to_4h_bias)
    tt_dense_h_to_4h_bias = ttnn.to_device(tt_dense_h_to_4h_bias, device)
    tt_output_tensor = ttnn_functional_bloom.mlp(
        tt_layernorm_output,
        tt_h_to_4h_weight,
        tt_h_to_4h_bias,
        tt_dense_h_to_4h_weight,
        tt_dense_h_to_4h_bias,
    )
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_layout(tt_output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.9947)


@skip_for_wormhole_b0()
def test_optimized_mlp(device):
    torch.manual_seed(0)
    torch_layernorm_output = torch_random((1, 64, 1024), -1, 1, dtype=torch.bfloat16)
    torch_h_to_4h_weight = torch_random((1024, 4096), -1, 1, dtype=torch.bfloat16)
    torch_h_to_4h_bias = torch_random((4096), -1, 1, dtype=torch.bfloat16)
    torch_dense_h_to_4h_weight = torch_random((4096, 1024), -1, 1, dtype=torch.bfloat16)
    torch_dense_h_to_4h_bias = torch_random((1024), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch_functional_bloom.mlp(
        torch_layernorm_output,
        torch_h_to_4h_weight,
        torch_h_to_4h_bias,
        torch_dense_h_to_4h_weight,
        torch_dense_h_to_4h_bias,
    )

    tt_layernorm_output = ttnn.from_torch(torch_layernorm_output)
    tt_layernorm_output = ttnn.to_device(tt_layernorm_output, device)
    tt_layernorm_output = ttnn.to_layout(tt_layernorm_output, ttnn.TILE_LAYOUT)

    tt_h_to_4h_weight = preprocess_linear_weight(torch_h_to_4h_weight.T, dtype=ttnn.bfloat16)
    tt_h_to_4h_weight = ttnn.to_device(tt_h_to_4h_weight, device)

    tt_h_to_4h_bias = preprocess_linear_bias(torch_h_to_4h_bias, dtype=ttnn.bfloat16)
    tt_h_to_4h_bias = ttnn.to_device(tt_h_to_4h_bias, device)

    tt_dense_h_to_4h_weight = preprocess_linear_weight(torch_dense_h_to_4h_weight.T, dtype=ttnn.bfloat16)
    tt_dense_h_to_4h_weight = ttnn.to_device(tt_dense_h_to_4h_weight, device)

    tt_dense_h_to_4h_bias = preprocess_linear_bias(torch_dense_h_to_4h_bias, dtype=ttnn.bfloat16)
    tt_dense_h_to_4h_bias = ttnn.to_device(tt_dense_h_to_4h_bias, device)

    tt_output_tensor = ttnn_optimized_functional_bloom.mlp(
        tt_layernorm_output,
        tt_h_to_4h_weight,
        tt_h_to_4h_bias,
        tt_dense_h_to_4h_weight,
        tt_dense_h_to_4h_bias,
    )
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_layout(tt_output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.9965)


@skip_for_wormhole_b0()
def test_multi_head_attention(device):
    torch.manual_seed(0)

    batch_size = 1
    sequence_size = 64
    num_heads = 16
    head_size = 64
    hidden_size = num_heads * head_size

    hidden_states = torch_random((batch_size, sequence_size, hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    hidden_states = torch_random((batch_size, sequence_size, hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    causal_mask = torch_random((batch_size, num_heads, sequence_size, head_size), -0.1, 0.1, dtype=torch.bfloat16)
    query_key_value_weight = torch_random((hidden_size, hidden_size * 3), -0.1, 0.1, dtype=torch.bfloat16)
    query_key_value_bias = torch_random((hidden_size * 3), -0.1, 0.1, dtype=torch.bfloat16)
    output_weight = torch_random((hidden_size, hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    output_bias = torch_random((hidden_size,), -0.1, 0.1, dtype=torch.bfloat16)
    alibi = torch_random((batch_size, num_heads, 1, head_size), -0.1, 0.1, dtype=torch.bfloat16)
    torch_attention_output = torch_functional_bloom.multi_head_attention(
        hidden_states,
        alibi,
        causal_mask,
        query_key_value_weight,
        query_key_value_bias,
        output_weight,
        output_bias,
        num_heads=num_heads,
    )

    hidden_states = ttnn.from_torch(hidden_states)
    causal_mask = ttnn.from_torch(causal_mask * -100)
    query_key_value_weight = ttnn.from_torch(query_key_value_weight)
    query_key_value_bias = ttnn.from_torch(query_key_value_bias)
    output_weight = ttnn.from_torch(output_weight)
    output_bias = ttnn.from_torch(output_bias)
    alibi = ttnn.from_torch(alibi, dtype=ttnn.bfloat16)

    hidden_states = ttnn.to_device(hidden_states, device)
    causal_mask = ttnn.to_device(causal_mask, device)
    query_key_value_weight = ttnn.to_device(query_key_value_weight, device)
    query_key_value_bias = ttnn.to_device(query_key_value_bias, device)
    output_weight = ttnn.to_device(output_weight, device)
    output_bias = ttnn.to_device(output_bias, device)
    alibi = ttnn.to_device(alibi, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    tt_attention_output = ttnn_functional_bloom.multi_head_attention(
        hidden_states,
        alibi,
        causal_mask,
        query_key_value_weight,
        query_key_value_bias,
        output_weight,
        output_bias,
        num_heads=num_heads,
        use_core_grid=False,
    )

    tt_attention_output = ttnn.from_device(tt_attention_output)
    tt_attention_output = ttnn.to_layout(tt_attention_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_attention_output = ttnn.to_torch(tt_attention_output)

    assert_with_pcc(torch_attention_output, tt_attention_output, 0.9985)


@skip_for_wormhole_b0()
def test_optimized_multi_head_attention(device):
    torch.manual_seed(0)

    batch_size = 8
    sequence_size = 384
    num_heads = 16
    head_size = 64
    hidden_size = num_heads * head_size

    hidden_states = torch_random((batch_size, sequence_size, hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    causal_mask = torch_random((batch_size, num_heads, sequence_size, sequence_size), -0.1, 0.1, dtype=torch.bfloat16)
    query_key_value_weight = torch_random((hidden_size, hidden_size * 3), -0.1, 0.1, dtype=torch.bfloat16)
    query_key_value_bias = torch_random((hidden_size * 3,), -0.1, 0.1, dtype=torch.bfloat16)
    output_weight = torch_random((hidden_size, hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    output_bias = torch_random((hidden_size,), -0.1, 0.1, dtype=torch.bfloat16)
    alibi = torch_random((batch_size, num_heads, 1, sequence_size), -0.1, 0.1, dtype=torch.bfloat16)
    torch_attention_output = torch_functional_bloom.multi_head_attention(
        hidden_states,
        alibi,
        causal_mask,
        query_key_value_weight,
        query_key_value_bias,
        output_weight,
        output_bias,
        num_heads=num_heads,
    )

    hidden_states = ttnn.from_torch(hidden_states)
    causal_mask = ttnn.from_torch(causal_mask * -100)
    query_key_value_weight = preprocess_linear_weight(query_key_value_weight.T, dtype=ttnn.bfloat16)
    query_key_value_bias = preprocess_linear_bias(query_key_value_bias, dtype=ttnn.bfloat16)
    output_weight = preprocess_linear_weight(output_weight.T, dtype=ttnn.bfloat16)
    output_bias = preprocess_linear_bias(output_bias, dtype=ttnn.bfloat16)
    alibi = ttnn.from_torch(alibi, dtype=ttnn.bfloat16)

    hidden_states = ttnn.to_device(hidden_states, device)
    hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
    causal_mask = ttnn.to_device(causal_mask, device)
    causal_mask = ttnn.to_layout(causal_mask, ttnn.TILE_LAYOUT)
    query_key_value_weight = ttnn.to_device(query_key_value_weight, device)
    query_key_value_bias = ttnn.to_device(query_key_value_bias, device)
    output_weight = ttnn.to_device(output_weight, device)
    output_bias = ttnn.to_device(output_bias, device)
    alibi = ttnn.to_device(alibi, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    alibi = ttnn.to_layout(alibi, ttnn.TILE_LAYOUT)

    tt_attention_output = ttnn_optimized_functional_bloom.multi_head_attention(
        hidden_states,
        alibi,
        causal_mask,
        query_key_value_weight,
        query_key_value_bias,
        output_weight,
        output_bias,
        num_heads=num_heads,
    )

    tt_attention_output = ttnn.from_device(tt_attention_output)
    tt_attention_output = ttnn.to_layout(tt_attention_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_attention_output = ttnn.to_torch(tt_attention_output)

    assert_with_pcc(torch_attention_output, tt_attention_output, 0.9977)
