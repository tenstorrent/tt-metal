# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.functional_bloom.reference import torch_functional_bloom
from models.experimental.functional_bloom.tt import ttnn_functional_bloom
from transformers import BloomForCausalLM, BloomTokenizerFast
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


# Verify that the torch functional model matches exactly the default model.
def test_torch_bloom_generate():
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
    input_text = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."
    # See torch_baseline_bloom.py for generating expected text
    expected_generated_text = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information. You can also add a few more sentences to the summary. The summary is a great way to get a quick"
    generated_text = torch_functional_bloom.generate_text_with_functional_approach(
        input_text, model, tokenizer, max_length=64
    )
    assert expected_generated_text == generated_text


def test_merge_heads(device):
    torch.manual_seed(0)
    torch_tensor = torch_random((16, 1, 64, 64), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output = torch_functional_bloom._merge_heads(torch_tensor, 16, 64)
    tt_tensor = ttnn.from_torch(torch_tensor)
    tt_tensor = ttnn.to_device(tt_tensor, device)
    tt_output = ttnn_functional_bloom._merge_heads(tt_tensor, 16, 64)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.99)


def test_layer_normalization(device):
    torch.manual_seed(0)
    torch_hidden = torch_random((1, 64, 1024), -0.1, 0.1, dtype=torch.bfloat16)
    torch_weight = torch_random((1024), -0.1, 0.1, dtype=torch.bfloat16)
    torch_bias = torch_random((1024), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output = torch_functional_bloom.layer_normalization(torch_hidden, torch_weight, torch_bias)
    tt_hidden = ttnn.from_torch(torch_hidden)
    tt_hidden = ttnn.to_device(tt_hidden, device)
    tt_weight = ttnn.from_torch(torch_weight)
    tt_weight = ttnn.to_device(tt_weight, device)
    tt_bias = ttnn.from_torch(torch_bias)
    tt_bias = ttnn.to_device(tt_bias, device)
    tt_output = ttnn_functional_bloom.layer_normalization(tt_hidden, tt_weight, tt_bias)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output, 0.99)


def test_create_query_key_value(device):
    torch.manual_seed(0)
    torch_hidden = torch_random((1, 64, 1024), -0.1, 0.1, dtype=torch.bfloat16)
    torch_weight = torch_random((1024, 3072), -0.1, 0.1, dtype=torch.bfloat16)
    torch_bias = torch_random((3072), -0.1, 0.1, dtype=torch.bfloat16)
    (torch_query_layer, torch_key_layer, torch_value_layer) = torch_functional_bloom.create_query_key_value(
        torch_hidden, torch_weight, torch_bias, 16, 64
    )
    tt_hidden = ttnn.from_torch(torch_hidden)
    tt_hidden = ttnn.to_device(tt_hidden, device)
    tt_weight = ttnn.from_torch(torch_weight)
    tt_weight = ttnn.to_device(tt_weight, device)
    tt_bias = ttnn.from_torch(torch_bias)
    tt_bias = ttnn.to_device(tt_bias, device)
    (tt_query_layer, tt_key_layer, tt_value_layer) = ttnn_functional_bloom.create_query_key_value(
        device, tt_hidden, tt_weight, tt_bias, 16, 64
    )
    tt_query_layer = ttnn.from_device(tt_query_layer)
    tt_query_layer = ttnn.to_layout(tt_query_layer, ttnn.ROW_MAJOR_LAYOUT)
    tt_query_layer = ttnn.to_torch(tt_query_layer)
    assert_with_pcc(torch_query_layer, tt_query_layer, 0.99)
    tt_key_layer = ttnn.from_device(tt_key_layer)
    tt_key_layer = ttnn.to_layout(tt_key_layer, ttnn.ROW_MAJOR_LAYOUT)
    tt_key_layer = ttnn.to_torch(tt_key_layer)
    assert_with_pcc(torch_key_layer, tt_key_layer, 0.99)
    tt_value_layer = ttnn.from_device(tt_value_layer)
    tt_value_layer = ttnn.to_layout(tt_value_layer, ttnn.ROW_MAJOR_LAYOUT)
    tt_value_layer = ttnn.to_torch(tt_value_layer)
    assert_with_pcc(torch_value_layer, tt_value_layer, 0.99)


def test_compute_attention_scores(device):
    torch.manual_seed(0)
    torch_query = torch_random((16, 1, 64, 64), -0.1, 0.1, dtype=torch.bfloat16)
    torch_key = torch_random((16, 1, 64, 64), -0.1, 0.1, dtype=torch.bfloat16)
    torch_alibi = torch_random((16, 1, 64, 64), -0.1, 0.1, dtype=torch.bfloat16)
    torch_attention_scores = torch_functional_bloom.compute_attention_scores(torch_query, torch_key, torch_alibi, 64, 1)
    tt_query = ttnn.from_torch(torch_query)
    tt_query = ttnn.to_device(tt_query, device)
    tt_key = ttnn.from_torch(torch_key)
    tt_key = ttnn.to_device(tt_key, device)
    tt_alibi = ttnn.from_torch(torch_alibi)
    tt_alibi = ttnn.to_device(tt_alibi, device)
    tt_attention_scores = ttnn_functional_bloom.compute_attention_scores(tt_query, tt_key, tt_alibi, 64, 1)
    tt_attention_scores = ttnn.from_device(tt_attention_scores)
    tt_attention_scores = ttnn.to_layout(tt_attention_scores, ttnn.ROW_MAJOR_LAYOUT)
    tt_attention_scores = ttnn.to_torch(tt_attention_scores)
    assert_with_pcc(torch_attention_scores, tt_attention_scores, 0.99)


@pytest.mark.skip(reason="Test failing do to softmax not handling tensors with negative values")
def test_compute_attention_probs(device):
    torch.manual_seed(0)
    torch_attention_scores = torch_random((1, 2, 4, 4), -1, 1, dtype=torch.bfloat16)
    torch_attention_mask = torch_random((1, 2, 4, 4), -1, 1, dtype=torch.bfloat16)
    torch_attention_probes = torch_functional_bloom.compute_attention_probs(
        torch_attention_scores, torch_attention_mask
    )
    tt_attention_scores = ttnn.from_torch(torch_attention_scores)
    tt_attention_scores = ttnn.to_device(tt_attention_scores, device)
    tt_attention_mask = ttnn.from_torch(torch_attention_mask)
    tt_attention_mask = ttnn.to_device(tt_attention_mask, device)
    tt_attention_probes = ttnn_functional_bloom.compute_attention_probs(tt_attention_scores, tt_attention_mask)
    tt_attention_probes = ttnn.from_device(tt_attention_probes)
    tt_attention_probes = ttnn.to_layout(tt_attention_probes, ttnn.ROW_MAJOR_LAYOUT)
    tt_attention_probes = ttnn.to_torch(tt_attention_probes)
    assert_with_pcc(torch_attention_probes, tt_attention_probes, 0.99)


def test_compute_context_layer(device):
    torch.manual_seed(0)
    torch_attention_probes = torch_random((16, 1, 64, 64), -1, 1, dtype=torch.bfloat16)
    torch_values_layer = torch_random((16, 1, 64, 64), -1, 1, dtype=torch.bfloat16)
    torch_context_layer = torch_functional_bloom.compute_context_layer(
        torch_attention_probes, torch_values_layer, 16, 64
    )
    tt_attention_probes = ttnn.from_torch(torch_attention_probes)
    tt_attention_probes = ttnn.to_device(tt_attention_probes, device)
    tt_values_layer = ttnn.from_torch(torch_values_layer)
    tt_values_layer = ttnn.to_device(tt_values_layer, device)
    tt_context_layer = ttnn_functional_bloom.compute_context_layer(tt_attention_probes, tt_values_layer, 16, 64)
    tt_context_layer = ttnn.from_device(tt_context_layer)
    tt_context_layer = ttnn.to_layout(tt_context_layer, ttnn.ROW_MAJOR_LAYOUT)
    tt_context_layer = ttnn.to_torch(tt_context_layer)
    assert_with_pcc(torch_context_layer, tt_context_layer, 0.99)


def test_finalize_output(device):
    torch.manual_seed(0)
    torch_context_layer = torch_random((1, 64, 1024), -1, 1, dtype=torch.bfloat16)
    torch_output_weight = torch_random((1024, 1024), -1, 1, dtype=torch.bfloat16)
    torch_output_bias = torch_random((1024), -1, 1, dtype=torch.bfloat16)
    torch_hidden_states = torch_random((1, 64, 1024), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch_functional_bloom.finalize_output(
        torch_context_layer, torch_output_weight, torch_output_bias, torch_hidden_states
    )
    tt_context_layer = ttnn.from_torch(torch_context_layer)
    tt_context_layer = ttnn.to_device(tt_context_layer, device)
    tt_output_weight = ttnn.from_torch(torch_output_weight)
    tt_output_weight = ttnn.to_device(tt_output_weight, device)
    tt_output_bias = ttnn.from_torch(torch_output_bias)
    tt_output_bias = ttnn.to_device(tt_output_bias, device)
    tt_hidden_states = ttnn.from_torch(torch_hidden_states)
    tt_hidden_states = ttnn.to_device(tt_hidden_states, device)
    tt_output_tensor = ttnn_functional_bloom.finalize_output(
        tt_context_layer, tt_output_weight, tt_output_bias, tt_hidden_states
    )
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_layout(tt_output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.99)


def test_mlp(device):
    torch.manual_seed(0)
    torch_attention_output = torch_random((1, 64, 1024), -1, 1, dtype=torch.bfloat16)
    torch_layernorm_output = torch_random((1, 64, 1024), -1, 1, dtype=torch.bfloat16)
    torch_h_to_4h_weight = torch_random((1024, 4096), -1, 1, dtype=torch.bfloat16)
    torch_h_to_4h_bias = torch_random((4096), -1, 1, dtype=torch.bfloat16)
    torch_dense_h_to_4h_weight = torch_random((4096, 1024), -1, 1, dtype=torch.bfloat16)
    torch_dense_h_to_4h_bias = torch_random((1024), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch_functional_bloom.mlp(
        torch_attention_output,
        torch_layernorm_output,
        torch_h_to_4h_weight,
        torch_h_to_4h_bias,
        torch_dense_h_to_4h_weight,
        torch_dense_h_to_4h_bias,
    )
    tt_attention_output = ttnn.from_torch(torch_attention_output)
    tt_attention_output = ttnn.to_device(tt_attention_output, device)
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
        tt_attention_output,
        tt_layernorm_output,
        tt_h_to_4h_weight,
        tt_h_to_4h_bias,
        tt_dense_h_to_4h_weight,
        tt_dense_h_to_4h_bias,
    )
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_layout(tt_output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.99)


@pytest.mark.skip(reason="Test failing do to softmax not handling tensors with negative values")
def test_bloom_multi_head_attention(device):
    torch.manual_seed(0)
    hidden_states = torch_random((1, 64, 1024), -0.1, 0.1, dtype=torch.bfloat16)
    hidden_states = torch_random((1, 64, 1024), -0.1, 0.1, dtype=torch.bfloat16)
    attention_mask = torch_random((1, 16, 64, 64), -0.1, 0.1, dtype=torch.bfloat16)
    input_layernorm_weight = torch_random((1024), -0.1, 0.1, dtype=torch.bfloat16)
    input_layernorm_bias = torch_random((1024), -0.1, 0.1, dtype=torch.bfloat16)
    query_key_value_weight = torch_random((1024, 3072), -0.1, 0.1, dtype=torch.bfloat16)
    query_key_value_bias = torch_random((3072), -0.1, 0.1, dtype=torch.bfloat16)
    output_weight = torch_random((1024, 1024), -0.1, 0.1, dtype=torch.bfloat16)
    output_bias = torch_random((1, 64, 1024), -0.1, 0.1, dtype=torch.bfloat16)
    alibi = torch_random((16, 1, 64, 64), -0.1, 0.1, dtype=torch.bfloat16)
    head_size = 64
    torch_attn_outputs = torch_functional_bloom.bloom_multi_head_attention(
        hidden_states,
        attention_mask,
        input_layernorm_weight,
        input_layernorm_bias,
        query_key_value_weight,
        query_key_value_bias,
        output_weight,
        output_bias,
        alibi,
        head_size=head_size,
    )

    hidden_states = ttnn.from_torch(hidden_states)
    attention_mask = ttnn.from_torch(attention_mask)
    input_layernorm_weight = ttnn.from_torch(input_layernorm_weight)
    input_layernorm_bias = ttnn.from_torch(input_layernorm_bias)
    query_key_value_weight = ttnn.from_torch(query_key_value_weight)
    query_key_value_bias = ttnn.from_torch(query_key_value_bias)
    output_weight = ttnn.from_torch(output_weight)
    output_bias = ttnn.from_torch(output_bias)
    alibi = ttnn.from_torch(alibi)

    hidden_states = ttnn.to_device(hidden_states, device)
    attention_mask = ttnn.to_device(attention_mask, device)
    input_layernorm_weight = ttnn.to_device(input_layernorm_weight, device)
    input_layernorm_bias = ttnn.to_device(input_layernorm_bias, device)
    query_key_value_weight = ttnn.to_device(query_key_value_weight, device)
    query_key_value_bias = ttnn.to_device(query_key_value_bias, device)
    output_weight = ttnn.to_device(output_weight, device)
    output_bias = ttnn.to_device(output_bias, device)
    alibi = ttnn.to_device(alibi, device)

    tt_attn_outputs = ttnn_functional_bloom.bloom_multi_head_attention(
        device,
        hidden_states,
        attention_mask,
        input_layernorm_weight,
        input_layernorm_bias,
        query_key_value_weight,
        query_key_value_bias,
        output_weight,
        output_bias,
        alibi,
        head_size=head_size,
    )

    torch_attention_output = torch_attn_outputs[0]
    torch_outputs = torch_attn_outputs[1][0]

    tt_attention_output = tt_attn_outputs[0]
    tt_outputs = tt_attn_outputs[1][0]

    tt_attention_output = ttnn.from_device(tt_attention_output)
    tt_attention_output = ttnn.to_torch(tt_attention_output)

    tt_outputs = ttnn.from_device(tt_outputs)
    tt_outputs = ttnn.to_layout(tt_outputs, ttnn.ROW_MAJOR_LAYOUT)
    tt_outputs = ttnn.to_torch(tt_outputs)

    assert_with_pcc(torch_attention_output, tt_attention_output, 0.99)
    assert_with_pcc(torch_outputs, tt_outputs, 0.99)


@pytest.mark.skip(reason="Models do not match even when using torch.bfloat16")
def test_ttnn_bloom_generate():
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
    input_text = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."
    # See torch_baseline_bloom.py for generating expected text
    expected_generated_text = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information. You can also add a few more sentences to the summary. The summary is a great way to get a quick"

    generated_text = ttnn_functional_bloom.generate_text_with_functional_approach(
        input_text, model, tokenizer, max_length=42 + 5
    )
    assert expected_generated_text == generated_text
