# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn.functional as F

import ttnn

from models.experimental.functional_bert.tt.ttnn_functional_bert import ttnn_multi_head_attention, ttnn_feedforward
from models.experimental.functional_bert.reference.torch_functional_bert import (
    torch_multi_head_attention,
    torch_feedforward,
)
from models.utility_functions import torch_random

from tests.ttnn.utils_for_testing import assert_with_pcc, torch_random
from models.utility_functions import skip_for_wormhole_b0


def ttnn_model(
    hidden_states,
    attention_mask,
    query_weight,
    query_bias,
    key_weight,
    key_bias,
    value_weight,
    value_bias,
    output_weight,
    output_bias,
    attention_layer_norm_weight,
    attention_layer_norm_bias,
    ff1_weight,
    ff1_bias,
    ff2_weight,
    ff2_bias,
    ff_layer_norm_weight,
    ff_layer_norm_bias,
    *,
    num_heads,
):
    multi_head_attention_output = ttnn_multi_head_attention(
        hidden_states,
        attention_mask,
        query_weight,
        query_bias,
        key_weight,
        key_bias,
        value_weight,
        value_bias,
        output_weight,
        output_bias,
        num_heads=num_heads,
    )
    hidden_states = ttnn.layer_norm(
        hidden_states + multi_head_attention_output, weight=attention_layer_norm_weight, bias=attention_layer_norm_bias
    )

    feedforward_output = ttnn_feedforward(
        hidden_states,
        ff1_weight,
        ff1_bias,
        ff2_weight,
        ff2_bias,
    )
    hidden_states = ttnn.layer_norm(
        hidden_states + feedforward_output, weight=ff_layer_norm_weight, bias=ff_layer_norm_bias
    )
    return hidden_states


def torch_model(
    hidden_states,
    attention_mask,
    query_weight,
    query_bias,
    key_weight,
    key_bias,
    value_weight,
    value_bias,
    output_weight,
    output_bias,
    attention_layer_norm_weight,
    attention_layer_norm_bias,
    ff1_weight,
    ff1_bias,
    ff2_weight,
    ff2_bias,
    ff_layer_norm_weight,
    ff_layer_norm_bias,
    *,
    num_heads,
):
    *_, hidden_size = hidden_states.shape
    multi_head_attention_output = torch_multi_head_attention(
        hidden_states,
        attention_mask,
        query_weight,
        query_bias,
        key_weight,
        key_bias,
        value_weight,
        value_bias,
        output_weight,
        output_bias,
        num_heads=num_heads,
    )
    hidden_states = F.layer_norm(
        hidden_states + multi_head_attention_output,
        normalized_shape=(hidden_size,),
        weight=attention_layer_norm_weight,
        bias=attention_layer_norm_bias,
    )

    feedforward_output = torch_feedforward(
        hidden_states,
        ff1_weight,
        ff1_bias,
        ff2_weight,
        ff2_bias,
    )
    hidden_states = F.layer_norm(
        hidden_states + feedforward_output,
        normalized_shape=(hidden_size,),
        weight=ff_layer_norm_weight,
        bias=ff_layer_norm_bias,
    )
    return hidden_states


# Note that our reshape requires the width and height to both be multiples of 32
# so the number of heads must be 32
@skip_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [2 * 32])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("head_size", [32])
def test_encoder(device, use_program_cache, batch_size, sequence_size, num_heads, head_size):
    torch.manual_seed(0)

    hidden_size = num_heads * head_size
    intermediate_size = hidden_size * 4

    torch_hidden_states = torch_random((batch_size, sequence_size, hidden_size), -0.1, 0.1, dtype=torch.bfloat16)

    torch_attention_mask = torch.zeros((sequence_size,), dtype=torch.bfloat16)
    torch_attention_mask[2:] = -1e9

    torch_query_weight = torch_random((hidden_size, hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    torch_query_bias = torch_random((hidden_size,), -0.1, 0.1, dtype=torch.bfloat16)
    torch_key_weight = torch_random((hidden_size, hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    torch_key_bias = torch_random((hidden_size,), -0.1, 0.1, dtype=torch.bfloat16)
    torch_value_weight = torch_random((hidden_size, hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    torch_value_bias = torch_random((hidden_size,), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_weight = torch_random((hidden_size, hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_bias = torch_random((hidden_size,), -0.1, 0.1, dtype=torch.bfloat16)
    torch_attention_layer_norm_weight = torch_random((hidden_size,), -0.1, 0.1, dtype=torch.bfloat16)
    torch_attention_layer_norm_bias = torch_random((hidden_size,), -0.1, 0.1, dtype=torch.bfloat16)
    torch_ff1_weight = torch_random((hidden_size, intermediate_size), -0.1, 0.1, dtype=torch.bfloat16)
    torch_ff1_bias = torch_random((intermediate_size,), -0.1, 0.1, dtype=torch.bfloat16)
    torch_ff2_weight = torch_random((intermediate_size, hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    torch_ff2_bias = torch_random((hidden_size,), -0.1, 0.1, dtype=torch.bfloat16)
    torch_ff_layer_norm_weight = torch_random((hidden_size,), -0.1, 0.1, dtype=torch.bfloat16)
    torch_ff_layer_norm_bias = torch_random((hidden_size,), -0.1, 0.1, dtype=torch.bfloat16)

    torch_output = torch_model(
        torch_hidden_states,
        torch_attention_mask,
        torch_query_weight,
        torch_query_bias,
        torch_key_weight,
        torch_key_bias,
        torch_value_weight,
        torch_value_bias,
        torch_output_weight,
        torch_output_bias,
        torch_attention_layer_norm_weight,
        torch_attention_layer_norm_bias,
        torch_ff1_weight,
        torch_ff1_bias,
        torch_ff2_weight,
        torch_ff2_bias,
        torch_ff_layer_norm_weight,
        torch_ff_layer_norm_bias,
        num_heads=num_heads,
    )

    assert torch_output.shape == (
        batch_size,
        sequence_size,
        hidden_size,
    ), f"Expected output shape to be {batch_size, sequence_size, hidden_size}, got {torch_output.shape}"

    hidden_states = ttnn.from_torch(torch_hidden_states)
    attention_mask = ttnn.from_torch(torch_attention_mask)

    query_weight = ttnn.from_torch(torch_query_weight)
    query_bias = ttnn.from_torch(torch_query_bias)
    key_weight = ttnn.from_torch(torch_key_weight)
    key_bias = ttnn.from_torch(torch_key_bias)
    value_weight = ttnn.from_torch(torch_value_weight)
    value_bias = ttnn.from_torch(torch_value_bias)
    output_weight = ttnn.from_torch(torch_output_weight)
    output_bias = ttnn.from_torch(torch_output_bias)
    attention_layer_norm_weight = ttnn.from_torch(torch_attention_layer_norm_weight)
    attention_layer_norm_bias = ttnn.from_torch(torch_attention_layer_norm_bias)
    ff1_weight = ttnn.from_torch(torch_ff1_weight)
    ff1_bias = ttnn.from_torch(torch_ff1_bias)
    ff2_weight = ttnn.from_torch(torch_ff2_weight)
    ff2_bias = ttnn.from_torch(torch_ff2_bias)
    ff_layer_norm_weight = ttnn.from_torch(torch_ff_layer_norm_weight)
    ff_layer_norm_bias = ttnn.from_torch(torch_ff_layer_norm_bias)

    hidden_states = ttnn.to_device(hidden_states, device)
    attention_mask = ttnn.to_device(attention_mask, device)
    query_weight = ttnn.to_device(query_weight, device)
    query_bias = ttnn.to_device(query_bias, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    key_weight = ttnn.to_device(key_weight, device)
    key_bias = ttnn.to_device(key_bias, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    value_weight = ttnn.to_device(value_weight, device)
    value_bias = ttnn.to_device(value_bias, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_weight = ttnn.to_device(output_weight, device)
    output_bias = ttnn.to_device(output_bias, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    attention_layer_norm_weight = ttnn.to_device(attention_layer_norm_weight, device)
    attention_layer_norm_bias = ttnn.to_device(attention_layer_norm_bias, device)
    ff1_weight = ttnn.to_device(ff1_weight, device)
    ff1_bias = ttnn.to_device(ff1_bias, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    ff2_weight = ttnn.to_device(ff2_weight, device)
    ff2_bias = ttnn.to_device(ff2_bias, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    ff_layer_norm_weight = ttnn.to_device(ff_layer_norm_weight, device)
    ff_layer_norm_bias = ttnn.to_device(ff_layer_norm_bias, device)

    tt_output = ttnn_model(
        hidden_states,
        attention_mask,
        query_weight,
        query_bias,
        key_weight,
        key_bias,
        value_weight,
        value_bias,
        output_weight,
        output_bias,
        attention_layer_norm_weight,
        attention_layer_norm_bias,
        ff1_weight,
        ff1_bias,
        ff2_weight,
        ff2_bias,
        ff_layer_norm_weight,
        ff_layer_norm_bias,
        num_heads=num_heads,
    )

    assert tt_output.shape == [
        batch_size,
        sequence_size,
        hidden_size,
    ], f"Expected output shape to be {batch_size, sequence_size, hidden_size}, got {tt_output.shape}"

    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.999)
