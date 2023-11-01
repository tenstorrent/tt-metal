# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn.functional as F

import ttnn

from models.experimental.functional_bert.torch_functional_bert import torch_multi_head_attention as torch_model
from models.experimental.functional_bert.ttnn_functional_bert import ttnn_multi_head_attention as ttnn_model
from models.utility_functions import torch_random

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


# Note that our reshape requires the width and height to both be multiples of 32
# so the number of heads must be 32
@skip_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [2 * 32])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("head_size", [32])
def test_multi_head_attention(device, use_program_cache, batch_size, sequence_size, num_heads, head_size):
    torch.manual_seed(0)

    hidden_size = num_heads * head_size

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
        head_size=head_size,
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
        head_size=head_size,
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
