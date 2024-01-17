# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("sequence_size", [384, 1024])
@pytest.mark.parametrize("target_sequence_size", [384, 4096])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_transformer_attention_softmax(
    batch_size,
    num_heads,
    sequence_size,
    target_sequence_size,
    input_dtype,
    input_memory_config,
    output_memory_config,
    *,
    device,
):
    torch.manual_seed(0)

    input_shape = (batch_size, num_heads, sequence_size, target_sequence_size)
    torch_input_tensor = torch_random(input_shape, 0, 1.0, dtype=torch.bfloat16)
    torch_output_tensor = ttnn.transformer._torch_attention_softmax(
        torch_input_tensor,
        head_size=None,
        attention_mask=None,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn.transformer.attention_softmax(
        input_tensor, head_size=None, attention_mask=None, memory_config=output_memory_config
    )
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    # TODO(arakhmati): attention_softmax should be more accurate
    assert_with_pcc(torch_output_tensor, output_tensor, 0.992)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_heads", [4, 16])
@pytest.mark.parametrize("sequence_size", [384, 1024])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_transformer_concatenate_heads(
    batch_size, num_heads, sequence_size, head_size, input_dtype, input_memory_config, output_memory_config, *, device
):
    torch.manual_seed(0)

    input_shape = (batch_size, num_heads, sequence_size, head_size)
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = ttnn.transformer._torch_concatenate_heads(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn.transformer.concatenate_heads(input_tensor, memory_config=output_memory_config)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [1024])
@pytest.mark.parametrize("num_heads", [4, 16])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_transformer_split_query_key_value_and_split_heads(
    batch_size, num_heads, sequence_size, head_size, input_dtype, input_memory_config, *, device
):
    torch.manual_seed(0)

    input_shape = (batch_size, sequence_size, num_heads * head_size * 3)
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)
    (
        torch_query_tensor,
        torch_key_tensor,
        torch_value_tensor,
    ) = ttnn.transformer._torch_split_query_key_value_and_split_heads(torch_input_tensor, num_heads=num_heads)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    query_tensor, key_tensor, value_tensor = ttnn.transformer.split_query_key_value_and_split_heads(
        input_tensor, num_heads=num_heads
    )
    query_tensor = ttnn.from_device(query_tensor)
    query_tensor = ttnn.to_torch(query_tensor)
    key_tensor = ttnn.from_device(key_tensor)
    key_tensor = ttnn.to_torch(key_tensor)
    value_tensor = ttnn.from_device(value_tensor)
    value_tensor = ttnn.to_torch(value_tensor)

    assert_with_pcc(torch_query_tensor, query_tensor, 0.999)
    assert_with_pcc(torch_key_tensor, key_tensor, 0.999)
    assert_with_pcc(torch_value_tensor, value_tensor, 0.999)
