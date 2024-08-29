# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random, is_blackhole, is_wormhole_b0


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
    golden_function = ttnn.get_golden_function(ttnn.transformer.attention_softmax)
    torch_output_tensor = golden_function(
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

    assert_with_pcc(torch_output_tensor, output_tensor, 0.992)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("sequence_size", [384, 1024])
@pytest.mark.parametrize("target_sequence_size", [384, 4096])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_transformer_attention_softmax_(
    batch_size,
    num_heads,
    sequence_size,
    target_sequence_size,
    input_dtype,
    input_memory_config,
    *,
    device,
):
    torch.manual_seed(0)

    input_shape = (batch_size, num_heads, sequence_size, target_sequence_size)
    torch_input_tensor = torch_random(input_shape, 0, 1.0, dtype=torch.bfloat16)
    torch_attention_mask = torch_random(
        (batch_size, 1, sequence_size, target_sequence_size), 0, 1.0, dtype=torch.bfloat16
    )

    golden_function = ttnn.get_golden_function(ttnn.transformer.attention_softmax_)
    torch_output_tensor = golden_function(
        torch_input_tensor,
        head_size=None,
        attention_mask=torch_attention_mask,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )
    attention_mask = ttnn.from_torch(
        torch_attention_mask,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn.transformer.attention_softmax_(
        input_tensor, head_size=None, attention_mask=attention_mask, causal_mask=True
    )
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.996)


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
    golden_function = ttnn.get_golden_function(ttnn.transformer.concatenate_heads)
    torch_output_tensor = golden_function(torch_input_tensor)

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
@pytest.mark.parametrize("num_kv_heads", [None])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_transformer_split_query_key_value_and_split_heads(
    batch_size, num_heads, sequence_size, head_size, num_kv_heads, input_dtype, input_memory_config, *, device
):
    torch.manual_seed(0)

    if num_kv_heads is not None:
        input_shape = (batch_size, sequence_size, (num_heads + num_kv_heads * 2) * head_size)
    else:
        input_shape = (batch_size, sequence_size, num_heads * 3 * head_size)
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.transformer.split_query_key_value_and_split_heads)
    (
        torch_query_tensor,
        torch_key_tensor,
        torch_value_tensor,
    ) = golden_function(torch_input_tensor, num_heads=num_heads, num_kv_heads=num_kv_heads)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    query_tensor, key_tensor, value_tensor = ttnn.transformer.split_query_key_value_and_split_heads(
        input_tensor, num_heads=num_heads, num_kv_heads=num_kv_heads
    )
    query_tensor = ttnn.to_torch(query_tensor)
    key_tensor = ttnn.to_torch(key_tensor)
    value_tensor = ttnn.to_torch(value_tensor)

    assert_with_pcc(torch_query_tensor, query_tensor, 0.999)
    assert_with_pcc(torch_key_tensor, key_tensor, 0.999)
    assert_with_pcc(torch_value_tensor, value_tensor, 0.999)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [1024])
@pytest.mark.parametrize("num_heads", [4, 16])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("num_kv_heads", [None])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_transformer_split_query_key_value_and_split_heads_with_kv_input_tensor(
    batch_size, num_heads, sequence_size, head_size, num_kv_heads, input_dtype, input_memory_config, *, device
):
    torch.manual_seed(0)

    input_shape = (batch_size, sequence_size, num_heads * head_size)
    kv_input_shape = (batch_size, sequence_size, num_heads * 2 * head_size)
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)
    torch_kv_input_tensor = torch_random(kv_input_shape, -0.1, 0.1, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.transformer.split_query_key_value_and_split_heads)

    (
        torch_query_tensor,
        torch_key_tensor,
        torch_value_tensor,
    ) = golden_function(torch_input_tensor, torch_kv_input_tensor, num_heads=num_heads, num_kv_heads=num_kv_heads)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )
    kv_input_tensor = ttnn.from_torch(
        torch_kv_input_tensor,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    query_tensor, key_tensor, value_tensor = ttnn.transformer.split_query_key_value_and_split_heads(
        input_tensor, kv_input_tensor, num_heads=num_heads, num_kv_heads=num_kv_heads
    )
    query_tensor = ttnn.to_torch(query_tensor)
    key_tensor = ttnn.to_torch(key_tensor)
    value_tensor = ttnn.to_torch(value_tensor)

    assert_with_pcc(torch_query_tensor, query_tensor, 0.999)
    assert_with_pcc(torch_key_tensor, key_tensor, 0.999)
    assert_with_pcc(torch_value_tensor, value_tensor, 0.999)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("num_heads", [71])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_falcon_split_query_key_value_and_split_heads(
    batch_size, num_heads, sequence_size, head_size, num_kv_heads, input_dtype, input_memory_config, *, device
):
    torch.manual_seed(0)

    if num_kv_heads is not None:
        input_shape = (batch_size, sequence_size, (num_heads + num_kv_heads * 2) * head_size)
    else:
        input_shape = (batch_size, sequence_size, num_heads * 3 * head_size)
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.transformer.split_query_key_value_and_split_heads)
    (
        torch_query_tensor,
        torch_key_tensor,
        torch_value_tensor,
    ) = golden_function(torch_input_tensor, num_heads=num_heads, num_kv_heads=num_kv_heads, transpose_key=False)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    query_tensor, key_tensor, value_tensor = ttnn.transformer.split_query_key_value_and_split_heads(
        input_tensor, num_heads=num_heads, num_kv_heads=num_kv_heads, transpose_key=False
    )
    query_tensor = ttnn.to_torch(query_tensor)
    key_tensor = ttnn.to_torch(key_tensor)
    value_tensor = ttnn.to_torch(value_tensor)

    assert_with_pcc(torch_query_tensor, query_tensor, 0.999)
    assert_with_pcc(torch_key_tensor, key_tensor, 0.999)
    assert_with_pcc(torch_value_tensor, value_tensor, 0.999)


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [197])
@pytest.mark.parametrize("num_heads", [12])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_vit_split_query_key_value_and_split_heads(
    batch_size, num_heads, sequence_size, head_size, input_dtype, input_memory_config, *, device
):
    torch.manual_seed(0)

    input_shape = (batch_size, sequence_size, num_heads * 3 * head_size)
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.transformer.split_query_key_value_and_split_heads)
    (
        torch_query_tensor,
        torch_key_tensor,
        torch_value_tensor,
    ) = golden_function(torch_input_tensor, num_heads=num_heads)

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
    query_tensor = ttnn.to_torch(query_tensor)
    key_tensor = ttnn.to_torch(key_tensor)
    value_tensor = ttnn.to_torch(value_tensor)

    assert list(query_tensor.shape) == [batch_size, num_heads, sequence_size, head_size]
    assert list(key_tensor.shape) == [batch_size, num_heads, head_size, sequence_size]
    assert list(value_tensor.shape) == [batch_size, num_heads, sequence_size, head_size]

    assert ttnn.pearson_correlation_coefficient(torch_query_tensor, query_tensor) > 0.999
    assert ttnn.pearson_correlation_coefficient(torch_key_tensor, key_tensor) > 0.999
    assert ttnn.pearson_correlation_coefficient(torch_value_tensor, value_tensor) > 0.999


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("sequence_size", [224, 384])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize(
    "batch_size, num_heads, core_grid",
    (
        (8, 12, ttnn.CoreGrid(y=8, x=12)),
        (8, 16, ttnn.CoreGrid(y=8, x=8)),
    ),
)
def test_sharded_split_query_key_value_and_split_heads(
    batch_size, num_heads, sequence_size, head_size, input_dtype, core_grid, *, device
):
    torch.manual_seed(0)

    input_shape = (batch_size, sequence_size, num_heads * 3 * head_size)
    input_memory_config = ttnn.create_sharded_memory_config(
        input_shape, core_grid=core_grid, strategy=ttnn.ShardStrategy.BLOCK
    )

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.transformer.split_query_key_value_and_split_heads)
    (
        torch_query_tensor,
        torch_key_tensor,
        torch_value_tensor,
    ) = golden_function(torch_input_tensor, num_heads=num_heads)

    # Sharded inputs requires the groups of heads to be interleaved (ie. {q1, k1, v1}, {q2, k2, v2}, ..., {qn, kn, vn})
    (torch_q, torch_k, torch_v) = torch.split(
        torch_input_tensor, [num_heads * head_size, num_heads * head_size, num_heads * head_size], dim=-1
    )
    torch_input_tensor_interleaved = torch.concat(
        [
            torch_q.reshape(batch_size, sequence_size, num_heads, head_size),
            torch_k.reshape(batch_size, sequence_size, num_heads, head_size),
            torch_v.reshape(batch_size, sequence_size, num_heads, head_size),
        ],
        dim=-1,
    ).reshape(input_shape)
    input_tensor = ttnn.from_torch(
        torch_input_tensor_interleaved,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    query_tensor, key_tensor, value_tensor = ttnn.transformer.split_query_key_value_and_split_heads(
        input_tensor, num_heads=num_heads, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
    )
    query_tensor = ttnn.to_torch(query_tensor)
    key_tensor = ttnn.to_torch(key_tensor)
    value_tensor = ttnn.to_torch(value_tensor)

    assert_with_pcc(torch_query_tensor, query_tensor, 0.999)
    assert_with_pcc(torch_key_tensor, key_tensor, 0.999)
    assert_with_pcc(torch_value_tensor, value_tensor, 0.999)


def test_split_query_key_value_and_split_heads_when_head_size_is_not_a_multiple_of_32(device):
    """
    This test is to check that the split_query_key_value_and_split_heads function raises an error when the head size is not a multiple of 32
    And then it shows what user could do to fix the error
    In the real scenario, the user would have to update the matmul that produces the input tensor for this operation to have the padding in the weights
    """

    torch.manual_seed(0)

    batch_size = 2
    sequence_size = 1024
    num_heads = 8
    head_size = 80
    padded_head_size = 96  # Head size padded to tile size
    input_dtype = ttnn.bfloat16
    input_memory_config = ttnn.DRAM_MEMORY_CONFIG

    input_shape = (batch_size, sequence_size, num_heads * 3 * head_size)
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.transformer.split_query_key_value_and_split_heads)
    (
        torch_query_tensor,
        torch_key_tensor,
        torch_value_tensor,
    ) = golden_function(
        torch_input_tensor,
        num_heads=num_heads,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    with pytest.raises(RuntimeError) as e:
        query_tensor, key_tensor, value_tensor = ttnn.transformer.split_query_key_value_and_split_heads(
            input_tensor, num_heads=num_heads
        )
        assert (
            "Head size must be a multiple of 32! Update the preceding matmul to have the padding in the weights!"
            in str(e.value)
        )

    # Manually each head to a mutliple of 32
    input_tensor_heads = torch.split(torch_input_tensor, head_size, dim=-1)
    input_tensor_heads = [
        torch.nn.functional.pad(head, (0, padded_head_size - head_size), "constant", 0) for head in input_tensor_heads
    ]
    input_tensor = torch.cat(input_tensor_heads, dim=-1)

    input_tensor = ttnn.from_torch(
        input_tensor,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    query_tensor, key_tensor, value_tensor = ttnn.transformer.split_query_key_value_and_split_heads(
        input_tensor, num_heads=num_heads
    )

    # Remove the padding
    query_tensor = ttnn.to_torch(query_tensor)[..., :head_size]
    key_tensor = ttnn.to_torch(key_tensor)[..., :head_size, :]
    value_tensor = ttnn.to_torch(value_tensor)[..., :head_size]

    assert_with_pcc(torch_query_tensor, query_tensor, 0.999)
    assert_with_pcc(torch_key_tensor, key_tensor, 0.999)
    assert_with_pcc(torch_value_tensor, value_tensor, 0.999)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.skip(reason="#9267: need to fix since it never ran in CI")
def test_concatenate_heads_when_head_size_is_not_a_multiple_of_32(device):
    """
    This test is to check that the concatenate_heads function raises an error when the head size is not a multiple of 32
    And then it shows what user could do to fix the error
    In the real scenario, the user would have to update the matmul that uses the output of this operation to have the padding in the weights
    """

    torch.manual_seed(0)

    batch_size = 2
    sequence_size = 1024
    num_heads = 8
    head_size = 80
    padded_head_size = 96  # Head size padded to tile size
    input_dtype = ttnn.bfloat16
    input_memory_config = ttnn.DRAM_MEMORY_CONFIG

    input_shape = (batch_size, num_heads, sequence_size, head_size)
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.transformer.concatenate_heads)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    with pytest.raises(RuntimeError) as e:
        output_tensor = ttnn.transformer.concatenate_heads(input_tensor)

    assert (
        "Head size must be a multiple of 32!  Update matmul that uses the output of this operation to have the padding in the weights!"
        in str(e.value)
    )

    input_tensor = torch.nn.functional.pad(torch_input_tensor, (0, padded_head_size - head_size), "constant", 0)
    input_tensor = ttnn.from_torch(
        input_tensor,
        device=device,
        dtype=input_dtype,
        memory_config=input_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )
    output_tensor = ttnn.transformer.concatenate_heads(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    # Remove the padding
    output_tensor = torch.cat(
        [chunk[..., :head_size] for chunk in torch.split(output_tensor, padded_head_size, dim=-1)], dim=-1
    )

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
