# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random, skip_for_wormhole_b0


def test_base_case(device):
    torch.manual_seed(1234)
    indices = ttnn.to_device(ttnn.from_torch(torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]]), dtype=ttnn.uint32), device)
    embedding_matrix = ttnn.to_device(ttnn.from_torch(torch.rand(10, 2), dtype=ttnn.bfloat16), device)
    indices_torch = ttnn.to_torch(ttnn.from_device(indices))
    embedding_matrix_torch = ttnn.to_torch(ttnn.from_device(embedding_matrix))
    expected_embeddings = torch.nn.functional.embedding(indices_torch, embedding_matrix_torch)
    embeddings = ttnn.embedding(indices, embedding_matrix)
    assert tuple(expected_embeddings.shape) == tuple(embeddings.shape)
    embeddings = ttnn.to_torch(ttnn.from_device(embeddings))
    assert_with_pcc(expected_embeddings, embeddings)


@pytest.mark.parametrize("batch_size", [1, 8, 9])
@pytest.mark.parametrize("sentence_size", [32, 256, 512])
@pytest.mark.parametrize("hidden_embedding_dim", [768, 4096])  # Bert_Num_Cols_768, Llama_Num_Cols
@pytest.mark.parametrize(
    "vocabulary_size", [512, 30522, 2048]
)  # Bert_Position_Embeddings_512, Bert_Word_Embeddings_30528, Llama_Position_Embeddings,
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_embedding(
    device,
    batch_size,
    sentence_size,
    hidden_embedding_dim,
    vocabulary_size,
    dtype,
    input_mem_config,
    output_mem_config,
    layout,
):
    torch.manual_seed(1234)

    torch_input_tensor = torch.randint(0, vocabulary_size - 1, (batch_size, sentence_size))
    torch_weights = torch_random((vocabulary_size, hidden_embedding_dim), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.embedding(torch_input_tensor, torch_weights)

    input_tensor = ttnn.to_device(ttnn.from_torch(torch_input_tensor), device, memory_config=input_mem_config)
    weights = ttnn.to_device(ttnn.from_torch(torch_weights, dtype=dtype), device, memory_config=input_mem_config)

    output_tensor = ttnn.embedding(input_tensor, weights, memory_config=output_mem_config, layout=layout)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sentence_size", [384])
@pytest.mark.parametrize("hidden_embedding_dim", [1024])
@pytest.mark.parametrize("vocabulary_size", [250880])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_bloom_embedding(
    device,
    batch_size,
    sentence_size,
    hidden_embedding_dim,
    vocabulary_size,
    dtype,
    input_mem_config,
    output_mem_config,
):
    torch.manual_seed(1234)

    torch_input_tensor = torch.randint(0, vocabulary_size - 1, (batch_size, sentence_size))
    torch_weights = torch_random((vocabulary_size, hidden_embedding_dim), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.embedding(torch_input_tensor, torch_weights)

    input_tensor = ttnn.to_device(ttnn.from_torch(torch_input_tensor), device, memory_config=input_mem_config)
    weights = ttnn.to_device(ttnn.from_torch(torch_weights, dtype=dtype), device, memory_config=input_mem_config)

    output_tensor = ttnn.embedding(input_tensor, weights, memory_config=output_mem_config, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("sentence_size", [32])
@pytest.mark.parametrize("hidden_embedding_dim", [4096])
@pytest.mark.parametrize("vocabulary_size", [32000])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_moe_embedding(
    device,
    batch_size,
    sentence_size,
    hidden_embedding_dim,
    vocabulary_size,
    dtype,
    input_mem_config,
    output_mem_config,
):
    torch.manual_seed(1234)

    torch_input_tensor = torch_random((batch_size, sentence_size), 0.0, vocabulary_size - 1.0, dtype=torch.bfloat16)
    torch_int_input_tensor = torch_input_tensor.type(torch.int32)
    torch_weights = torch_random((vocabulary_size, hidden_embedding_dim), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.embedding(torch_int_input_tensor, torch_weights)

    input_tensor = ttnn.to_device(ttnn.from_torch(torch_input_tensor), device, memory_config=input_mem_config)
    weights = ttnn.to_device(ttnn.from_torch(torch_weights, dtype=dtype), device, memory_config=input_mem_config)

    output_tensor = ttnn.embedding(input_tensor, weights, memory_config=output_mem_config, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)
