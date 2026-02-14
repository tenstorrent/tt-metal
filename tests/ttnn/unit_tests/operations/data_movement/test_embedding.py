# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import torch_random, run_for_wormhole_b0, skip_for_blackhole
import numpy as np


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


@pytest.mark.parametrize("batch_size", [55, 100, 200])
@pytest.mark.parametrize("sentence_size", [1, 10, 100])
@pytest.mark.parametrize("hidden_embedding_dim", [1, 128])  # Bert_Num_Cols_768, Llama_Num_Cols
@pytest.mark.parametrize(
    "vocabulary_size", [768]
)  # Bert_Position_Embeddings_512, Bert_Word_Embeddings_30528, Llama_Position_Embeddings,
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_embedding_unaligned_RM_pages(
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


@pytest.mark.parametrize("batch_size", [1, 8, 9])
@pytest.mark.parametrize("sentence_size", [32, 256, 512])
@pytest.mark.parametrize("hidden_embedding_dim", [768, 4096])  # Bert_Num_Cols_768, Llama_Num_Cols
@pytest.mark.parametrize(
    "vocabulary_size", [512, 30522, 2048]
)  # Bert_Position_Embeddings_512, Bert_Word_Embeddings_30528, Llama_Position_Embeddings,
@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("indices_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("weight_layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT])
def test_embedding_tiled_input(
    device,
    batch_size,
    sentence_size,
    hidden_embedding_dim,
    vocabulary_size,
    input_mem_config,
    output_mem_config,
    indices_layout,
    weight_layout,
    output_layout,
):
    torch.manual_seed(1234)

    torch_input_tensor = torch.randint(0, vocabulary_size - 1, (batch_size, sentence_size))
    torch_weights = torch_random((vocabulary_size, hidden_embedding_dim), -0.1, 0.1, dtype=torch.bfloat16)
    # torch_output_tensor = torch.nn.functional.embedding(torch_input_tensor, torch_weights)
    torch_embedding = torch.nn.Embedding.from_pretrained(torch_weights)
    torch_output_tensor = torch_embedding(torch_input_tensor)

    input_tensor = ttnn.to_device(
        ttnn.from_torch(torch_input_tensor, dtype=ttnn.uint32, layout=indices_layout),
        device,
        memory_config=input_mem_config,
    )
    weights = ttnn.to_device(
        ttnn.from_torch(torch_weights, dtype=ttnn.bfloat16, layout=weight_layout),
        device,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.embedding(
        input_tensor,
        weights,
        embeddings_type=ttnn.EmbeddingsType.GENERIC,  # Default embeddings type
        dtype=ttnn.bfloat16,
        memory_config=output_mem_config,  # Default memory config
        queue_id=0,  # Default queue id
        layout=output_layout,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


def reverse_embedding_output(output_tensor, weights_tensor):
    """
    Reverse the embedding operation by finding the indices of rows in the output tensor
    that match rows in the weights tensor.

    Args:
        output_tensor (torch.Tensor): The output tensor from the embedding operation.
        weights_tensor (torch.Tensor): The weights tensor used in the embedding operation.

    Returns:
        torch.Tensor: A tensor containing the indices corresponding to the rows in the output tensor.
    """
    reversed_indices = torch.empty(output_tensor.size(0), output_tensor.size(1), dtype=torch.long)

    for i in range(output_tensor.size(0)):  # Iterate over batch dimension
        for j in range(output_tensor.size(1)):  # Iterate over sentence dimension
            # Compare each row of the output tensor with the weights tensor
            matching_row = (weights_tensor == output_tensor[i, j]).all(dim=1)
            index = torch.where(matching_row)[0]
            if index.numel() > 0:
                reversed_indices[i, j] = index.item()  # Fill the matching index
            else:
                reversed_indices[i, j] = -1  # Use -1 if no matching row is found (edge case)

    return reversed_indices


def create_tile_tensor(height, width, tile_size):
    """
    Creates a 2D tensor where each element represents the tile it belongs to.

    Parameters:
        height (int): The height of the tensor (number of rows).
        width (int): The width of the tensor (number of columns).
        tile_size (int): The size of each square tile (tile_size x tile_size).

    Returns:
        torch.Tensor: A 2D tensor with tile indices.
    """
    # Calculate the number of tiles in each dimension
    tiles_per_row = (width + tile_size - 1) // tile_size
    tiles_per_col = (height + tile_size - 1) // tile_size

    # Create row and column indices
    row_indices = torch.arange(height).unsqueeze(1) // tile_size
    col_indices = torch.arange(width).unsqueeze(0) // tile_size

    # Calculate tile indices
    tile_tensor = row_indices * tiles_per_row + col_indices
    print(tile_tensor.shape)
    return tile_tensor


@pytest.mark.parametrize("batch_size", [1, 8, 9])
@pytest.mark.parametrize("sentence_size", [32, 256, 512])
@pytest.mark.parametrize("hidden_embedding_dim", [768, 4096])  # Bert_Num_Cols_768, Llama_Num_Cols
@pytest.mark.parametrize(
    "vocabulary_size", [512, 30522, 2048]
)  # Bert_Position_Embeddings_512, Bert_Word_Embeddings_30528, Llama_Position_Embeddings,
@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT])
def test_tiled(
    device,
    batch_size,
    sentence_size,
    hidden_embedding_dim,
    vocabulary_size,
    input_mem_config,
    output_mem_config,
    layout,
):
    torch.manual_seed(1234)

    torch_input_tensor = torch.randint(0, vocabulary_size - 1, (batch_size, sentence_size))
    torch_weights = torch_random((vocabulary_size, hidden_embedding_dim), -0.1, 0.1, dtype=torch.bfloat16)
    torch_embedding = torch.nn.Embedding.from_pretrained(torch_weights)
    torch_output_tensor = torch_embedding(torch_input_tensor)

    input_tensor = ttnn.to_device(
        ttnn.from_torch(torch_input_tensor, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT),
        device,
        memory_config=input_mem_config,
    )
    weights = ttnn.to_device(
        ttnn.from_torch(torch_weights, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        device,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.embedding(
        input_tensor,
        weights,
        embeddings_type=ttnn.EmbeddingsType.GENERIC,  # Default embeddings type
        dtype=ttnn.bfloat16,
        memory_config=output_mem_config,  # Default memory config
        queue_id=0,  # Default queue id
        layout=layout,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("batch_size, sentence_size, hidden_embedding_dim, vocabulary_size", [(10, 96, 2048, 128256)])
@pytest.mark.parametrize(
    "output_memory_layout, num_cores_x, num_cores_y",
    [
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, 8, 4),
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 6, 1),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, 4, 6),
    ],
)
@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_embedding_tiled_sharded_output(
    device,
    batch_size,
    sentence_size,
    hidden_embedding_dim,
    vocabulary_size,
    output_memory_layout,
    num_cores_x,
    num_cores_y,
    input_mem_config,
):
    torch.manual_seed(1234)
    layout = ttnn.TILE_LAYOUT

    output_shape = (batch_size, 1, sentence_size, hidden_embedding_dim)
    shard_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))]
    )
    fused_height = output_shape[0] * output_shape[1] * output_shape[2]
    width = output_shape[-1]
    if output_memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        shard_shape = (fused_height, width // (num_cores_x * num_cores_y))
    elif output_memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        shard_shape = (fused_height // (num_cores_x * num_cores_y), width)
    else:
        shard_shape = (fused_height // num_cores_y, width // num_cores_x)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(
        output_memory_layout,
        ttnn.BufferType.L1,
        shard_spec,
    )

    torch_input_tensor = torch.randint(0, vocabulary_size - 1, (batch_size, sentence_size))
    torch_weights = torch_random((vocabulary_size, hidden_embedding_dim), -0.1, 0.1, dtype=torch.bfloat16)
    torch_embedding = torch.nn.Embedding.from_pretrained(torch_weights)
    torch_output_tensor = torch_embedding(torch_input_tensor)

    input_tensor = ttnn.to_device(
        ttnn.from_torch(torch_input_tensor, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        device,
        memory_config=input_mem_config,
    )
    weights = ttnn.to_device(
        ttnn.from_torch(torch_weights, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        device,
        memory_config=input_mem_config,
    )

    output_tensor = ttnn.embedding(
        input_tensor,
        weights,
        embeddings_type=ttnn.EmbeddingsType.GENERIC,  # Default embeddings type
        dtype=ttnn.bfloat16,
        memory_config=output_mem_config,  # Default memory config
        queue_id=0,  # Default queue id
        layout=layout,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_tg_llama_sharded_embedding(
    device,
):
    torch.manual_seed(1234)
    unharvested_grid_size = (7, 10)
    compute_grid_size = device.compute_with_storage_grid_size()
    if unharvested_grid_size[0] > compute_grid_size.x or unharvested_grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {unharvested_grid_size} grid size to run this test but core grid is {compute_grid_size}")
    batch_size = 8
    vocabulary_size = 4096
    hidden_embedding_dim = 128
    token_padding = 31
    sentence_size = 1 + token_padding
    torch_input_tensor = torch.randint(0, vocabulary_size - 1, (batch_size, sentence_size))
    torch_weights = torch.randn(vocabulary_size, hidden_embedding_dim)
    torch_output_tensor = torch.nn.functional.embedding(torch_input_tensor, torch_weights)

    start_core = ttnn.CoreCoord(1, 0)
    core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    num_cores = batch_size
    shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(start_core, num_cores, core_grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        (batch_size * sentence_size // num_cores, hidden_embedding_dim),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )

    input_tensor = ttnn.as_tensor(
        torch_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    weights = ttnn.as_tensor(
        torch_weights,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.embedding(input_tensor, weights, layout=ttnn.TILE_LAYOUT, memory_config=output_mem_config)
    output_tensor = ttnn.reshape(
        output_tensor,
        ttnn.Shape((batch_size, 1, hidden_embedding_dim)),
        ttnn.Shape((batch_size, 32, hidden_embedding_dim)),
    )
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(output_tensor, torch_output_tensor[:, 0, :].unsqueeze(1))


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_tg_llama_sharded_rm_embedding(device):
    torch.manual_seed(1234)
    unharvested_grid_size = (7, 10)
    compute_grid_size = device.compute_with_storage_grid_size()
    if unharvested_grid_size[0] > compute_grid_size.x or unharvested_grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {unharvested_grid_size} grid size to run this test but core grid is {compute_grid_size}")
    batch_size = 16
    num_heads = 8
    vocabulary_size = 4096
    hidden_embedding_dim = 128
    torch_input_tensor = torch.randint(0, vocabulary_size - 1, (batch_size, num_heads))
    torch_weights = torch.randn(vocabulary_size, hidden_embedding_dim)
    torch_output_tensor = torch.nn.functional.embedding(torch_input_tensor, torch_weights)

    start_core = ttnn.CoreCoord(1, 0)
    core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    num_cores = batch_size
    shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(start_core, num_cores, core_grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        ((batch_size * num_heads) // num_cores, hidden_embedding_dim),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )

    input_tensor = ttnn.as_tensor(
        torch_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    weights = ttnn.as_tensor(
        torch_weights,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for i in range(2):
        output_tensor = ttnn.embedding(
            input_tensor, weights, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=output_mem_config
        )
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(output_tensor, torch_output_tensor)
    assert device.num_program_cache_entries() == 1


@pytest.mark.parametrize("hidden_embedding_dim", [16384])
@pytest.mark.parametrize("vocabulary_size", [32768])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
def test_embedding_oom(
    device,
    hidden_embedding_dim,
    vocabulary_size,
    dtype,
    input_mem_config,
    output_mem_config,
):
    torch.manual_seed(1234)

    torch_input_tensor = torch_random((1, 2048), 0.0, vocabulary_size - 1.0, dtype=torch.bfloat16)
    torch_int_input_tensor = torch_input_tensor.type(torch.int32)
    torch_weights = torch_random((vocabulary_size, hidden_embedding_dim), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.embedding(torch_int_input_tensor, torch_weights)

    input_tensor = ttnn.to_device(ttnn.from_torch(torch_input_tensor), device, memory_config=input_mem_config)
    weights = ttnn.to_device(ttnn.from_torch(torch_weights, dtype=dtype), device, memory_config=input_mem_config)

    output_tensor = ttnn.embedding(input_tensor, weights, memory_config=output_mem_config, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shape, input_shard_shape",
    [
        ((2, 2), (1, 1)),  # smallest possible shape
        ((4,), (2,)),  # 1d
        ((4, 8), (2, 2)),  # 2d small shard shape to test alignment
        ((4, 64), (2, 32)),  # 2d
        ((8, 64, 64), (4, 32, 32)),  # 3d
        ((4, 2, 64, 64), (2, 2, 32, 32)),  # 4d
        ((4, 4, 8, 8, 8), (2, 2, 4, 4, 4)),  # 5d
        ((8, 32, 32), (8, 16)),  # shard_shape dim < input_shape dim
        ((8, 24, 24), (8, 8)),  # uneven num of shards
        ((20, 20), (10, 10)),  # non-power of 2 shapes
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),  # single core
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))]),  # row
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]),  # col
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 3))]),  # row with offset
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 2))]),  # uneven num of cores
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))]),  # grid
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                ttnn.CoreRange(ttnn.CoreCoord(2, 1), ttnn.CoreCoord(3, 2)),
            ]
        ),  # non-contiguous row
    ],
)
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("tensors_nd_sharded", ["input_only", "output_only", "input_and_output"])
def test_nd_sharded_embedding(
    device,
    input_shape,
    input_shard_shape,
    shard_orientation,
    shard_core_grid,
    output_layout,
    tensors_nd_sharded,
):
    if output_layout == ttnn.TILE_LAYOUT:
        pytest.skip("Tile layout is not supported for ND-sharded tensors")

    torch.manual_seed(1234)

    compute_grid_size = device.compute_with_storage_grid_size()
    compute_grid = ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1)
    )
    if not compute_grid.contains(shard_core_grid):
        pytest.skip(f"Need {shard_core_grid} grid size to run this test but core grid is {compute_grid}")

    vocabulary_size = 32
    # We can test real word scenarios in separate test, but this one is for testing ND tensors
    hidden_embedding_dim = 16

    weights_shape = (vocabulary_size, hidden_embedding_dim)
    output_shape = input_shape + (weights_shape[-1],)

    out_shard_shape = input_shard_shape + (hidden_embedding_dim,)  # don't shard width for output tensor yet

    numel = int(np.prod(input_shape))
    torch_input_tensor = (torch.arange(numel, dtype=torch.long) % weights_shape[0]).reshape(input_shape)

    numel = int(np.prod(weights_shape))
    torch_weights = torch.arange(numel, dtype=torch.int16).reshape(weights_shape).to(torch.bfloat16)

    torch_output_tensor = torch.nn.functional.embedding(torch_input_tensor, torch_weights)

    if tensors_nd_sharded == "input_only" or tensors_nd_sharded == "input_and_output":
        in_mem_config = ttnn.MemoryConfig(
            buffer_type=ttnn.BufferType.L1,
            nd_shard_spec=ttnn.NdShardSpec(input_shard_shape, shard_core_grid, shard_orientation),
        )
    else:
        in_mem_config = ttnn.DRAM_MEMORY_CONFIG

    if tensors_nd_sharded == "output_only" or tensors_nd_sharded == "input_and_output":
        output_mem_config = ttnn.MemoryConfig(
            buffer_type=ttnn.BufferType.L1,
            nd_shard_spec=ttnn.NdShardSpec(
                out_shard_shape,
                shard_core_grid,
                shard_orientation,
            ),
        )
    else:
        output_mem_config = ttnn.DRAM_MEMORY_CONFIG

    input_tensor = ttnn.from_torch(
        torch_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=in_mem_config
    )

    weights = ttnn.as_tensor(
        torch_weights,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.embedding(input_tensor, weights, layout=output_layout, memory_config=output_mem_config)
    output_tensor.deallocate(True)  # For some shapes, there isn’t enough L1 memory to store the output tensors.
    output_tensor = ttnn.embedding(input_tensor, weights, layout=output_layout, memory_config=output_mem_config)

    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == output_shape
    assert_with_pcc(output_tensor, torch_output_tensor)
    assert device.num_program_cache_entries() == 1
