# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import ttnn

from models.common.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "batch_size, sentence_size, hidden_embedding_dim, vocabulary_size, input_shard_shape, num_cores",
    [
        (8, 32, 128, 1024, (4, 16), 4),
        (4, 64, 256, 512, (2, 32), 4),
    ],
)
def test_embedding_dram_sharded_output(
    device,
    batch_size,
    sentence_size,
    hidden_embedding_dim,
    vocabulary_size,
    input_shard_shape,
    num_cores,
):
    """Test embedding with ND-sharded input and output stored in DRAM using height-sharded layout.
    DRAM sharding only supports cores with y=0 (single row); input can be ND-sharded on any grid."""
    torch.manual_seed(1234)

    # DRAM sharding only supports a single row of cores (y=0)
    num_cores_x = num_cores
    num_cores_y = 1

    input_shape = (batch_size, sentence_size)
    for dim in range(len(input_shape)):
        if input_shape[dim] % input_shard_shape[dim] != 0:
            pytest.skip(f"input_shape {input_shape} must be divisible by input_shard_shape {input_shard_shape}")

    num_input_shards = math.prod(input_shape) // math.prod(input_shard_shape)
    if num_input_shards != num_cores:
        pytest.skip(f"Number of input shards {num_input_shards} must equal num_cores {num_cores}")

    # Input ND sharding can use any grid that has num_cores; use same row (y=0) for consistency with DRAM
    input_shard_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))]
    )

    # ND-sharded input (L1)
    in_mem_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(
            input_shard_shape,
            input_shard_core_grid,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # Output: DRAM height-sharded
    fused_height = batch_size * sentence_size
    if fused_height % num_cores != 0:
        pytest.skip(f"Fused height {fused_height} must be divisible by num_cores {num_cores}")

    width = hidden_embedding_dim
    out_shard_shape = (fused_height // num_cores, width)
    output_shard_spec = ttnn.ShardSpec(input_shard_core_grid, out_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.DRAM,
        output_shard_spec,
    )

    torch_input_tensor = torch.randint(0, vocabulary_size - 1, input_shape)
    torch_weights = torch_random((vocabulary_size, hidden_embedding_dim), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.embedding(torch_input_tensor, torch_weights)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=in_mem_config,
    )
    weights = ttnn.as_tensor(
        torch_weights,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.embedding(
        input_tensor,
        weights,
        embeddings_type=ttnn.EmbeddingsType.GENERIC,
        dtype=ttnn.bfloat16,
        memory_config=output_mem_config,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
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
@pytest.mark.parametrize(
    "input_layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "output_layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize("tensors_nd_sharded", ["input_only", "output_only", "input_and_output"])
def test_nd_sharded_embedding(
    device,
    input_shape,
    input_shard_shape,
    shard_orientation,
    shard_core_grid,
    input_layout,
    output_layout,
    tensors_nd_sharded,
):
    if output_layout == ttnn.TILE_LAYOUT:
        pytest.skip("Tile layout is not supported for ND-sharded tensors")

    if input_layout == ttnn.TILE_LAYOUT and (
        input_shard_shape[-1] % ttnn.TILE_SIZE != 0 or input_shard_shape[-2] % ttnn.TILE_SIZE != 0
    ):
        pytest.skip("input shard shape last two dimensions must be divisible by ttnn.TILE_SIZE")

    torch.manual_seed(1234)

    compute_grid_size = device.compute_with_storage_grid_size()
    compute_grid = ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1)
    )
    if not compute_grid.contains(shard_core_grid):
        pytest.skip(f"Need {shard_core_grid} grid size to run this test but core grid is {compute_grid}")

    vocabulary_size = 32
    # The small hidden dimension allows testing multiple sizes without exceeding L1 capacity.
    hidden_embedding_dim = 16

    weights_shape = (vocabulary_size, hidden_embedding_dim)
    output_shape = input_shape + (weights_shape[-1],)

    out_shard_shape = input_shard_shape + (hidden_embedding_dim,)  # don't shard width for output tensor yet

    numel = int(math.prod(input_shape))
    torch_input_tensor = (torch.arange(numel, dtype=torch.long) % weights_shape[0]).reshape(input_shape)

    numel = int(math.prod(weights_shape))
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

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=input_layout, memory_config=in_mem_config)

    weights = ttnn.as_tensor(
        torch_weights,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.embedding(input_tensor, weights, layout=output_layout, memory_config=output_mem_config)
    output_tensor.deallocate(True)  # For some shapes, there isn't enough L1 memory to store the output tensors.
    output_tensor = ttnn.embedding(input_tensor, weights, layout=output_layout, memory_config=output_mem_config)

    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == output_shape
    assert_with_pcc(output_tensor, torch_output_tensor)
    assert device.num_program_cache_entries() == 1


@pytest.mark.parametrize(
    "input_shape, input_shard_shape",
    [
        ((199,), (1,)),  # 1d: 199 pages (prime, exceeds any device core count)
        ((10, 20), (2, 1)),  # 2d: 200 pages
        ((5, 6, 7), (1, 2, 1)),  # 3d: 210 pages
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),  # single core
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))]),  # 4x4 grid
    ],
)
@pytest.mark.parametrize("tensors_nd_sharded", ["input_only", "input_and_output"])
def test_nd_sharded_embedding_uneven_work_split(
    device,
    input_shape,
    input_shard_shape,
    shard_core_grid,
    tensors_nd_sharded,
):
    """Test ND-sharded embedding where num_input_pages is not evenly divisible by the
    device compute grid, exercising the num_pages_per_core_group_2 code path in
    embeddings_nd_sharded_program_factory.cpp (split_work_to_cores uneven division)."""
    torch.manual_seed(1234)

    compute_grid_size = device.compute_with_storage_grid_size()
    max_compute_cores = compute_grid_size.x * compute_grid_size.y
    index_elems_per_page = input_shard_shape[-1]
    num_input_pages = int(math.prod(input_shape)) // index_elems_per_page
    if num_input_pages <= max_compute_cores or num_input_pages % max_compute_cores == 0:
        pytest.skip(
            f"num_input_pages ({num_input_pages}) does not exercise core_group_2 "
            f"on this device with {max_compute_cores} compute cores"
        )

    vocabulary_size = 32
    hidden_embedding_dim = 16
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR

    weights_shape = (vocabulary_size, hidden_embedding_dim)
    output_shape = input_shape + (weights_shape[-1],)
    out_shard_shape = input_shard_shape + (hidden_embedding_dim,)

    numel = int(math.prod(input_shape))
    torch_input_tensor = (torch.arange(numel, dtype=torch.long) % weights_shape[0]).reshape(input_shape)

    numel = int(math.prod(weights_shape))
    torch_weights = torch.arange(numel, dtype=torch.int16).reshape(weights_shape).to(torch.bfloat16)

    torch_output_tensor = torch.nn.functional.embedding(torch_input_tensor, torch_weights)

    if tensors_nd_sharded in ("input_only", "input_and_output"):
        in_mem_config = ttnn.MemoryConfig(
            buffer_type=ttnn.BufferType.L1,
            nd_shard_spec=ttnn.NdShardSpec(input_shard_shape, shard_core_grid, shard_orientation),
        )
    else:
        in_mem_config = ttnn.DRAM_MEMORY_CONFIG

    if tensors_nd_sharded in ("output_only", "input_and_output"):
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

    output_tensor = ttnn.embedding(input_tensor, weights, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=output_mem_config)
    output_tensor.deallocate(True)
    output_tensor = ttnn.embedding(input_tensor, weights, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=output_mem_config)

    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == output_shape
    assert_with_pcc(output_tensor, torch_output_tensor)
    assert device.num_program_cache_entries() == 1
