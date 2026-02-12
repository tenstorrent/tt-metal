# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_equal

TTNN_TO_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize(
    "shape_output_end",
    [
        ([2, 2], [1, 1]),
        ([1, 1, 2, 2], [0, 0, 1, 1]),
        ([1, 1, 32, 32], [0, 0, 31, 31]),
        ([1, 1, 128, 256], [0, 0, 127, 255]),
        ([1, 32, 32, 128], [0, 31, 31, 127]),
        # Need sfpu untilize for fp32 #30400, #33795
        # ([1, 1, 128, 7328], [0, 0, 119, 7299]),
        # ([4128, 512], [4127, 511]),
    ],
)
@pytest.mark.parametrize("input_buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
@pytest.mark.parametrize("output_buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
def test_untilize_with_unpadding_fp32(device, dtype, shape_output_end, input_buffer_type, output_buffer_type):
    torch.manual_seed(42)
    shape, output_end = shape_output_end
    torch_tensor = torch.rand(shape, dtype=TTNN_TO_TORCH_DTYPE[dtype])

    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, input_buffer_type)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, output_buffer_type)
    tile_tensor = ttnn.from_torch(
        torch_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )
    untilized = ttnn.untilize_with_unpadding(
        tile_tensor, output_tensor_end=output_end, memory_config=output_memory_config
    )
    result = ttnn.to_torch(untilized)

    # Slice from 0 to output_end[i]+1 for each dimension
    slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
    torch_result = torch_tensor[slices]

    assert torch.equal(result, torch_result), f"untilize_with_unpadding lost {dtype} precision"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "shape, output_end, shard_shape, num_cores",
    [
        # HEIGHT_SHARDED: shard along height dimension
        # Shape [1, 1, 256, 128] with 4 cores -> shard_shape [64, 128]
        ([1, 1, 256, 128], [0, 0, 255, 127], (64, 128), 4),
        # Unpadding case: output is smaller than input
        ([1, 1, 256, 128], [0, 0, 200, 100], (64, 128), 4),
        # 2 cores
        ([1, 1, 128, 64], [0, 0, 127, 63], (64, 64), 2),
        ([1, 1, 128, 64], [0, 0, 100, 50], (64, 64), 2),
    ],
)
@pytest.mark.parametrize("output_sharded", [True, False])
def test_untilize_with_unpadding_height_sharded(
    device, dtype, shape, output_end, shard_shape, num_cores, output_sharded
):
    """Test untilize_with_unpadding with HEIGHT_SHARDED input.

    HEIGHT_SHARDED input can output to either HEIGHT_SHARDED or INTERLEAVED.
    """
    torch.manual_seed(42)
    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)

    # Create HEIGHT_SHARDED input memory config
    shard_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_cores - 1))})
    input_shard_spec = ttnn.ShardSpec(shard_core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
    )

    # Output memory config - either HEIGHT_SHARDED or INTERLEAVED
    if output_sharded:
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
        )
    else:
        output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Create input tensor on device with sharded memory config
    tile_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tile_tensor = ttnn.to_device(tile_tensor, device, memory_config=input_memory_config)

    untilized = ttnn.untilize_with_unpadding(
        tile_tensor, output_tensor_end=output_end, memory_config=output_memory_config
    )
    result = ttnn.to_torch(untilized)

    # Compute expected result
    slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
    torch_result = torch_tensor[slices]

    assert_equal(result, torch_result)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "shape, output_end, shard_shape, num_cores",
    [
        # WIDTH_SHARDED: shard along width dimension
        # Shape [1, 1, 64, 256] with 4 cores -> shard_shape [64, 64]
        ([1, 1, 64, 256], [0, 0, 63, 255], (64, 64), 4),
        # Unpadding case: output width is smaller
        ([1, 1, 64, 256], [0, 0, 63, 200], (64, 64), 4),
        # 2 cores
        ([1, 1, 32, 128], [0, 0, 31, 127], (32, 64), 2),
        ([1, 1, 32, 128], [0, 0, 31, 100], (32, 64), 2),
    ],
)
@pytest.mark.parametrize("output_sharded", [True, False])
def test_untilize_with_unpadding_width_sharded(
    device, dtype, shape, output_end, shard_shape, num_cores, output_sharded
):
    """Test untilize_with_unpadding with WIDTH_SHARDED input.

    WIDTH_SHARDED input can output to either WIDTH_SHARDED or INTERLEAVED (unbatched only for interleaved).
    """
    torch.manual_seed(42)
    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)

    # Create WIDTH_SHARDED input memory config
    shard_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_cores - 1))})
    input_shard_spec = ttnn.ShardSpec(shard_core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    # Output memory config - either WIDTH_SHARDED or INTERLEAVED
    if output_sharded:
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec
        )
    else:
        output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Create input tensor on device with sharded memory config
    tile_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tile_tensor = ttnn.to_device(tile_tensor, device, memory_config=input_memory_config)

    untilized = ttnn.untilize_with_unpadding(
        tile_tensor, output_tensor_end=output_end, memory_config=output_memory_config
    )
    result = ttnn.to_torch(untilized)

    # Compute expected result
    slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
    torch_result = torch_tensor[slices]

    assert_equal(result, torch_result)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "shape, output_end, shard_shape, grid_size",
    [
        # BLOCK_SHARDED: shard along both height and width
        # Shape [1, 1, 128, 128] with 2x2 grid -> shard_shape [64, 64]
        ([1, 1, 128, 128], [0, 0, 127, 127], (64, 64), (2, 2)),
        # Unpadding case
        ([1, 1, 128, 128], [0, 0, 100, 100], (64, 64), (2, 2)),
        # 4x4 grid
        ([1, 1, 256, 256], [0, 0, 255, 255], (64, 64), (4, 4)),
        ([1, 1, 256, 256], [0, 0, 200, 200], (64, 64), (4, 4)),
    ],
)
def test_untilize_with_unpadding_block_sharded(device, dtype, shape, output_end, shard_shape, grid_size):
    """Test untilize_with_unpadding with BLOCK_SHARDED input.

    BLOCK_SHARDED input must output to INTERLEAVED and input must be unbatched.
    """
    torch.manual_seed(42)
    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)

    # Create BLOCK_SHARDED input memory config
    shard_core_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1))}
    )
    input_shard_spec = ttnn.ShardSpec(shard_core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    # Output must be INTERLEAVED for BLOCK_SHARDED input
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Create input tensor on device with sharded memory config
    tile_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tile_tensor = ttnn.to_device(tile_tensor, device, memory_config=input_memory_config)

    untilized = ttnn.untilize_with_unpadding(
        tile_tensor, output_tensor_end=output_end, memory_config=output_memory_config
    )
    result = ttnn.to_torch(untilized)

    # Compute expected result
    slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
    torch_result = torch_tensor[slices]

    assert_equal(result, torch_result)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True])  # , False])
@pytest.mark.parametrize(
    "tensor_shape, output_end",
    [
        ([3, 64, 64], [1, 50, 62]),
        ([3, 64, 64], [1, 31, 62]),
        # ([4, 4, 3, 64, 64], [2, 3, 0, 31, 31]),
        #   ([4, 3, 64, 64], [2, 0, 31, 31]),
        # ([1, 64, 64], [0, 63, s63]),
        # ([2, 32, 32], [0, 31, 31]),
        # ([2, 256, 512], [0, 255, 511]),
        # ([4, 4, 256, 512], [0, 3, 255, 511]),
    ],
)
@pytest.mark.parametrize(
    "input_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        # ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        # ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 2))}),
        # ttnn.CoreRangeSet(
        #     {
        #         ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
        #         ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
        #     }
        # ),
    ],
)
def test_untilize_with_unpadding_multi_core_nd_sharded_to_interleaved(
    device,
    dtype,
    use_pack_untilize,
    tensor_shape,
    output_end,
    input_shard_orientation,
    shard_core_grid,
):
    torch.manual_seed(0)
    shard_dims = list(range(len(tensor_shape) - 2, len(tensor_shape)))
    tensor_spec = ttnn.TensorSpec(
        shape=tensor_shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1
    ).sharded_across_dims(shard_dims, shard_core_grid, input_shard_orientation)
    nd_shard_spec = tensor_spec.memory_config.nd_shard_spec
    assert nd_shard_spec is not None
    print(f"nd_shard_spec.shard_shape: {nd_shard_spec.shard_shape}")

    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    try:
        input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, spec=tensor_spec, device=device)
    except Exception as e:
        pytest.xfail(f"from_torch failed while building sharded tensor: {e}")

    # slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))

    # print(input_ttnn_tensor[slices].shape)

    # output_end = [dim - 1 for dim in tensor_shape]
    ttnn_output_tensor = ttnn.untilize_with_unpadding(
        input_ttnn_tensor,
        output_tensor_end=output_end,
        memory_config=output_memory_config,
        use_multicore=True,
        use_pack_untilize=use_pack_untilize,
    )
    slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))

    assert_equal(input_torch_tensor[slices], ttnn.to_torch(ttnn_output_tensor))
