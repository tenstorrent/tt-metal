# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
import math
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
@pytest.mark.parametrize("use_pack_untilize", [True, False])
@pytest.mark.parametrize(
    "tensor_shape, output_end",
    [
        ([3, 64, 64], [1, 50, 62]),
        ([3, 64, 64], [1, 29, 62]),
        ([5, 64, 64], [2, 50, 50]),
        ([4, 5, 64, 64], [1, 2, 50, 50]),
        ([3, 64, 64], [1, 31, 62]),
        ([1, 64, 64], [0, 63, 63]),
        ([2, 256, 512], [0, 255, 511]),
        ([4, 4, 256, 512], [0, 3, 255, 511]),
        ([4, 4, 256, 512], [2, 1, 255, 511]),
        ([4, 4, 256, 512], [2, 1, 126, 255]),
        ([4, 3, 64, 64], [2, 0, 31, 31]),
        ([4, 4, 3, 64, 64], [2, 3, 0, 31, 31]),
    ],
)
@pytest.mark.parametrize(
    "input_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 2))}),
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
            }
        ),
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

    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    try:
        input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, spec=tensor_spec, device=device)
    except Exception as e:
        pytest.xfail(f"from_torch failed while building sharded tensor: {e}")

    ttnn_output_tensor = ttnn.untilize_with_unpadding(
        input_ttnn_tensor,
        output_tensor_end=output_end,
        memory_config=output_memory_config,
        use_multicore=True,
        use_pack_untilize=use_pack_untilize,
    )
    # In untilize_with_unpadding, if the tensor has rank > 4, it ignores the output_end parameter
    if len(tensor_shape) > 4:
        assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))
    else:
        slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
        assert_equal(input_torch_tensor[slices], ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True, False])
@pytest.mark.parametrize(
    "tensor_shape, shard_shape, output_end",
    [
        ([3, 128, 160], ttnn.Shape([2, 64, 64]), [1, 20, 100]),
        ([3, 160, 160], ttnn.Shape([2, 64, 64]), [1, 79, 100]),
        ([3, 192, 160], ttnn.Shape([2, 64, 64]), [1, 50, 62]),
        ([3, 192, 128], ttnn.Shape([2, 64, 64]), [1, 50, 62]),
        ([4, 128, 160], ttnn.Shape([3, 96, 96]), [2, 50, 30]),
        ([2, 4, 128, 160], ttnn.Shape([2, 3, 96, 96]), [1, 2, 50, 100]),
        ([3, 160, 160], ttnn.Shape([3, 96, 96]), [1, 100, 0]),
    ],
)
@pytest.mark.parametrize(
    "input_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 2))}),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))}),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 2))}),
    ],
)
def test_untilize_with_unpadding_multi_core_nd_shard_to_interleaved_uneven_input_shard_spec(
    device,
    dtype,
    use_pack_untilize,
    tensor_shape,
    shard_shape,
    output_end,
    input_shard_orientation,
    shard_core_grid,
):
    torch.manual_seed(0)
    tensor_spec = ttnn.TensorSpec(
        shape=tensor_shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1
    ).sharded(shard_shape, shard_core_grid, orientation=input_shard_orientation)

    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, spec=tensor_spec, device=device)

    ttnn_output_tensor = ttnn.untilize_with_unpadding(
        input_ttnn_tensor,
        output_tensor_end=output_end,
        memory_config=output_memory_config,
        use_multicore=True,
        use_pack_untilize=use_pack_untilize,
    )

    if len(tensor_shape) > 4:
        assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))
    else:
        slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
        assert_equal(input_torch_tensor[slices], ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True, False])
@pytest.mark.parametrize("tensor_shape", [[8, 256, 256]])
@pytest.mark.parametrize(
    "input_shard_shape",
    [
        ttnn.Shape([3, 96, 96]),
    ],
)
@pytest.mark.parametrize(
    "output_shard_shape, output_end",
    [
        (ttnn.Shape([2, 64, 64]), [3, 127, 127]),
        (ttnn.Shape([2, 96, 96]), [2, 127, 127]),  # The following tests are for output unevenly sharded case
        (ttnn.Shape([5, 96, 96]), [3, 127, 127]),
        (ttnn.Shape([3, 20, 40]), [3, 127, 127]),
        (ttnn.Shape([5, 20, 40]), [3, 127, 127]),
    ],
)
@pytest.mark.parametrize(
    "input_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 2))}),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))}),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 2))}),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
    ],
)
def test_untilize_with_unpadding_multicore_nd_shard_to_nd_shard_spec_different_shard_specs(
    device,
    dtype,
    use_pack_untilize,
    tensor_shape,
    input_shard_shape,
    output_shard_shape,
    output_end,
    input_shard_orientation,
    shard_core_grid,
):
    torch.manual_seed(0)
    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=input_shard_orientation
    )
    tensor_spec = ttnn.TensorSpec(
        shape=tensor_shape,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        nd_shard_spec=input_nd_shard_spec,
        buffer_type=ttnn.BufferType.L1,
    )

    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, spec=tensor_spec, device=device)

    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_shard_shape, grid=shard_core_grid, orientation=input_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)
    ttnn_output_tensor = ttnn.untilize_with_unpadding(
        input_ttnn_tensor,
        output_tensor_end=output_end,
        memory_config=output_memory_config,
        use_multicore=True,
        use_pack_untilize=use_pack_untilize,
    )

    if len(tensor_shape) > 4:
        assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))
    else:
        slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
        assert_equal(input_torch_tensor[slices], ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True, False])
@pytest.mark.parametrize("tensor_shape", [[4, 128, 128]])
@pytest.mark.parametrize(
    "input_shard_shape",
    [
        ttnn.Shape([3, 96, 96]),
    ],
)
@pytest.mark.parametrize(
    "output_shard_shape, output_end",
    [
        (ttnn.Shape([160, 40]), [2, 120, 100]),
    ],
)
@pytest.mark.parametrize(
    "input_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 5))}),
    ],
)
def test_untilize_with_unpadding_multicore_nd_shard_round_robin_input_to_grid_2d_output(
    device,
    dtype,
    use_pack_untilize,
    tensor_shape,
    input_shard_shape,
    output_shard_shape,
    output_end,
    input_shard_orientation,
    shard_core_grid,
):
    torch.manual_seed(0)
    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape,
        grid=shard_core_grid,
        orientation=input_shard_orientation,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    tensor_spec = ttnn.TensorSpec(
        shape=tensor_shape,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        nd_shard_spec=input_nd_shard_spec,
        buffer_type=ttnn.BufferType.L1,
    )

    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, spec=tensor_spec, device=device)

    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_shard_shape,
        grid=shard_core_grid,
        orientation=input_shard_orientation,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.GRID_2D,
    )
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)
    ttnn_output_tensor = ttnn.untilize_with_unpadding(
        input_ttnn_tensor,
        output_tensor_end=output_end,
        memory_config=output_memory_config,
        use_multicore=True,
        use_pack_untilize=use_pack_untilize,
    )

    if len(tensor_shape) > 4:
        assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))
    else:
        slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
        assert_equal(input_torch_tensor[slices], ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True, False])
@pytest.mark.parametrize("tensor_shape", [[4, 192, 256]])
@pytest.mark.parametrize(
    "input_shard_shape",
    [
        ttnn.Shape([3, 64, 128]),
        ttnn.Shape([64, 128]),
        ttnn.Shape([2, 64, 128]),
        ttnn.Shape([1, 64, 128]),
    ],
)
@pytest.mark.parametrize(
    "output_shard_shape, output_end",
    [
        (ttnn.Shape([32, 128]), [1, 95, 127]),
        (ttnn.Shape([96, 128]), [1, 95, 127]),
        (ttnn.Shape([1, 96, 128]), [1, 95, 127]),
        (ttnn.Shape([2, 96, 128]), [1, 95, 127]),
        (ttnn.Shape([64, 128]), [1, 95, 127]),
    ],
)
@pytest.mark.parametrize(
    "input_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 2))}),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))}),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 2))}),
    ],
)
def test_untilize_with_unpadding_multicore_nd_shard_to_nd_shard_spec_different_shard_specs_shard_shape_flattened(
    device,
    dtype,
    use_pack_untilize,
    tensor_shape,
    input_shard_shape,
    output_shard_shape,
    output_end,
    input_shard_orientation,
    shard_core_grid,
):
    torch.manual_seed(0)
    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=input_shard_orientation
    )
    tensor_spec = ttnn.TensorSpec(
        shape=tensor_shape,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        nd_shard_spec=input_nd_shard_spec,
        buffer_type=ttnn.BufferType.L1,
    )

    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, spec=tensor_spec, device=device)

    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_shard_shape, grid=shard_core_grid, orientation=input_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)
    ttnn_output_tensor = ttnn.untilize_with_unpadding(
        input_ttnn_tensor,
        output_tensor_end=output_end,
        memory_config=output_memory_config,
        use_multicore=True,
        use_pack_untilize=use_pack_untilize,
    )

    if len(tensor_shape) > 4:
        assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))
    else:
        slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
        assert_equal(input_torch_tensor[slices], ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True, False])
@pytest.mark.parametrize("tensor_shape", [[8, 256, 256]])
@pytest.mark.parametrize(
    "input_shard_shape",
    [
        ttnn.Shape([3, 96, 96]),
    ],
)
@pytest.mark.parametrize("output_end", [(ttnn.Shape([3, 127, 127]))])
@pytest.mark.parametrize(
    "output_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize(
    "output_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "input_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "num_shard_cores, standard_shard_core_grid, block_shard_core_grid",
    [
        (
            4,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        )
    ],
)
def test_untilize_with_unpadding_multicore_nd_shard_to_legacy_shard(
    device,
    dtype,
    use_pack_untilize,
    tensor_shape,
    input_shard_shape,
    output_end,
    output_memory_layout,
    output_shard_orientation,
    input_shard_orientation,
    num_shard_cores,
    standard_shard_core_grid,
    block_shard_core_grid,
):
    torch.manual_seed(0)

    shard_core_grid = standard_shard_core_grid
    if output_memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        shard_core_grid = block_shard_core_grid
    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=input_shard_orientation
    )
    input_tensor_spec = ttnn.TensorSpec(
        shape=tensor_shape,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        nd_shard_spec=input_nd_shard_spec,
        buffer_type=ttnn.BufferType.L1,
    )

    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, spec=input_tensor_spec, device=device)

    output_tensor_shape = [dim + 1 for dim in output_end]
    num_tensor_dims = len(output_tensor_shape)
    output_tensor_height = 1
    for i in range(num_tensor_dims - 1):
        output_tensor_height *= output_tensor_shape[i]
    output_tensor_width = output_tensor_shape[num_tensor_dims - 1]

    # Shard shapes
    height_sharded_shard_shape = (output_tensor_height // num_shard_cores, output_tensor_width)
    width_sharded_shard_shape = (output_tensor_height, output_tensor_width // num_shard_cores)
    block_sharded_shard_shape = (
        output_tensor_height // int(math.sqrt(num_shard_cores)),
        output_tensor_width // int(math.sqrt(num_shard_cores)),
    )

    # Shard Memory Layout Map
    shard_memory_layout_map = {
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED: {
            "shard_grid": standard_shard_core_grid,
            "shard_shape": height_sharded_shard_shape,
        },
        ttnn.TensorMemoryLayout.WIDTH_SHARDED: {
            "shard_grid": standard_shard_core_grid,
            "shard_shape": width_sharded_shard_shape,
        },
        ttnn.TensorMemoryLayout.BLOCK_SHARDED: {
            "shard_grid": block_shard_core_grid,
            "shard_shape": block_sharded_shard_shape,
        },
    }

    # Output memory config
    output_shard_memory_layout = shard_memory_layout_map[output_memory_layout]
    output_shard_spec = ttnn.ShardSpec(
        output_shard_memory_layout["shard_grid"], output_shard_memory_layout["shard_shape"], output_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)
    ttnn_output_tensor = ttnn.untilize_with_unpadding(
        input_ttnn_tensor,
        output_tensor_end=output_end,
        memory_config=output_memory_config,
        use_multicore=True,
        use_pack_untilize=use_pack_untilize,
    )

    if len(tensor_shape) > 4:
        assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))
    else:
        slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
        assert_equal(input_torch_tensor[slices], ttnn.to_torch(ttnn_output_tensor))
