# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import math

import pytest
import torch
import ttnn

from models.common.utility_functions import is_slow_dispatch
from tests.ttnn.utils_for_testing import assert_equal

TTNN_TO_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


@pytest.fixture
def isolate_program_cache(device):
    """Ensure each test starts with an empty program cache and cleans up after."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


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


def test_untilize_with_unpadding_reuses_cache_when_only_width_l1_heuristic_changes(device, isolate_program_cache):
    """
    Regression test for issue #46533 first-shot fix.

    This shape keeps the height-side CB estimate tiny (1 tile row) while making the
    width-side estimate large enough to react to unrelated resident L1 buffers.
    enough_space_width is not consumed by factory selection or descriptor creation,
    so changing only that heuristic must not fork the program cache.
    """
    torch.manual_seed(0)

    input_shape = [1, 10240, 32]
    output_end = [0, 10239, 31]
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    assert device.num_program_cache_entries() == 0, "Program cache should be empty before the test"

    with device.cache_entries_counter.measure():
        output_tensor = ttnn.untilize_with_unpadding(
            input_tensor, output_tensor_end=output_end, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    assert_equal(ttnn.to_torch(output_tensor), torch_input)
    assert device.cache_entries_counter.total == 1

    ttnn.synchronize_device(device)

    hog_shape = [1, 1, 32, 16384]
    hog_tensors = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape(hog_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
        for _ in range(3)
    ]
    assert len(hog_tensors) == 3

    with device.cache_entries_counter.measure():
        output_tensor = ttnn.untilize_with_unpadding(
            input_tensor, output_tensor_end=output_end, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    assert_equal(ttnn.to_torch(output_tensor), torch_input)
    assert device.cache_entries_counter.total == 1


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


@pytest.mark.parametrize(
    "hw, out_channels",
    [
        (2048, 2),  # 4 bytes, below 16-byte NOC alignment
        (2048, 4),  # 8 bytes, below 16-byte NOC alignment
        (2048, 8),  # 16 bytes, at alignment boundary
        (2048, 16),  # W==16 fast path must not fire for interleaved output
    ],
)
def test_untilize_with_unpadding_height_sharded_narrow_width_regression(device, hw, out_channels):
    torch.manual_seed(0)
    torch_input = torch.rand((hw, out_channels), dtype=torch.bfloat16)

    num_cores = 64
    shard_shape = [hw // num_cores, max(out_channels, 32)]  # pad width to TILE_WIDTH
    core_range_set = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size())
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_range_set, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    output_tensor_end = [hw - 1, out_channels - 1]

    tt_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    if out_channels == 16:
        # Force an interleaved output so the W==16 fast path (sharded-only) must not fire.
        tt_out = ttnn.untilize_with_unpadding(
            tt_tensor, output_tensor_end=output_tensor_end, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        assert tt_out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    else:
        tt_out = ttnn.untilize_with_unpadding(tt_tensor, output_tensor_end=output_tensor_end)
    result = ttnn.to_torch(tt_out)

    assert_equal(result, torch_input)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("output_width", [30, 31, 32])
def test_untilize_with_unpadding_interleaved_to_height_sharded_unaligned_row(device, dtype, output_width):
    """INTERLEAVED input -> HEIGHT_SHARDED output whose row size is not a multiple of the
    buffer alignment (16 bytes in L1).
    """
    torch.manual_seed(0)
    shape = [1, 1, 256, 64]
    output_end = [0, 0, 255, output_width - 1]
    torch_input = torch.rand(shape, dtype=TTNN_TO_TORCH_DTYPE[dtype])

    # Shard width tracks the unpadded output width, so the output row size in bytes stays
    # unaligned - padding the shard out to a tile width would hide the mis-striding.
    num_cores = 4
    shard_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_cores - 1))})
    output_shard_spec = ttnn.ShardSpec(
        shard_core_grid, (shape[2] // num_cores, output_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
    )

    tile_tensor = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    untilized = ttnn.untilize_with_unpadding(
        tile_tensor, output_tensor_end=output_end, memory_config=output_memory_config
    )
    result = ttnn.to_torch(untilized)

    slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
    assert_equal(result, torch_input[slices])


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
@pytest.mark.parametrize(
    "tensor_shape, output_end",
    [
        ([2, 3, 4, 64, 64], [0, 1, 2, 31, 31]),
        ([2, 3, 4, 64, 64], [1, 2, 3, 63, 63]),
    ],
)
def test_untilize_with_unpadding_rank_gt_4_uses_output_tensor_end(device, dtype, tensor_shape, output_end):
    torch.manual_seed(0)
    torch_tensor = torch.arange(1, 1 + math.prod(tensor_shape), dtype=TTNN_TO_TORCH_DTYPE[dtype]).reshape(tensor_shape)

    tile_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    untilized = ttnn.untilize_with_unpadding(tile_tensor, output_tensor_end=output_end)
    result = ttnn.to_torch(untilized)

    expected_shape = [end + 1 for end in output_end]
    assert list(untilized.shape) == expected_shape
    slices = tuple(slice(0, end + 1) for end in output_end)
    assert_equal(result, torch_tensor[slices])


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
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
        # Blocked until reshape supports ND-sharded tensors without going through
        # ttnn::experimental::view. The rank>4 wrapper in untilize_with_unpadding
        # (build_ndiml_untilize_val -> squeeze_from_ND_to_4D -> ttnn::reshape) routes
        # through PerformView -> view_device, which rejects ND-sharded inputs for any
        # rank change other than 0D/1D -> 2D expansion. See:
        #   - ttnn/core/tensor/tensor_ops.cpp view_device (TT_FATAL on ND_SHARDED)
        #   - ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp PerformView
        #   - ttnn/cpp/ttnn/operations/data_movement/common/common.cpp squeeze_from_ND_to_4D
        # GitHub issue: #36172
        pytest.param(
            [4, 4, 3, 64, 64],
            [2, 3, 0, 31, 31],
            marks=pytest.mark.skip(
                reason="blocked until reshape supports ND-sharded tensors without using ttnn::experimental::view"
            ),
        ),
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
    )
    # In untilize_with_unpadding, if the tensor has rank > 4, it ignores the output_end parameter
    if len(tensor_shape) > 4:
        assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))
    else:
        slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
        assert_equal(input_torch_tensor[slices], ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
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
    )

    if len(tensor_shape) > 4:
        assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))
    else:
        slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
        assert_equal(input_torch_tensor[slices], ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
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
    )

    if len(tensor_shape) > 4:
        assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))
    else:
        slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
        assert_equal(input_torch_tensor[slices], ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
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
    )

    if len(tensor_shape) > 4:
        assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))
    else:
        slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
        assert_equal(input_torch_tensor[slices], ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
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
    )

    if len(tensor_shape) > 4:
        assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))
    else:
        slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
        assert_equal(input_torch_tensor[slices], ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[8, 256, 256]])
@pytest.mark.parametrize(
    "input_shard_shape",
    [
        ttnn.Shape([3, 96, 96]),
    ],
)
@pytest.mark.parametrize("output_end", [ttnn.Shape([3, 127, 127])])
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
    )

    if len(tensor_shape) > 4:
        assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))
    else:
        slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
        assert_equal(input_torch_tensor[slices], ttnn.to_torch(ttnn_output_tensor))


# ---------------------------------------------------------------------------
# Legacy 2D HEIGHT_SHARDED (tiled) input -> INTERLEAVED (L1 or DRAM) untilize-with-unpadding where:
#   * outer dim > 1 (global_batch > 1: several logical matrices), and
#   * inner H is NOT tile-aligned, so each matrix carries its own interior tile-padding tail.
#
# The writer walks each core's absolute rows, maps every one to its (matrix, row-in-matrix), strips
# each matrix's interior pad rows, and lands it at its own row offset in the output -- for any
# alignment of matrices to cores.
#
# Expected behavior: bitwise match (assert_equal).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "shard_layout, tensor_shape, output_end, shard_shape, num_cores",
    [
        # --- HEIGHT_SHARDED ---
        # Baseline (no outer dim): single shard.
        # Tensor [1,1,30,64] padded [1,1,32,64] => single shard of (32, 64).
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [1, 1, 30, 64], [0, 0, 29, 63], (32, 64), 1),
        # Outer dim 2 on dim 0.
        # Tensor [2,1,30,64] padded [2,1,32,64] => physical (64, 64) = 2 shards of (32, 64).
        # Each shard is one matrix slice (32 rows including 2 tile-padded tail rows).
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [2, 1, 30, 64], [1, 0, 29, 63], (32, 64), 2),
        # Outer dim 4: four matrices, one per core.
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [4, 1, 30, 64], [3, 0, 29, 63], (32, 64), 4),
        # Outer product spread across dims 0 and 1 (2 x 2 = 4 slices).
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [2, 2, 30, 64], [1, 1, 29, 63], (32, 64), 4),
    ],
    ids=lambda p: str(p).replace(" ", "") if isinstance(p, list) else None,
)
@pytest.mark.parametrize(
    "output_buffer_type",
    [ttnn.BufferType.DRAM, ttnn.BufferType.L1],
    ids=["dram", "l1"],
)
def test_untilize_with_unpadding_sharded_multi_batch_unpadding_regression(
    device, dtype, shard_layout, tensor_shape, output_end, shard_shape, num_cores, output_buffer_type
):
    """Legacy 2D HEIGHT_SHARDED (tiled) input, non-tile-aligned H, outer_dim > 1, interleaved output.

    Each outer-dim slice is a logical matrix carrying its own interior tile-padding tail.
    untilize-with-unpadding strips each matrix's pad rows and lands it at its own row offset in the
    output; the result must match the sliced torch reference bitwise.
    """
    torch.manual_seed(42)
    torch_tensor = torch.rand(tensor_shape, dtype=torch.bfloat16)

    shard_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_cores - 1))})
    input_shard_spec = ttnn.ShardSpec(shard_core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1, input_shard_spec)

    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, output_buffer_type)

    tile_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tile_tensor = ttnn.to_device(tile_tensor, device, memory_config=input_memory_config)

    untilized = ttnn.untilize_with_unpadding(
        tile_tensor, output_tensor_end=output_end, memory_config=output_memory_config
    )
    result = ttnn.to_torch(untilized)

    slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
    torch_result = torch_tensor[slices]

    assert_equal(result, torch_result)


@pytest.mark.parametrize(
    "padded_shape, output_shape",
    [
        ((1, 1, 32, 7328), (1, 1, 30, 7300)),
        ((1, 1, 128, 7328), (1, 1, 100, 7328)),
        ((1, 1, 64, 8192), (1, 1, 60, 8192)),
        ((1, 1, 96, 7392), (1, 1, 90, 7392)),
        ((1, 1, 160, 6304), (1, 1, 150, 6301)),
        ((2, 1, 64, 7328), (2, 1, 60, 7328)),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_untilize_with_unpadding_block_per_node_cb_size(
    device, padded_shape, output_shape, dtype, isolate_program_cache
):
    torch.manual_seed(42)
    dram_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    keep_alive = []
    entries = None
    for i in range(2):
        torch_input = torch.randn(padded_shape, dtype=torch.bfloat16)
        tt_tiled = ttnn.from_torch(
            torch_input,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dram_cfg,
            device=device,
        )

        output_end = [d - 1 for d in output_shape]
        tt_rm = ttnn.untilize_with_unpadding(tt_tiled, output_end, use_multicore=True)
        keep_alive += [tt_tiled, tt_rm]

        assert tt_rm.layout == ttnn.ROW_MAJOR_LAYOUT
        # bfloat8_b quantizes on the way to the device, so the reference is the device's view of the
        # input rather than the original torch tensor.
        device_input = ttnn.to_torch(tt_tiled)
        torch_golden = device_input[: output_shape[0], : output_shape[1], : output_shape[2], : output_shape[3]]
        assert_equal(torch_golden, ttnn.to_torch(tt_rm))

        if i == 0:
            entries = device.num_program_cache_entries()
            assert entries >= 1, "the first invocation should have populated the program cache"
        else:
            assert (
                device.num_program_cache_entries() == entries
            ), "untilize_with_unpadding must reuse the cached program on a cache hit"


@pytest.mark.parametrize(
    "factory", ["single_core", "multi_core_interleaved", "multi_core_sharded", "multi_core_nd_sharded"]
)
def test_untilize_with_unpadding_spec_factories_program_cache_addr_change(device, factory, isolate_program_cache):
    torch.manual_seed(0)
    shape = [1, 1, 256, 128]
    output_end = [0, 0, 200, 100]
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    def to_device(torch_tensor):
        if factory in ("single_core", "multi_core_interleaved"):
            return ttnn.from_torch(
                torch_tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        if factory == "multi_core_sharded":
            shard_spec = ttnn.ShardSpec(shard_grid, (64, 128), ttnn.ShardOrientation.ROW_MAJOR)
            memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
            host_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            return ttnn.to_device(host_tensor, device, memory_config=memory_config)
        tensor_spec = ttnn.TensorSpec(
            shape=shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1
        ).sharded_across_dims([2, 3], shard_grid, ttnn.ShardOrientation.ROW_MAJOR)
        assert tensor_spec.memory_config.nd_shard_spec is not None
        return ttnn.from_torch(torch_tensor, spec=tensor_spec, device=device)

    slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
    keep_alive = []  # retain prior tensors so each iteration allocates at a new address
    entries = None
    for i in range(3):
        torch_input = torch.rand(shape, dtype=torch.bfloat16)
        tt_input = to_device(torch_input)
        tt_output = ttnn.untilize_with_unpadding(
            tt_input,
            output_tensor_end=output_end,
            memory_config=output_memory_config,
            use_multicore=factory != "single_core",
        )
        keep_alive += [tt_input, tt_output]

        assert_equal(torch_input[slices], ttnn.to_torch(tt_output))

        if i == 0:
            entries = device.num_program_cache_entries()
            assert entries == 1, "the first invocation should build exactly one program"
        else:
            assert device.num_program_cache_entries() == entries, f"{factory} must reuse the cached program on a hit"


@pytest.mark.parametrize(
    "padded_width, out_width",
    [
        (1024, 512),
        (1056, 1056),
        (1056, 512),
        (1056, 1050),
        (1056, 500),
        (4128, 2560),
    ],
)
def test_untilize_with_unpadding_width_crop(device, padded_width, out_width):
    torch.manual_seed(42)
    height = 128
    torch_input = torch.randn(1, height, padded_width, dtype=torch.bfloat16)

    tile_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    untilized = ttnn.untilize_with_unpadding(
        tile_tensor, [0, height - 1, out_width - 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # A bf16 tilize/untilize round trip is an identity, so this is exact.
    assert_equal(ttnn.to_torch(untilized), torch_input[:, :, :out_width])


# An empty input gives the factories 0 blocks, so split_blocks_for_tilize returns empty core ranges
# and no WorkUnitSpec is emitted - while the dataflow buffers have already been declared. The spec
# then fails CollectSpecData with "DFB 'mci_out' has no producer", which surfaced as a TT_FATAL out
# of to_layout(TILE -> ROW_MAJOR) on any zero-volume tensor. The op should hand back the empty
# output instead of building a program with nothing to run.
@pytest.mark.parametrize(
    "shape",
    [
        (0,),  # rank 1, empty
        (2, 3, 0),  # empty last dim
        (2, 0, 4),  # empty interior dim
        (0, 3, 4),  # empty leading dim
        (2, 3, 0, 5),  # rank 4
        (0, 0, 4),  # more than one empty dim
        # Tile-aligned empty shapes: the padded shape already equals the logical one, so to_layout
        # takes the ttnn::untilize branch rather than untilize_with_unpadding. These SIGFPE'd
        # before the guard was added there too.
        (0, 64),
        (0, 32),
        (32, 0),
    ],
)
def test_untilize_with_unpadding_zero_volume(shape, device):
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    tilized = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Must not raise, and must come back with the same (empty) shape.
    untilized = ttnn.to_layout(tilized, ttnn.ROW_MAJOR_LAYOUT)

    result = ttnn.to_torch(untilized)
    assert result.shape == torch_input.shape
    assert result.numel() == 0


# The empty output is allocated, not filled: going through a host tensor would upload to the
# device, and writes are rejected outright during trace capture
# (fd_mesh_command_queue.cpp: "Writes are not supported during trace capture").
@pytest.mark.skipif(is_slow_dispatch(), reason="trace capture is not supported in slow dispatch")
def test_untilize_with_unpadding_zero_volume_in_trace_capture(device):
    torch_input = torch.rand((2, 3, 0), dtype=torch.bfloat16)
    tilized = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Compile outside the capture first, as any traced op requires.
    ttnn.to_layout(tilized, ttnn.ROW_MAJOR_LAYOUT)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    ttnn.to_layout(tilized, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    # Replaying it is the real model scenario; releasing stops the trace leaking into later tests
    # that share the device fixture.
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.release_trace(device, trace_id)


# An empty input carries no element to unpad into a non-empty output. to_layout never asks for one
# (it passes the wrapped sentinel for an empty dim), but a direct caller can, and the empty path
# skips the device operation's validation - so the op has to reject it rather than invent zeros.
def test_untilize_with_unpadding_zero_volume_rejects_nonempty_output(device, expect_error):
    torch_input = torch.rand((0, 32), dtype=torch.bfloat16)
    tilized = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # output_end is inclusive, so [0, 31] asks for a [1, 32] output out of an empty input.
    with expect_error(RuntimeError, "zero-volume input requires a zero-volume output"):
        ttnn.untilize_with_unpadding(tilized, ttnn.Shape([0, 31]))


# A sharded empty input is the case that routes to ttnn::untilize rather than
# untilize_with_unpadding, because its padded shape already equals its logical one. The early
# return feeds memory_config.value_or(input.memory_config()) into TensorLayout, so this is also the
# check that a shard spec can be carried on a zero-volume shape at all.
# Note the tensor has to be built by handing the sharded config to from_torch: routing an existing
# empty tensor through ttnn.to_memory_config segfaults, which is a separate zero-volume defect in
# that op.
@pytest.mark.parametrize("shape", [(0, 64), (0, 32)])
def test_untilize_with_unpadding_zero_volume_sharded(shape, device):
    # The shard width has to match the tensor's own width, or from_torch rejects the pairing
    # before the op is ever reached.
    sharded_config = ttnn.create_sharded_memory_config(
        shape=[32, shape[-1]],
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    tilized = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_config,
    )
    assert tilized.memory_config().is_sharded()

    untilized = ttnn.to_layout(tilized, ttnn.ROW_MAJOR_LAYOUT)

    result = ttnn.to_torch(untilized)
    assert result.shape == torch_input.shape
    assert result.numel() == 0


# The empty output is allocated with the input's tensor topology. Filling it through a host tensor
# instead uploads a single shard, which the mesh then replicates - silently turning a sharded input
# into a replicated output. Verified against that: with the host-upload version these cases come
# back as PlacementReplicate() while the input is PlacementShard(0).
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2_mesh")], indirect=True)
@pytest.mark.parametrize(
    "shape",
    [
        (2, 0),  # empty, shardable along dim 0
        (2, 3, 0),  # empty last dim, shardable along dim 0
    ],
)
def test_untilize_with_unpadding_zero_volume_preserves_mesh_topology(mesh_device, shape):
    if mesh_device.get_num_devices() < 2:
        pytest.skip("needs at least 2 devices to tell a sharded topology from a replicated one")

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    tilized = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    topology_in = str(tilized.tensor_topology())

    untilized = ttnn.to_layout(tilized, ttnn.ROW_MAJOR_LAYOUT)

    assert str(untilized.tensor_topology()) == topology_in


# Rank 0 is a scalar, not an empty tensor: with no dimensions its volume is the empty product, 1,
# so it never reaches the zero-volume guard. A rank-0 *empty* tensor cannot be built for the same
# reason. Kept as a sanity check because nothing else in the ttnn tests builds a rank-0 tensor.
def test_untilize_with_unpadding_rank_0(device):
    torch_input = torch.rand((), dtype=torch.bfloat16)
    assert torch_input.numel() == 1

    tilized = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    assert tilized.logical_volume() == 1

    untilized = ttnn.to_layout(tilized, ttnn.ROW_MAJOR_LAYOUT)

    result = ttnn.to_torch(untilized)
    assert result.shape == torch_input.shape
    assert_equal(result, torch_input)


# device() is null for a host tensor and create_device_tensor dereferences it, so without a guard
# the empty branch segfaults - where the normal path would have fallen through to the device
# operation's validation error.
@pytest.mark.parametrize("shape", [(0, 64), (2, 3, 0)])
def test_untilize_zero_volume_host_tensor_is_rejected(shape, expect_error):
    host_tensor = ttnn.from_torch(torch.rand(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    with expect_error(RuntimeError, "must be allocated on a device"):
        ttnn.untilize(host_tensor)


# Unpadding only ever shrinks, so no output extent may exceed the padded input. Volume alone does
# not catch that: [0, 64] with inclusive ends [0, UINT32_MAX] is zero-volume but shaped [1, 0].
def test_untilize_with_unpadding_zero_volume_rejects_grown_extent(device, expect_error):
    tilized = ttnn.from_torch(
        torch.rand((0, 64), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    with expect_error(RuntimeError, "exceeds the padded input extent"):
        ttnn.untilize_with_unpadding(tilized, ttnn.Shape([0, 4294967295]))


# Both empty shortcuts run ahead of the device operation's validation, so they have to repeat its
# layout check. Without it an empty ROW_MAJOR input is accepted only because it is empty, while the
# same tensor at any non-zero size raises "Can only untilize tile major data".
@pytest.mark.parametrize("shape", [(0, 64), (2, 0)])
def test_untilize_zero_volume_row_major_input_is_rejected(device, shape, expect_error):
    row_major = ttnn.from_torch(
        torch.rand(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ends = ttnn.Shape([extent - 1 if extent else 4294967295 for extent in shape])

    with expect_error(RuntimeError, "Can only untilize tile major data"):
        ttnn.untilize(row_major)

    with expect_error(RuntimeError, "Can only untilize tile major data"):
        ttnn.untilize_with_unpadding(row_major, ends)


# A WIDTH_SHARDED spec pins the shard height to the physical height exactly, and untilizing shrinks
# that height from tile-padded to logical, so the input's shard spec cannot be reused unchanged.
# Before the fix this raised "Shard height 32 must match physical height 4 for width sharded".
# HEIGHT_SHARDED and BLOCK_SHARDED are checked on the width or by div_up and already passed; they
# are here to keep that true.
@pytest.mark.parametrize(
    "memory_layout, shard_shape",
    [
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, [32, 32]),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, [32, 32]),
    ],
)
def test_untilize_with_unpadding_zero_volume_width_sharded(device, memory_layout, shard_shape):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    sharded = ttnn.MemoryConfig(
        memory_layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    )
    tilized = ttnn.from_torch(
        torch.rand((32, 0), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded,
    )

    # Inclusive ends, so [3, UINT32_MAX] unpads the tile-padded height 32 down to 4 and keeps the
    # empty width at 0.
    output = ttnn.untilize_with_unpadding(tilized, ttnn.Shape([3, 4294967295]))

    assert list(output.shape) == [4, 0]
    assert output.layout == ttnn.ROW_MAJOR_LAYOUT
    assert output.memory_config().memory_layout == memory_layout


# A zero-WIDTH output is not supported by this op at any size, so the empty shortcut does not
# support it either -- deliberately, not by omission. Measured on the non-empty twin:
# height-sharded [32, 64] -> [32, 0] kills the process with SIGFPE, and interleaved
# [32, 64] -> [32, 0] hangs the device. Raising is strictly better than both, and making the
# empty path succeed would let an empty input do something no non-empty input can.
def test_untilize_with_unpadding_zero_volume_zero_width_sharded_is_rejected(device, expect_error):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, 64], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tilized = ttnn.from_torch(
        torch.rand((0, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded,
    )

    with expect_error(RuntimeError, "must match physical width"):
        ttnn.untilize_with_unpadding(tilized, ttnn.Shape([4294967295, 4294967295]))


# Interleaved has no shard width to contradict, so there the zero-width output is allocatable and
# the shortcut does return it -- the case the sharded test above cannot have.
def test_untilize_with_unpadding_zero_volume_zero_width_interleaved(device):
    tilized = ttnn.from_torch(
        torch.rand((0, 64), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    output = ttnn.untilize_with_unpadding(tilized, ttnn.Shape([4294967295, 4294967295]))

    assert list(output.shape) == [0, 0]
    assert output.layout == ttnn.ROW_MAJOR_LAYOUT


# The shortcuts run ahead of the device operation, so they also have to repeat its checks on the
# ARGUMENTS, not just on the shape. Each case below was measured against its non-empty twin: before
# the fix the empty input and the non-empty one disagreed, and now they raise the same error.
def _two_cores():
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})


def _height_sharded(shard=[32, 64]):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )


def test_untilize_zero_volume_rejects_sub_core_grids_without_multicore(device, expect_error):
    tilized = ttnn.from_torch(
        torch.rand((0, 64), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    with expect_error(RuntimeError, "use_multicore"):
        ttnn.untilize(tilized, use_multicore=False, sub_core_grids=_two_cores())


def test_untilize_with_unpadding_zero_volume_rejects_sub_core_grids_when_sharded(device, expect_error):
    tilized = ttnn.from_torch(
        torch.rand((0, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_height_sharded(),
    )

    with expect_error(RuntimeError, "does not support sub core grid"):
        ttnn.untilize_with_unpadding(tilized, ttnn.Shape([4294967295, 63]), sub_core_grids=_two_cores())


# A sharded memory config that names a layout but carries no shard spec. The device operation fills
# it in from the input, so the empty path has to as well -- this one FAILED before the fix while the
# non-empty call succeeded.
def test_untilize_with_unpadding_zero_volume_inherits_missing_shard_spec(device):
    tilized = ttnn.from_torch(
        torch.rand((0, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_height_sharded(),
    )

    output = ttnn.untilize_with_unpadding(
        tilized, ttnn.Shape([4294967295, 63]), memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
    )

    assert list(output.shape) == [0, 64]
    assert output.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def test_untilize_with_unpadding_zero_volume_rejects_incompatible_sharded_output(device, expect_error):
    tilized = ttnn.from_torch(
        torch.rand((0, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_height_sharded(),
    )
    block_sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_two_cores(), [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )

    with expect_error(RuntimeError, "must be HEIGHT_SHARDED"):
        ttnn.untilize_with_unpadding(tilized, ttnn.Shape([4294967295, 63]), memory_config=block_sharded)


# Output-memory-config handling on the empty path, each case measured against its non-empty twin.
def _crs(x1=0, y1=0):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x1, y1))})


def _sharded(layout, shard, grid, buffer_type=ttnn.BufferType.L1):
    return ttnn.MemoryConfig(layout, buffer_type, ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR))


# A padded interleaved input belongs to untilize_with_unpadding, which derives the output shard
# geometry. ttnn.untilize used to answer it from its own branch and reject the conversion.
def test_untilize_zero_volume_padded_interleaved_to_width_sharded(device):
    tilized = ttnn.from_torch(
        torch.rand((2, 3, 0), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    width_sharded = _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, [32, 32], _crs())

    output = ttnn.untilize(tilized, memory_config=width_sharded)

    assert list(output.shape) == [2, 3, 0]
    assert output.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED


def test_untilize_zero_volume_rejects_dram_block_sharded_output(device, expect_error):
    tilized = ttnn.from_torch(
        torch.rand((0, 64), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    dram_block = _sharded(ttnn.TensorMemoryLayout.BLOCK_SHARDED, [32, 64], _crs(), ttnn.BufferType.DRAM)

    with expect_error(RuntimeError, "don't support DRAM block sharding"):
        ttnn.untilize(tilized, memory_config=dram_block)


def test_untilize_with_unpadding_zero_volume_rejects_block_to_height_sharded(device, expect_error):
    tilized = ttnn.from_torch(
        torch.rand((0, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded(ttnn.TensorMemoryLayout.BLOCK_SHARDED, [32, 32], _crs(x1=1)),
    )
    height_sharded = _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [32, 64], _crs())

    with expect_error(RuntimeError, "must be BLOCK_SHARDED"):
        ttnn.untilize_with_unpadding(tilized, ttnn.Shape([4294967295, 63]), memory_config=height_sharded)


def test_untilize_with_unpadding_zero_volume_rejects_dram_height_sharded_output(device, expect_error):
    tilized = ttnn.from_torch(
        torch.rand((0, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [32, 64], _crs()),
    )
    dram_height = _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [32, 64], _crs(), ttnn.BufferType.DRAM)

    with expect_error(RuntimeError, "must be in L1"):
        ttnn.untilize_with_unpadding(tilized, ttnn.Shape([4294967295, 63]), memory_config=dram_height)


# The caller's shard width of 32 needs four columns for a 128-wide output; the grid has two. The
# device operation derives 64 from the grid, and so does the empty path now.
def test_untilize_with_unpadding_zero_volume_derives_block_shard_width(device):
    tilized = ttnn.from_torch(
        torch.rand((0, 128), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    block_sharded = _sharded(ttnn.TensorMemoryLayout.BLOCK_SHARDED, [32, 32], _crs(x1=1))

    output = ttnn.untilize_with_unpadding(tilized, ttnn.Shape([4294967295, 127]), memory_config=block_sharded)

    assert list(output.shape) == [0, 128]
    assert output.memory_config().shard_spec.shape == [32, 64]


def test_untilize_zero_volume_rejects_narrow_shard_on_single_core(device, expect_error):
    tilized = ttnn.from_torch(
        torch.rand((0, 64), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    narrow = _sharded(ttnn.TensorMemoryLayout.BLOCK_SHARDED, [32, 16], _crs(x1=3))

    with expect_error(RuntimeError, "must be a multiple of tile width"):
        ttnn.untilize(tilized, memory_config=narrow, use_multicore=False)


def test_untilize_with_unpadding_zero_volume_rejects_dram_width_sharded_output(device, expect_error):
    tilized = ttnn.from_torch(
        torch.rand((32, 0), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, [32, 32], _crs()),
    )
    dram_width = _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, [32, 32], _crs(), ttnn.BufferType.DRAM)

    with expect_error(RuntimeError, "must be in L1"):
        ttnn.untilize_with_unpadding(tilized, ttnn.Shape([3, 4294967295]), memory_config=dram_width)


# A same-layout output is reshaped from the input's shard, not the caller's: an explicit [32, 32]
# cannot describe a 64-wide height-sharded output, and the non-empty call derives [32, 64] too.
def test_untilize_with_unpadding_zero_volume_same_layout_output_follows_input_shard(device):
    tilized = ttnn.from_torch(
        torch.rand((0, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [32, 64], _crs()),
    )
    narrower = _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [32, 32], _crs())

    output = ttnn.untilize_with_unpadding(tilized, ttnn.Shape([4294967295, 63]), memory_config=narrower)

    assert list(output.shape) == [0, 64]
    assert output.memory_config().shard_spec.shape == [32, 64]


# The validator's batch-dimension loop used `rank() - 2` on an unsigned rank, so a rank-1 input
# wrapped to 4294967295 and compared the padded height against dim 0. Both of these failed.
@pytest.mark.parametrize("shape, end", [((0,), 4294967295), ((32,), 31)])
def test_untilize_with_unpadding_rank_1_width_sharded(device, shape, end):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    width_sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tilized = ttnn.from_torch(
        torch.rand(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=width_sharded,
    )

    output = ttnn.untilize_with_unpadding(tilized, ttnn.Shape([end]))

    assert list(output.shape) == list(shape)


# The zero-extent fallback exists for empty inputs. A NON-empty input asked for an empty sharded
# output must still be rejected on the zero shard extent, as it was before that fallback existed --
# otherwise the spec is accepted and the writer targets storage that was never allocated.
def test_untilize_with_unpadding_nonempty_input_rejects_empty_sharded_output(device, expect_error):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    height_sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, 64], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tilized = ttnn.from_torch(
        torch.rand((32, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=height_sharded,
    )

    with expect_error(RuntimeError, "greater than 0 in each sharded dim"):
        ttnn.untilize_with_unpadding(tilized, ttnn.Shape([4294967295, 63]))
