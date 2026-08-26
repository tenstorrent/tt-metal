# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.unit_tests.base_functionality.test_narrow import assert_quality


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16, ttnn.int32])
@pytest.mark.parametrize("tensor_shape", [[2, 2, 256, 512]])
def test_untilize_single_core_interleaved_to_interleaved(device, dtype, tensor_shape):
    torch.manual_seed(42)
    # Input memory config
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Output memory config
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Test
    if dtype == ttnn.int32:
        input_torch_tensor = torch.randint(-1000, 1000, tensor_shape, dtype=torch.int32)
    else:
        input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=False)

    if dtype == ttnn.int32:
        assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))
    else:
        assert_quality(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), dtype)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[2, 2, 256, 512]])
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
    "num_shard_cores, standard_shard_core_grid, block_shard_core_grid",
    [
        [
            4,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ],
        [
            16,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                }
            ),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ],
    ],
)
def test_untilize_single_core_interleaved_to_sharded(
    device,
    dtype,
    tensor_shape,
    output_memory_layout,
    output_shard_orientation,
    num_shard_cores,
    standard_shard_core_grid,
    block_shard_core_grid,
):
    num_tensor_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_tensor_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[num_tensor_dims - 1]

    # Shard shapes
    height_sharded_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_sharded_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    block_sharded_shard_shape = (
        tensor_height // int(math.sqrt(num_shard_cores)),
        tensor_width // int(math.sqrt(num_shard_cores)),
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

    # Input memory config
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Output memory config
    output_shard_memory_layout = shard_memory_layout_map[output_memory_layout]
    output_shard_spec = ttnn.ShardSpec(
        output_shard_memory_layout["shard_grid"], output_shard_memory_layout["shard_shape"], output_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=False)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[32, 256]])
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
    "num_shard_cores, standard_shard_core_grid, block_shard_core_grid",
    [
        [
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ],
        [
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))}),
        ],
    ],
)
def test_untilize_single_core_interleaved_to_sharded_writer_kernel_tensor_addrgen_test(
    device,
    dtype,
    tensor_shape,
    output_memory_layout,
    output_shard_orientation,
    num_shard_cores,
    standard_shard_core_grid,
    block_shard_core_grid,
):
    """
    This test tests the single-core equivalent case that caused the failure solved in PR #35686.
    The old single core writer kernel using the ShardedAddrGen API gives an incorrect output on this test case, whereas TensorAccessor gives the right output.

    """
    if output_memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        pytest.skip(
            "Width sharded case results in shard with width < tile width, which is not supported in single core implementation."
        )

    num_tensor_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_tensor_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[num_tensor_dims - 1]

    # Shard shapes
    height_sharded_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_sharded_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    # below formula assumes the CoreRangeSet is not disjoint
    block_grid_size = block_shard_core_grid.bounding_box().grid_size()
    if output_shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        block_sharded_shard_shape = (
            tensor_height // block_grid_size.y,
            tensor_width // block_grid_size.x,
        )
    else:
        block_sharded_shard_shape = (
            tensor_height // block_grid_size.x,
            tensor_width // block_grid_size.y,
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

    # Input memory config
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Output memory config
    output_shard_memory_layout = shard_memory_layout_map[output_memory_layout]
    output_shard_spec = ttnn.ShardSpec(
        output_shard_memory_layout["shard_grid"], output_shard_memory_layout["shard_shape"], output_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=False)
    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[2, 2, 256, 512]])
@pytest.mark.parametrize(
    "input_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
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
        [
            4,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ],
        [
            16,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                }
            ),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ],
    ],
)
def test_untilize_single_core_sharded_to_interleaved(
    device,
    dtype,
    tensor_shape,
    input_memory_layout,
    input_shard_orientation,
    num_shard_cores,
    standard_shard_core_grid,
    block_shard_core_grid,
):
    num_tensor_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_tensor_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[num_tensor_dims - 1]

    # Shard shapes
    height_sharded_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_sharded_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    block_sharded_shard_shape = (
        tensor_height // int(math.sqrt(num_shard_cores)),
        tensor_width // int(math.sqrt(num_shard_cores)),
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

    # Input memory config
    input_shard_memory_layout = shard_memory_layout_map[input_memory_layout]
    output_shard_spec = ttnn.ShardSpec(
        input_shard_memory_layout["shard_grid"], input_shard_memory_layout["shard_shape"], input_shard_orientation
    )
    input_memory_config = ttnn.MemoryConfig(input_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    # Output memory config
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=False)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[2, 2, 256, 512]])
@pytest.mark.parametrize(
    "input_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
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
    "num_shard_cores, standard_shard_core_grid, block_shard_core_grid",
    [
        [
            4,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ],
        [
            16,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                }
            ),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ],
    ],
)
def test_untilize_single_core_sharded_to_sharded(
    device,
    dtype,
    tensor_shape,
    input_memory_layout,
    input_shard_orientation,
    output_memory_layout,
    output_shard_orientation,
    num_shard_cores,
    standard_shard_core_grid,
    block_shard_core_grid,
):
    num_tensor_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_tensor_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[num_tensor_dims - 1]

    # Shard shapes
    height_sharded_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_sharded_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    block_sharded_shard_shape = (
        tensor_height // int(math.sqrt(num_shard_cores)),
        tensor_width // int(math.sqrt(num_shard_cores)),
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

    # Input memory config
    input_shard_memory_layout = shard_memory_layout_map[input_memory_layout]
    input_shard_spec = ttnn.ShardSpec(
        input_shard_memory_layout["shard_grid"], input_shard_memory_layout["shard_shape"], input_shard_orientation
    )
    input_memory_config = ttnn.MemoryConfig(input_memory_layout, ttnn.BufferType.L1, input_shard_spec)

    # Output memory config
    output_shard_memory_layout = shard_memory_layout_map[output_memory_layout]
    output_shard_spec = ttnn.ShardSpec(
        output_shard_memory_layout["shard_grid"], output_shard_memory_layout["shard_shape"], output_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=False)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
@pytest.mark.parametrize("tensor_shape", [[1, 1, 512, 512]])
@pytest.mark.parametrize("input_buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
@pytest.mark.parametrize("output_buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
@pytest.mark.parametrize(
    "input_memory_layout",
    [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ],
)
@pytest.mark.parametrize(
    "output_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.INTERLEAVED,
    ],
)
def test_untilize_single_core_buffer_type_variations(
    device,
    dtype,
    tensor_shape,
    input_buffer_type,
    output_buffer_type,
    input_memory_layout,
    output_memory_layout,
):
    height_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        (128, 512),
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    # Input memory config
    if input_memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        input_memory_config = ttnn.MemoryConfig(input_memory_layout, input_buffer_type)
    else:
        input_memory_config = ttnn.MemoryConfig(input_memory_layout, input_buffer_type, height_shard_spec)

    # Output memory config
    if output_memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        output_memory_config = ttnn.MemoryConfig(output_memory_layout, output_buffer_type)
    else:
        output_memory_config = ttnn.MemoryConfig(output_memory_layout, output_buffer_type, height_shard_spec)

    # Test
    if dtype == ttnn.bfloat16:
        input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    else:
        input_torch_tensor = torch.randint(-(2 ** (30)), 2 ** (30) - 1, tensor_shape, dtype=torch.int32)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=False)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape",
    [
        [2, 256, 512],
        [4128, 512],  # multiple blocks per core and a cliff core
    ],
)
def test_untilize_multi_core_interleaved_to_interleaved(device, dtype, tensor_shape):
    # Input memory config
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Output memory config
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_quality(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), dtype)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape",
    [
        [2, 256, 512],
        [4128, 512],  # multiple blocks per core and a cliff core
        [32, 256],  # used in deepseek before MoE Gate (bfloat16, height sharded on 32 cores)
    ],
)
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
    "num_shard_cores, standard_shard_core_grid, block_shard_core_grid",
    [
        [
            4,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ],
        [
            16,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                }
            ),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ],
        [
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
        ],
    ],
)
def test_untilize_multi_core_interleaved_to_sharded(
    device,
    dtype,
    tensor_shape,
    output_memory_layout,
    output_shard_orientation,
    num_shard_cores,
    standard_shard_core_grid,
    block_shard_core_grid,
):
    num_tensor_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_tensor_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[num_tensor_dims - 1]

    # Shard shapes
    height_sharded_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_sharded_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    # below formula assumes the CoreRangeSet is not disjoint
    block_grid_size = block_shard_core_grid.bounding_box().grid_size()
    if output_shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        block_sharded_shard_shape = (
            tensor_height // block_grid_size.y,
            tensor_width // block_grid_size.x,
        )
    else:
        block_sharded_shard_shape = (
            tensor_height // block_grid_size.x,
            tensor_width // block_grid_size.y,
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

    # Input memory config
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Output memory config
    input_shard_memory_layout = shard_memory_layout_map[output_memory_layout]
    output_shard_spec = ttnn.ShardSpec(
        input_shard_memory_layout["shard_grid"], input_shard_memory_layout["shard_shape"], output_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[160, 160]])
@pytest.mark.parametrize(
    "output_memory_layout, output_shard_shape, output_shard_core_grid",
    [
        (
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))}),
        ),
        (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 2))}),
        ),
        (
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (160, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
        (
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 3))}),
        ),
        (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))}),
        ),
        (
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (160, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "output_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
def test_untilize_multi_core_interleaved_to_uneven_sharded(
    device,
    dtype,
    tensor_shape,
    output_memory_layout,
    output_shard_shape,
    output_shard_core_grid,
    output_shard_orientation,
):
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    output_shard_spec = ttnn.ShardSpec(output_shard_core_grid, output_shard_shape, output_shard_orientation)
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[4, 160, 160]])
@pytest.mark.parametrize(
    "output_nd_shard_shape, output_shard_core_grid",
    [
        (
            ttnn.Shape([3, 64, 64]),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))}),
        ),
        (
            ttnn.Shape([5, 64, 160]),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 2))}),
        ),
        (
            ttnn.Shape([3, 160, 64]),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "output_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
def test_untilize_multi_core_interleaved_to_uneven_nd_sharded(
    device,
    dtype,
    tensor_shape,
    output_nd_shard_shape,
    output_shard_core_grid,
    output_shard_orientation,
):
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_nd_shard_shape, grid=output_shard_core_grid, orientation=output_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)

    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape",
    [
        [2, 256, 512],
        [4, 4, 256, 512],
    ],
)
@pytest.mark.parametrize(
    "input_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
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
        [
            4,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ],
        [
            16,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                }
            ),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ],
    ],
)
def test_untilize_multi_core_sharded_to_interleaved(
    device,
    dtype,
    tensor_shape,
    input_memory_layout,
    input_shard_orientation,
    num_shard_cores,
    standard_shard_core_grid,
    block_shard_core_grid,
):
    num_tensor_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_tensor_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[num_tensor_dims - 1]

    # Shard shapes
    height_sharded_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_sharded_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    block_sharded_shard_shape = (
        tensor_height // int(math.sqrt(num_shard_cores)),
        tensor_width // int(math.sqrt(num_shard_cores)),
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

    # Input memory config
    input_shard_memory_layout = shard_memory_layout_map[input_memory_layout]
    output_shard_spec = ttnn.ShardSpec(
        input_shard_memory_layout["shard_grid"], input_shard_memory_layout["shard_shape"], input_shard_orientation
    )
    input_memory_config = ttnn.MemoryConfig(input_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    # Output memory config
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[160, 160]])
@pytest.mark.parametrize(
    "input_memory_layout, input_shard_shape, input_shard_core_grid",
    [
        [
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (128, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
        ],
        [
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (160, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
        ],
        [
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (128, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ],
    ],
)
@pytest.mark.parametrize(
    "input_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
def test_untilize_multi_core_sharded_to_interleaved_uneven_input_shard_spec(
    device,
    dtype,
    tensor_shape,
    input_memory_layout,
    input_shard_shape,
    input_shard_core_grid,
    input_shard_orientation,
):
    # Input Memory config
    input_shard_spec = ttnn.ShardSpec(input_shard_core_grid, input_shard_shape, input_shard_orientation)
    input_memory_config = ttnn.MemoryConfig(input_memory_layout, ttnn.BufferType.L1, input_shard_spec)

    # Output memory config
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[2, 2, 128, 512]])
@pytest.mark.parametrize(
    "input_memory_layout, output_memory_layout",
    [
        [ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED],
        [ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED],
        [ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED],
        [ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED],
        [ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED],
        [ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED],
    ],
)
@pytest.mark.parametrize("input_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("output_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "num_shard_cores, standard_shard_core_grid, block_shard_core_grid",
    [
        [
            4,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ],
        [
            16,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                }
            ),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ],
    ],
)
def test_untilize_multi_core_sharded_to_sharded_different_shard_types(
    device,
    dtype,
    tensor_shape,
    input_memory_layout,
    output_memory_layout,
    input_shard_orientation,
    output_shard_orientation,
    num_shard_cores,
    standard_shard_core_grid,
    block_shard_core_grid,
):
    num_tensor_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_tensor_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[num_tensor_dims - 1]

    # Shard shapes
    height_sharded_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_sharded_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    block_sharded_shard_shape = (
        tensor_height // int(math.sqrt(num_shard_cores)),
        tensor_width // int(math.sqrt(num_shard_cores)),
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

    # Input memory config
    input_shard_memory_layout = shard_memory_layout_map[input_memory_layout]
    input_shard_spec = ttnn.ShardSpec(
        input_shard_memory_layout["shard_grid"], input_shard_memory_layout["shard_shape"], input_shard_orientation
    )
    input_memory_config = ttnn.MemoryConfig(input_memory_layout, ttnn.BufferType.L1, input_shard_spec)

    # Output memory config
    output_shard_memory_layout = shard_memory_layout_map[output_memory_layout]
    output_shard_spec = ttnn.ShardSpec(
        output_shard_memory_layout["shard_grid"], output_shard_memory_layout["shard_shape"], output_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[160, 160]])
@pytest.mark.parametrize(
    "input_memory_layout, input_shard_shape, input_shard_core_grid, output_memory_layout, output_shard_shape, output_shard_core_grid",
    [
        [
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (128, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (160, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 4))}),
        ],
        [
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (128, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 4))}),
        ],
        [
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (160, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 4))}),
        ],
        [
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (160, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 4))}),
        ],
        [
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (128, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 4))}),
        ],
        [
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (128, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (160, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 4))}),
        ],
        [
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (128, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (160, 48),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        ],
        [
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (160, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ],
        [
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (160, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        ],
        [
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
        [
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ],
        [
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ],
        [
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (160, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
        [
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ],
        [
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (160, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (160, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        ],
    ],
)
@pytest.mark.parametrize("input_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("output_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_untilize_multi_core_sharded_to_sharded_different_shard_types_uneven_input_shard_spec(
    device,
    dtype,
    tensor_shape,
    input_memory_layout,
    input_shard_shape,
    input_shard_core_grid,
    output_memory_layout,
    output_shard_shape,
    output_shard_core_grid,
    input_shard_orientation,
    output_shard_orientation,
):
    # Input Memory config
    input_shard_spec = ttnn.ShardSpec(input_shard_core_grid, input_shard_shape, input_shard_orientation)
    input_memory_config = ttnn.MemoryConfig(input_memory_layout, ttnn.BufferType.L1, input_shard_spec)

    # Output memory config
    output_shard_spec = ttnn.ShardSpec(output_shard_core_grid, output_shard_shape, output_shard_orientation)
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[2, 2, 128, 512]])
@pytest.mark.parametrize(
    "memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize("input_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("output_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "input_num_shard_cores, input_standard_shard_core_grid, input_block_shard_core_grid",
    [
        [
            4,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ],
        [
            16,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                }
            ),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ],
    ],
)
@pytest.mark.parametrize(
    "output_num_shard_cores, output_standard_shard_core_grid, output_block_shard_core_grid",
    [
        [
            4,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ],
        [
            16,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                }
            ),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ],
    ],
)
def test_untilize_multi_core_sharded_to_sharded_same_shard_type_different_shard_spec(
    device,
    dtype,
    tensor_shape,
    memory_layout,
    input_shard_orientation,
    output_shard_orientation,
    input_num_shard_cores,
    input_standard_shard_core_grid,
    input_block_shard_core_grid,
    output_num_shard_cores,
    output_standard_shard_core_grid,
    output_block_shard_core_grid,
):
    num_tensor_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_tensor_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[num_tensor_dims - 1]

    # Input Shard shapes
    input_height_sharded_shard_shape = (tensor_height // input_num_shard_cores, tensor_width)
    input_width_sharded_shard_shape = (tensor_height, tensor_width // input_num_shard_cores)
    input_block_sharded_shard_shape = (
        tensor_height // int(math.sqrt(input_num_shard_cores)),
        tensor_width // int(math.sqrt(input_num_shard_cores)),
    )

    # Input Shard shapes
    output_height_sharded_shard_shape = (tensor_height // output_num_shard_cores, tensor_width)
    output_width_sharded_shard_shape = (tensor_height, tensor_width // output_num_shard_cores)
    output_block_sharded_shard_shape = (
        tensor_height // int(math.sqrt(output_num_shard_cores)),
        tensor_width // int(math.sqrt(output_num_shard_cores)),
    )

    # Input Shard Memory Layout Map
    input_shard_memory_layout_map = {
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED: {
            "shard_grid": input_standard_shard_core_grid,
            "shard_shape": input_height_sharded_shard_shape,
        },
        ttnn.TensorMemoryLayout.WIDTH_SHARDED: {
            "shard_grid": input_standard_shard_core_grid,
            "shard_shape": input_width_sharded_shard_shape,
        },
        ttnn.TensorMemoryLayout.BLOCK_SHARDED: {
            "shard_grid": input_block_shard_core_grid,
            "shard_shape": input_block_sharded_shard_shape,
        },
    }

    # Output Shard Memory Layout Map
    output_shard_memory_layout_map = {
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED: {
            "shard_grid": output_standard_shard_core_grid,
            "shard_shape": output_height_sharded_shard_shape,
        },
        ttnn.TensorMemoryLayout.WIDTH_SHARDED: {
            "shard_grid": output_standard_shard_core_grid,
            "shard_shape": output_width_sharded_shard_shape,
        },
        ttnn.TensorMemoryLayout.BLOCK_SHARDED: {
            "shard_grid": output_block_shard_core_grid,
            "shard_shape": output_block_sharded_shard_shape,
        },
    }

    # Input memory config
    input_shard_memory_layout = input_shard_memory_layout_map[memory_layout]
    input_shard_spec = ttnn.ShardSpec(
        input_shard_memory_layout["shard_grid"], input_shard_memory_layout["shard_shape"], input_shard_orientation
    )
    input_memory_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, input_shard_spec)

    # Output memory config
    output_shard_memory_layout = output_shard_memory_layout_map[memory_layout]
    output_shard_spec = ttnn.ShardSpec(
        output_shard_memory_layout["shard_grid"], output_shard_memory_layout["shard_shape"], output_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, output_shard_spec)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[160, 160]])
@pytest.mark.parametrize(
    "memory_layout, input_shard_shape, input_shard_core_grid, output_shard_shape, output_shard_core_grid",
    [
        [
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (128, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
            (32, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 4))}),
        ],
        [
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (160, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
            (160, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 4))}),
        ],
        [
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (128, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            (32, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 4))}),
        ],
    ],
)
@pytest.mark.parametrize("input_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("output_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
def test_untilize_multi_core_sharded_to_sharded_same_shard_type_different_shard_spec_uneven_input_shard_spec(
    device,
    dtype,
    tensor_shape,
    memory_layout,
    input_shard_shape,
    input_shard_core_grid,
    output_shard_shape,
    output_shard_core_grid,
    input_shard_orientation,
    output_shard_orientation,
):
    # Input Memory config
    input_shard_spec = ttnn.ShardSpec(input_shard_core_grid, input_shard_shape, input_shard_orientation)
    input_memory_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, input_shard_spec)

    # Output memory config
    output_shard_spec = ttnn.ShardSpec(output_shard_core_grid, output_shard_shape, output_shard_orientation)
    output_memory_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, output_shard_spec)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[2, 2, 128, 512]])
@pytest.mark.parametrize(
    "memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "num_shard_cores, standard_shard_core_grid, block_shard_core_grid",
    [
        [
            4,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ],
        [
            16,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                }
            ),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ],
    ],
)
def test_untilize_multi_core_sharded_to_sharded_same_shard_type_and_shard_spec(
    device,
    dtype,
    tensor_shape,
    memory_layout,
    shard_orientation,
    num_shard_cores,
    standard_shard_core_grid,
    block_shard_core_grid,
):
    # This test targets a special case implementation for when
    # the input and output shard types and shard specs are identical

    num_tensor_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_tensor_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[num_tensor_dims - 1]

    # Shard shapes
    height_sharded_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_sharded_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    block_sharded_shard_shape = (
        tensor_height // int(math.sqrt(num_shard_cores)),
        tensor_width // int(math.sqrt(num_shard_cores)),
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

    # Memory config
    shard_memory_layout = shard_memory_layout_map[memory_layout]
    shard_spec = ttnn.ShardSpec(
        shard_memory_layout["shard_grid"], shard_memory_layout["shard_shape"], shard_orientation
    )
    memory_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[160, 160]])
@pytest.mark.parametrize(
    "memory_layout, shard_shape, shard_core_grid",
    [
        [
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (128, 160),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
        ],
        [
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (160, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
        ],
        [
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (128, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ],
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
def test_untilize_multi_core_sharded_to_sharded_same_shard_type_and_shard_spec_uneven_shard_spec(
    device,
    dtype,
    tensor_shape,
    memory_layout,
    shard_shape,
    shard_core_grid,
    shard_orientation,
):
    # This test targets a special case implementation for when
    # the input and output shard types and shard specs are identical

    # Memory config
    shard_spec = ttnn.ShardSpec(shard_core_grid, shard_shape, shard_orientation)
    memory_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[1, 1, 512, 512]])
@pytest.mark.parametrize("input_buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
@pytest.mark.parametrize("output_buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
@pytest.mark.parametrize(
    "input_memory_layout",
    [
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ],
)
@pytest.mark.parametrize(
    "output_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.INTERLEAVED,
    ],
)
def test_untilize_multi_core_buffer_type_variations(
    device,
    dtype,
    tensor_shape,
    input_buffer_type,
    output_buffer_type,
    input_memory_layout,
    output_memory_layout,
):
    height_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        (128, 512),
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    # Input memory config
    if input_memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        input_memory_config = ttnn.MemoryConfig(input_memory_layout, input_buffer_type)
    else:
        input_memory_config = ttnn.MemoryConfig(input_memory_layout, input_buffer_type, height_shard_spec)

    # Output memory config
    if output_memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        output_memory_config = ttnn.MemoryConfig(output_memory_layout, output_buffer_type)
    else:
        output_memory_config = ttnn.MemoryConfig(output_memory_layout, output_buffer_type, height_shard_spec)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize(
    "tensor_shape",
    [[32, 32], [1, 1, 128, 128], [32, 64, 64], [32, 32, 32, 32]],
)
@pytest.mark.parametrize("input_buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
@pytest.mark.parametrize("output_buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
def test_untilize_fp32(device, tensor_shape, input_buffer_type, output_buffer_type):
    torch.manual_seed(42)

    torch_tensor = torch.rand(tensor_shape, dtype=torch.float32)

    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, input_buffer_type)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, output_buffer_type)

    tile_tensor = ttnn.from_torch(
        torch_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory_config
    )
    untilized = ttnn.untilize(tile_tensor, memory_config=output_memory_config)
    result = ttnn.to_torch(untilized)

    assert_equal(result, torch_tensor)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_multicore", [True, False])
@pytest.mark.parametrize(
    "shape_pairs",
    [
        # Pairs of shapes with same volume but different dimensions
        # These would cause hash collisions if only volume is used in hash
        ([[1, 256, 128], [1, 128, 256]], 32768),  # 8x4 tiles vs 4x8 tiles
        ([[1, 320, 128], [1, 128, 320]], 40960),  # 10x4 tiles vs 4x10 tiles
        ([[1, 4, 128, 128], [1, 2, 128, 256]], 65536),  # 4D tensors with same volume
        ([[1, 8, 128, 64], [1, 4, 64, 256]], 65536),  # Different 4D arrangements
    ],
)
def test_untilize_same_volume_different_shapes(device, dtype, use_multicore, shape_pairs):
    """
    Regression test for program cache hash collision issue.

    This test verifies that tensors with the same volume but different shapes
    are correctly handled by untilize without hash collisions in the program cache.

    The bug was that compute_program_hash() used input_shape.volume() instead of
    the full shape, causing tensors like (1, 256, 128) and (1, 128, 256) to have
    the same hash and incorrectly share cached programs.
    """
    shapes, expected_volume = shape_pairs

    # Verify test setup - shapes should have same volume
    for shape in shapes:
        volume = 1
        for dim in shape:
            volume *= dim
        assert volume == expected_volume, f"Shape {shape} has volume {volume}, expected {expected_volume}"

    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    # Run untilize on all shapes in sequence to trigger potential cache reuse issues
    for shape in shapes:
        torch.manual_seed(42)
        input_torch_tensor = torch.randn(shape, dtype=torch.bfloat16)

        input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)

        ttnn_output_tensor = ttnn.untilize(
            input_ttnn_tensor, memory_config=output_memory_config, use_multicore=use_multicore
        )

        assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape, output_nd_shard_shape", [([2, 2, 256, 512], [2, 64, 64])])
@pytest.mark.parametrize(
    "output_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "standard_shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
            }
        ),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
    ],
)
def test_untilize_single_core_interleaved_to_nd_sharded(
    device,
    dtype,
    tensor_shape,
    output_nd_shard_shape,
    output_shard_orientation,
    standard_shard_core_grid,
):
    # Output ND shard shape
    output_nd_shard_shape = ttnn.Shape(output_nd_shard_shape)

    # Output ND shard spec and memory config
    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_nd_shard_shape, grid=standard_shard_core_grid, orientation=output_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)

    # Input memory config (interleaved)
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=False)
    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape, output_nd_shard_shape", [([2, 2, 256, 512], [2, 64, 64])])
@pytest.mark.parametrize(
    "output_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize(
    "input_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
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
            16,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                }
            ),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ),
    ],
)
def test_untilize_single_core_legacy_sharded_to_nd_sharded(
    device,
    dtype,
    tensor_shape,
    output_nd_shard_shape,
    output_shard_orientation,
    input_memory_layout,
    input_shard_orientation,
    num_shard_cores,
    standard_shard_core_grid,
    block_shard_core_grid,
):
    num_tensor_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_tensor_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[num_tensor_dims - 1]

    # Input shard shapes
    height_sharded_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_sharded_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    block_sharded_shard_shape = (
        tensor_height // int(math.sqrt(num_shard_cores)),
        tensor_width // int(math.sqrt(num_shard_cores)),
    )

    input_shard_memory_layout_map = {
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

    # Input memory config (legacy sharded)
    input_shard_memory_layout = input_shard_memory_layout_map[input_memory_layout]
    input_shard_spec = ttnn.ShardSpec(
        input_shard_memory_layout["shard_grid"],
        input_shard_memory_layout["shard_shape"],
        input_shard_orientation,
    )
    input_memory_config = ttnn.MemoryConfig(input_memory_layout, ttnn.BufferType.L1, input_shard_spec)

    # Output ND shard spec and memory config
    output_nd_shard_shape = ttnn.Shape(output_nd_shard_shape)
    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_nd_shard_shape, grid=standard_shard_core_grid, orientation=output_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=False)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[2, 2, 256, 512]])
@pytest.mark.parametrize(
    "input_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
@pytest.mark.parametrize("output_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "input_nd_shard_shape, output_nd_shard_shape",
    [
        (ttnn.Shape([2, 128, 64]), ttnn.Shape([2, 64, 64])),
        (ttnn.Shape([2, 64, 64]), ttnn.Shape([2, 64, 64])),
    ],
)
@pytest.mark.parametrize(
    "standard_shard_core_grid",
    [
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
    ],
)
def test_untilize_single_core_nd_sharded_to_nd_sharded(
    device,
    dtype,
    tensor_shape,
    input_shard_orientation,
    output_shard_orientation,
    input_nd_shard_shape,
    output_nd_shard_shape,
    standard_shard_core_grid,
):
    torch.manual_seed(0)

    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_nd_shard_shape, grid=standard_shard_core_grid, orientation=input_shard_orientation
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

    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_nd_shard_shape, grid=standard_shard_core_grid, orientation=output_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)

    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=False)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape, input_nd_shard_shape", [([2, 2, 256, 512], [2, 64, 64])])
@pytest.mark.parametrize(
    "input_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
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
    "output_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize(
    "num_shard_cores, standard_shard_core_grid, block_shard_core_grid",
    [
        (
            4,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
        ),
    ],
)
def test_untilize_single_core_nd_sharded_to_legacy_sharded(
    device,
    dtype,
    tensor_shape,
    input_nd_shard_shape,
    input_shard_orientation,
    output_shard_orientation,
    output_memory_layout,
    num_shard_cores,
    standard_shard_core_grid,
    block_shard_core_grid,
):
    torch.manual_seed(0)

    num_tensor_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_tensor_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[num_tensor_dims - 1]

    # Input ND shard spec
    input_nd_shard_shape = ttnn.Shape(input_nd_shard_shape)
    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_nd_shard_shape, grid=standard_shard_core_grid, orientation=input_shard_orientation
    )
    input_tensor_spec = ttnn.TensorSpec(
        shape=tensor_shape,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        nd_shard_spec=input_nd_shard_spec,
        buffer_type=ttnn.BufferType.L1,
    )

    # Output legacy shard shapes
    height_sharded_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_sharded_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    block_grid_size = block_shard_core_grid.bounding_box().grid_size()
    if output_shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        block_sharded_shard_shape = (
            tensor_height // block_grid_size.y,
            tensor_width // block_grid_size.x,
        )
    else:
        block_sharded_shard_shape = (
            tensor_height // block_grid_size.x,
            tensor_width // block_grid_size.y,
        )

    output_shard_memory_layout_map = {
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

    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, spec=input_tensor_spec, device=device)

    shard_info = output_shard_memory_layout_map[output_memory_layout]
    output_shard_spec = ttnn.ShardSpec(shard_info["shard_grid"], shard_info["shard_shape"], output_shard_orientation)
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor,
        memory_config=output_memory_config,
        use_multicore=False,
    )
    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape, input_nd_shard_shape", [([2, 2, 256, 512], [2, 64, 64])])
@pytest.mark.parametrize(
    "input_shard_orientation",
    [
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardOrientation.COL_MAJOR,
    ],
)
def test_untilize_single_core_nd_sharded_to_interleaved(
    device,
    dtype,
    tensor_shape,
    input_nd_shard_shape,
    input_shard_orientation,
):
    torch.manual_seed(0)

    # Input ND shard spec
    standard_shard_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))})
    input_nd_shard_shape = ttnn.Shape(input_nd_shard_shape)
    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_nd_shard_shape, grid=standard_shard_core_grid, orientation=input_shard_orientation
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

    # Output interleaved memory config
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=False)
    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape",
    [
        [2, 256, 512],
        [4128, 512],  # multiple blocks per core and a cliff core
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
    "num_shard_cores, shard_core_grid",
    [
        [
            4,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
        ],
        [
            16,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                }
            ),
        ],
    ],
)
@pytest.mark.parametrize("use_legacy_2D_shard_style", [True, False])
def test_untilize_multi_core_interleaved_to_nd_sharded(
    device,
    dtype,
    tensor_shape,
    output_shard_orientation,
    num_shard_cores,
    shard_core_grid,
    use_legacy_2D_shard_style,
):
    # Input memory config
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    if use_legacy_2D_shard_style:
        num_tensor_dims = len(tensor_shape)
        tensor_height = 1
        for i in range(num_tensor_dims - 1):
            tensor_height *= tensor_shape[i]
        tensor_width = tensor_shape[num_tensor_dims - 1]
        # ND shard shape (2D slice)
        nd_shard_shape = ttnn.Shape(
            [
                tensor_height // int(math.sqrt(num_shard_cores)),
                tensor_width // int(math.sqrt(num_shard_cores)),
            ]
        )
        # Output memory config with ND shard spec
        nd_shard_spec = ttnn.NdShardSpec(
            shard_shape=nd_shard_shape, grid=shard_core_grid, orientation=output_shard_orientation
        )

    else:
        # Output memory config with ND shard spec built via sharded_across_dims
        shard_dims = list(range(len(tensor_shape) - 2, len(tensor_shape)))  # shard last two dims
        tensor_spec = ttnn.TensorSpec(
            shape=tensor_shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1
        ).sharded_across_dims(shard_dims, shard_core_grid, output_shard_orientation)
        nd_shard_spec = tensor_spec.memory_config.nd_shard_spec
        assert nd_shard_spec is not None
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)
    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape",
    [
        [2, 64, 64],
        [1, 64, 64],
        [2, 32, 32],
        [2, 256, 512],
        [4, 4, 256, 512],
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
def test_untilize_multi_core_nd_sharded_to_interleaved(
    device,
    dtype,
    tensor_shape,
    input_shard_orientation,
    shard_core_grid,
):
    torch.manual_seed(0)
    # Build an ND shard spec by sharding the last two dimensions across the grid
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
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, shard_shape",
    [
        ([3, 128, 160], ttnn.Shape([2, 64, 64])),
        ([3, 160, 160], ttnn.Shape([2, 64, 64])),
        ([3, 192, 160], ttnn.Shape([2, 64, 64])),
        ([3, 192, 128], ttnn.Shape([2, 64, 64])),
        ([4, 128, 160], ttnn.Shape([3, 96, 96])),
        ([2, 4, 128, 160], ttnn.Shape([2, 3, 96, 96])),
        ([3, 160, 160], ttnn.Shape([3, 96, 96])),
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
def test_untilize_multi_core_nd_shard_to_interleaved_uneven_input_shard_spec(
    device,
    dtype,
    tensor_shape,
    shard_shape,
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

    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[4, 128, 128]])
@pytest.mark.parametrize(
    "input_shard_shape",
    [
        ttnn.Shape([3, 96, 96]),
    ],
)
@pytest.mark.parametrize(
    "output_shard_shape",
    [
        ttnn.Shape([2, 64, 64]),
        ttnn.Shape([2, 96, 96]),  # The following tests are for output unevenly sharded case
        ttnn.Shape([5, 96, 96]),
        ttnn.Shape([3, 20, 40]),
        ttnn.Shape([5, 20, 40]),
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
def test_untilize_multicore_nd_shard_to_nd_shard_spec_different_shard_specs(
    device,
    dtype,
    tensor_shape,
    input_shard_shape,
    output_shard_shape,
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
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[4, 128, 128]])
@pytest.mark.parametrize(
    "input_shard_shape",
    [
        ttnn.Shape([96, 96]),
    ],
)
@pytest.mark.parametrize(
    "output_shard_shape",
    [
        ttnn.Shape([2, 64, 64]),
        ttnn.Shape([2, 96, 96]),  # The following tests are for output unevenly sharded case
        ttnn.Shape([5, 96, 96]),
        ttnn.Shape([3, 20, 40]),
        ttnn.Shape([5, 20, 40]),
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
def test_untilize_multicore_nd_shard_to_nd_shard_spec_grid_2d_input_to_round_robin_1d_output(
    device,
    dtype,
    tensor_shape,
    input_shard_shape,
    output_shard_shape,
    input_shard_orientation,
    shard_core_grid,
):
    torch.manual_seed(0)
    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape,
        grid=shard_core_grid,
        orientation=input_shard_orientation,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.GRID_2D,
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
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[4, 128, 128]])
@pytest.mark.parametrize(
    "input_shard_shape",
    [
        ttnn.Shape([96, 96]),
    ],
)
@pytest.mark.parametrize(
    "output_shard_shape",
    [
        ttnn.Shape([160, 40]),
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
def test_untilize_multicore_nd_shard_to_nd_shard_spec_different_shard_specs_grid_2d_input_to_grid_2d_output(
    device,
    dtype,
    tensor_shape,
    input_shard_shape,
    output_shard_shape,
    input_shard_orientation,
    shard_core_grid,
):
    torch.manual_seed(0)
    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape,
        grid=shard_core_grid,
        orientation=input_shard_orientation,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.GRID_2D,
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
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[4, 128, 128]])
@pytest.mark.parametrize(
    "input_shard_shape",
    [
        ttnn.Shape([3, 96, 96]),
    ],
)
@pytest.mark.parametrize(
    "output_shard_shape",
    [
        ttnn.Shape([160, 40]),
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
def test_untilize_multicore_nd_shard_round_robin_input_to_grid_2d_output(
    device,
    dtype,
    tensor_shape,
    input_shard_shape,
    output_shard_shape,
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
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[2, 96, 128]])
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
    "output_shard_shape",
    [
        ttnn.Shape([32, 128]),
        ttnn.Shape([96, 128]),
        ttnn.Shape([1, 96, 128]),
        ttnn.Shape([2, 96, 128]),
        ttnn.Shape([64, 128]),
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
def test_untilize_multicore_nd_shard_to_nd_shard_spec_different_shard_specs_shard_shape_flattened(
    device,
    dtype,
    tensor_shape,
    input_shard_shape,
    output_shard_shape,
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
    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[4, 128, 128]])
@pytest.mark.parametrize(
    "input_shard_shape",
    [
        ttnn.Shape([3, 96, 96]),
    ],
)
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
def test_untilize_multicore_nd_shard_to_legacy_shard(
    device,
    dtype,
    tensor_shape,
    input_shard_shape,
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
    num_tensor_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_tensor_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[num_tensor_dims - 1]

    # Shard shapes
    height_sharded_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_sharded_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    block_sharded_shard_shape = (
        tensor_height // int(math.sqrt(num_shard_cores)),
        tensor_width // int(math.sqrt(num_shard_cores)),
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

    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[4, 128, 128]])
@pytest.mark.parametrize("input_shard_shape", [ttnn.Shape([3, 96, 96])])
@pytest.mark.parametrize(
    "output_memory_layout, output_shard_shape, output_shard_core_grid",
    [
        (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (192, 128),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 2))}),
        ),
        (
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (512, 96),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (192, 96),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2))}),
        ),
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
def test_untilize_multicore_nd_shard_to_legacy_shard_uneven_output(
    device,
    dtype,
    tensor_shape,
    input_shard_shape,
    output_memory_layout,
    output_shard_shape,
    output_shard_core_grid,
    output_shard_orientation,
    input_shard_orientation,
):
    torch.manual_seed(0)

    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape,
        grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 2))}),
        orientation=input_shard_orientation,
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

    output_shard_spec = ttnn.ShardSpec(output_shard_core_grid, output_shard_shape, output_shard_orientation)
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("tensor_shape", [[4, 128, 128]])
@pytest.mark.parametrize(
    "output_nd_shard_shape",
    [
        ttnn.Shape([2, 64, 64]),
    ],
)
@pytest.mark.parametrize(
    "input_memory_layout",
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
def test_untilize_multicore_legacy_shard_to_nd_shard(
    device,
    dtype,
    tensor_shape,
    output_nd_shard_shape,
    input_memory_layout,
    output_shard_orientation,
    input_shard_orientation,
    num_shard_cores,
    standard_shard_core_grid,
    block_shard_core_grid,
):
    torch.manual_seed(0)
    shard_core_grid = standard_shard_core_grid
    if input_memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        shard_core_grid = block_shard_core_grid

    num_tensor_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_tensor_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[num_tensor_dims - 1]

    # Shard shapes for input legacy sharding
    height_sharded_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_sharded_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    block_sharded_shard_shape = (
        tensor_height // int(math.sqrt(num_shard_cores)),
        tensor_width // int(math.sqrt(num_shard_cores)),
    )

    input_shard_memory_layout_map = {
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

    # Input memory config (legacy sharding)
    input_shard_memory_layout = input_shard_memory_layout_map[input_memory_layout]
    input_shard_spec = ttnn.ShardSpec(
        input_shard_memory_layout["shard_grid"], input_shard_memory_layout["shard_shape"], input_shard_orientation
    )
    input_memory_config = ttnn.MemoryConfig(input_memory_layout, ttnn.BufferType.L1, input_shard_spec)

    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=input_memory_config,
    )

    # Output ND shard spec
    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_nd_shard_shape, grid=shard_core_grid, orientation=output_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)

    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


def untilize_nd_shard_spec_to_same_shard_spec_test_helper(
    device, shape, dtype, core_start, core_end, shard_across_dims
):
    """
    Test untilize with ND shard spec.
    """
    torch.manual_seed(0)
    core_ranges = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(core_start), ttnn.CoreCoord(core_end))})

    nd_spec = ttnn.TensorSpec(
        shape=shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1
    ).sharded_across_dims(shard_across_dims, core_ranges)

    torch_tensor = torch.randn(shape, dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(torch_tensor, spec=nd_spec, device=device)

    untilized_tensor = ttnn.untilize(ttnn_tensor)
    assert_equal(torch_tensor, ttnn.to_torch(untilized_tensor))


@pytest.mark.parametrize("shape", [[4, 512, 768]])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    ("core_start", "core_end"),
    [((0, 0), (1, 3)), ((0, 0), (1, 2)), ((0, 0), (3, 3)), ((0, 0), (4, 4)), ((1, 1), (2, 3))],
)
@pytest.mark.parametrize("shard_across_dims", [[0, 1], [0, 1, 2], [1, 2]])
def test_untilize_nd_shard_spec_to_same_shard_spec3D(device, shape, dtype, core_start, core_end, shard_across_dims):
    untilize_nd_shard_spec_to_same_shard_spec_test_helper(device, shape, dtype, core_start, core_end, shard_across_dims)


@pytest.mark.parametrize("shape", [[64, 3, 256, 256]])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    ("core_start", "core_end"),
    [
        ((0, 0), (7, 7)),
        ((0, 0), (7, 6)),
    ],
)
@pytest.mark.parametrize("shard_across_dims", [[0, 1], [0, 1, 2], [0, 1, 2, 3]])
def test_untilize_nd_shard_spec_to_same_shard_spec4D(device, shape, dtype, core_start, core_end, shard_across_dims):
    untilize_nd_shard_spec_to_same_shard_spec_test_helper(device, shape, dtype, core_start, core_end, shard_across_dims)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "tensor_shape, shard_shape",
    [
        ([3, 128, 160], ttnn.Shape([2, 64, 64])),
        ([3, 160, 160], ttnn.Shape([2, 64, 64])),
        ([3, 192, 160], ttnn.Shape([2, 64, 64])),
        ([3, 192, 128], ttnn.Shape([2, 64, 64])),
        ([4, 128, 160], ttnn.Shape([3, 96, 96])),
        ([2, 4, 128, 160], ttnn.Shape([2, 3, 96, 96])),
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
def test_untilize_nd_shard_to_same_shard_spec_uneven_input_shard_spec(
    device,
    dtype,
    tensor_shape,
    shard_shape,
    input_shard_orientation,
    shard_core_grid,
):
    torch.manual_seed(0)
    tensor_spec = ttnn.TensorSpec(
        shape=tensor_shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1
    ).sharded(shard_shape, shard_core_grid, orientation=input_shard_orientation)

    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, spec=tensor_spec, device=device)

    ttnn_output_tensor = ttnn.untilize(input_ttnn_tensor)

    assert_equal(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor))


# --- Codegen-path coverage ---
#
# ttnn.untilize routes gate-supported cases to codegen and the rest to native, and offers no way to
# ask for one: the verification-only entries below live in the private module for that reason (see
# untilize_force.hpp). The nightly routing suite only asserts the *rejected* cases fall back to
# native, so nothing there fails if a codegen kernel itself breaks -- these pin codegen and compare
# it against native on the same input. That comparison is exact for every dtype below because
# untilize only relayouts values, so any mismatch is a real kernel bug rather than tolerance.
#
# One case per writer the program factory can pick (untilize_codegen_supported.cpp decides which
# shapes are in scope at all):
#   multi-tile-row, tile-aligned -> row-parallel writer, one tile row per core
#   single tile-row, Wt > 1      -> column-parallel writer, tile columns split across the grid
#   non-tile-aligned bfloat16    -> row-parallel writer's unpadding path, which skips pad rows
# bfloat8_b appears only tile-aligned: the gate routes non-aligned bfloat8_b to native because
# the reference casts it to bfloat16 first, a step this implementation does not have.
codegen_supported_cases = [
    # (tensor_shape, dtype, output_buffer_type)
    ([2, 2, 64, 128], ttnn.bfloat16, ttnn.BufferType.DRAM),
    ([2, 2, 64, 128], ttnn.bfloat8_b, ttnn.BufferType.DRAM),
    ([2, 2, 64, 128], ttnn.bfloat16, ttnn.BufferType.L1),
    ([32, 512], ttnn.bfloat16, ttnn.BufferType.DRAM),
    ([1, 2, 100, 68], ttnn.bfloat16, ttnn.BufferType.DRAM),
]

codegen_case_ids = [
    "row_parallel|bfloat16|dram",
    "row_parallel|bfloat8_b|dram",
    "row_parallel|bfloat16|l1",
    "column_parallel|bfloat16|dram",
    "unpadding|bfloat16|dram",
]


_force_native = ttnn._ttnn.operations.data_movement.untilize_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.untilize_force_codegen


def _codegen_input_tensor(device, tensor_shape, dtype):
    return ttnn.from_torch(
        torch.randn(tensor_shape, dtype=torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )


@pytest.mark.parametrize("tensor_shape, dtype, output_buffer_type", codegen_supported_cases, ids=codegen_case_ids)
def test_untilize_codegen(device, tensor_shape, dtype, output_buffer_type):
    torch.manual_seed(42)
    input_ttnn_tensor = _codegen_input_tensor(device, tensor_shape, dtype)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, output_buffer_type)

    golden = _force_native(input_ttnn_tensor, memory_config=output_memory_config)
    output = _force_codegen(input_ttnn_tensor, memory_config=output_memory_config)

    assert output.shape == golden.shape, f"Output shape {output.shape} does not match native shape {golden.shape}"
    assert_equal(ttnn.to_torch(golden), ttnn.to_torch(output))


@pytest.mark.parametrize(
    "tensor_shape",
    [[2, 2, 64, 128], [32, 512]],
    ids=["row_parallel", "column_parallel"],
)
def test_pc_untilize_codegen(device, tensor_shape):
    torch.manual_seed(42)
    num_iters = 3
    input_tensors = [_codegen_input_tensor(device, tensor_shape, ttnn.bfloat16) for _ in range(num_iters)]
    goldens = [ttnn.to_torch(_force_native(tensor)) for tensor in input_tensors]

    for i in range(num_iters):
        with device.cache_entries_counter.measure():
            output = _force_codegen(input_tensors[i])

        assert_equal(goldens[i], ttnn.to_torch(output))
        if i == 0:
            base_count = device.cache_entries_counter.total
        else:
            assert device.cache_entries_counter.total == base_count, "program cache entries differ on same configs"


# The sub_core_grids factory shares its writer kernel
# (writer_unary_stick_layout_split_rows_interleaved_parallel_columns) with the
# parallelize-column factory. That kernel consumes (num_sticks / TILE_HEIGHT) *
# num_tiles_per_core tiles via wait_front, but reader/compute only push
# num_tiles_per_core tiles per core. ntiles_per_column > 1 violates this
# producer/consumer contract, so validate_on_program_cache_miss must reject it.
@pytest.mark.parametrize(
    "input_shape, num_cores",
    [
        ([1, 1, 64, 1024], 8),  # ntiles_per_column=2
        ([1, 1, 96, 512], 8),  # ntiles_per_column=3
    ],
)
def test_untilize_sub_core_grids_multi_tile_height_rejected(device, input_shape, num_cores, expect_error):
    """Tall tensors (ntiles_per_column > 1) with sub_core_grids must be rejected, not hung."""
    torch.manual_seed(0)
    input_torch = torch.randn(input_shape, dtype=torch.bfloat16)
    input_ttnn = ttnn.from_torch(
        input_torch,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    sub_core_grids = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    with expect_error(RuntimeError, "sub_core_grid untilize only supports"):
        ttnn.untilize(input_ttnn, use_multicore=True, sub_core_grids=sub_core_grids)


def test_untilize_sub_core_grids_single_tile_height(device):
    """Single-tile-height tensor (ntiles_per_column == 1) with sub_core_grids still works."""
    torch.manual_seed(0)
    input_shape = [1, 1, 32, 1024]  # height=32 = one tile row → ntiles_per_column=1
    num_cores = 8
    input_torch = torch.randn(input_shape, dtype=torch.bfloat16)
    input_ttnn = ttnn.from_torch(
        input_torch,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    sub_core_grids = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    output = ttnn.untilize(input_ttnn, use_multicore=True, sub_core_grids=sub_core_grids)
    assert_equal(input_torch, ttnn.to_torch(output))


@pytest.mark.parametrize("warmup", [False, True], ids=["cold", "warmup_then_constrain"])
@pytest.mark.parametrize("headroom_regime", ["shallower_cb_plan", "no_cb_plan_fits"])
def test_untilize_codegen_with_resident_l1_buffers(device, headroom_regime, warmup):
    """Regression: the codegen CB plan must be budgeted against LIVE L1 occupancy.

    Statically allocated CBs grow upward from the allocator's base L1 address while L1 tensors are
    allocated downward from the top of L1, so the space actually available to a program's CBs is
    lowest_occupied_compute_l1_address() - base, not the whole of L1. The codegen path originally
    planned against the whole of L1, which is only correct on an otherwise-idle device; with a
    model's weights/trace buffers resident in L1 it planned a CB region that overlapped them and
    ProgramImpl::validate_circular_buffer_region() aborted the run with

        Statically allocated circular buffers in program N clash with L1 buffers on core range ...

    rather than degrading to a shallower CB plan (or falling back to native). Seen end-to-end on
    Gemma-3-4B in trace mode; reproduced here at the op level by pinning a persistent interleaved
    L1 tensor before untilizing.

    Two regimes, both of which must simply produce the right answer:
      - shallower_cb_plan: enough headroom for the single-buffered codegen plan but not the
        double-buffered one, so the program factory must degrade the tier it picks.
      - no_cb_plan_fits: not even the single-buffered codegen plan fits, so the program factory
        must build the native-equivalent program instead of failing.

    This asserts only on the result, never on which implementation served it.

    warmup=True first untilizes with free L1 (caches a roomier plan), deallocates that
    output, then constrains L1. The CB-tier / Native-block-split is in the program-cache
    key, so the second call must miss and rebuild rather than clash.
    """
    torch.manual_seed(42)

    TILE_BYTES = 2048  # bfloat16 tile
    info = ttnn._ttnn.reports.get_device_info(device)

    # Wt wide enough that the whole-of-L1 budget selects a double-buffered CB plan, but still
    # inside the gate's wide-chunk threshold (2 * Wt * 2048 <= 800_000 => Wt <= 195). Two tile
    # rows keep this on the row-parallel writer, whose CB plan is sized by Wt.
    wt = 192
    height_tiles = 2
    single_buffer_bytes = 2 * wt * TILE_BYTES  # cb_in + cb_out, one slot each
    double_in_bytes = 3 * wt * TILE_BYTES  # 2x cb_in + 1x cb_out

    # cb_limit is exactly the whole-of-L1 budget the buggy plan used: worker L1 minus the base.
    if info.cb_limit < double_in_bytes:
        pytest.skip(f"needs at least {double_in_bytes} B of CB space, device offers {info.cb_limit} B")

    if headroom_regime == "shallower_cb_plan":
        # A headroom window that fits the single-buffer plan but NOT the double-buffered one, so
        # the only non-clashing codegen plan is unambiguous and an unbudgeted build provably
        # overruns it.
        headroom_target = single_buffer_bytes + 64 * 1024
        expected_headroom = (single_buffer_bytes, double_in_bytes)
    else:
        # Below every codegen tier, so the factory has to hand off to native. Still generous
        # enough for native's own blocked CBs, which it sizes against the same live L1 space.
        headroom_target = 256 * 1024
        if headroom_target >= single_buffer_bytes:
            pytest.skip("device L1 headroom cannot be driven below the smallest codegen CB plan")
        expected_headroom = (0, single_buffer_bytes)

    tiles_per_bank = (info.cb_limit - headroom_target) // TILE_BYTES
    if tiles_per_bank <= 0:
        pytest.skip("device L1 is too small to leave a meaningful headroom window")

    # Built before the L1 reservation so that any device-side tilize in from_torch runs with the
    # usual amount of L1 available; only the untilize under test should see the pressure.
    # untilize only relayouts values, so the input is its own golden.
    input_torch_tensor = torch.randn([1, 1, 32 * height_tiles, 32 * wt], dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    if warmup:
        warmup_output = ttnn.untilize(input_ttnn_tensor)
        assert_equal(input_torch_tensor, ttnn.to_torch(warmup_output))
        ttnn.deallocate(warmup_output)

    # Interleaved L1 spreads pages round-robin over every bank, so tiles_per_bank tiles per bank
    # pushes the lowest occupied L1 address down on all of them. Allocated directly on device: the
    # contents are irrelevant, only the occupancy matters.
    resident = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32 * tiles_per_bank, 32 * info.l1_num_banks]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    )
    try:
        # Guard against a vacuous pass: if the reservation did not land where intended there is no
        # L1 pressure left to regress against.
        actual_headroom = resident.buffer_address() - info.address_at_first_l1_cb_buffer
        low, high = expected_headroom
        assert low <= actual_headroom < high, (
            f"resident L1 buffer left {actual_headroom} B above the CB base; the {headroom_regime} "
            f"regime needs it in [{low}, {high})"
        )

        output = ttnn.untilize(input_ttnn_tensor)

        assert_equal(input_torch_tensor, ttnn.to_torch(output))
    finally:
        ttnn.deallocate(resident)


@pytest.mark.parametrize(
    "tensor_shape",
    [
        (1, 1, 32, 7328),
        (1, 1, 128, 7328),
        (1, 1, 64, 8192),
        (1, 1, 96, 7392),
        (1, 1, 160, 6304),
        (2, 1, 64, 7328),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_untilize_block_per_node_cb_size(device, tensor_shape, dtype):
    torch.manual_seed(42)
    dram_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    device.disable_and_clear_program_cache()
    device.enable_program_cache()

    keep_alive = []
    entries = None
    for i in range(2):
        torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
        tt_tiled = ttnn.from_torch(
            torch_input,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dram_cfg,
            device=device,
        )

        tt_rm = ttnn.untilize(tt_tiled, use_multicore=True)
        keep_alive += [tt_tiled, tt_rm]

        assert tt_rm.layout == ttnn.ROW_MAJOR_LAYOUT
        assert_equal(torch_input, ttnn.to_torch(tt_rm))

        if i == 0:
            entries = device.num_program_cache_entries()
            assert entries >= 1, "the first invocation should have populated the program cache"
        else:
            assert (
                device.num_program_cache_entries() == entries
            ), "untilize must reuse the cached program on a cache hit"

    device.disable_and_clear_program_cache()
