# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True, False])
@pytest.mark.parametrize("tensor_shape", [[2, 2, 256, 512]])
def test_untilize_single_core_interleaved_to_interleaved(device, dtype, use_pack_untilize, tensor_shape):
    # Input memory config
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Output memory config
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=output_memory_config, use_multicore=False, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True])
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
    use_pack_untilize,
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
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=output_memory_config, use_multicore=False, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True])
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
    use_pack_untilize,
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
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=output_memory_config, use_multicore=False, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True])
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
    use_pack_untilize,
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
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=output_memory_config, use_multicore=False, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True])
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
    use_pack_untilize,
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
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=output_memory_config, use_multicore=False, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True, False])
@pytest.mark.parametrize(
    "tensor_shape",
    [
        [2, 256, 512],
        [4128, 512],  # multiple blocks per core and a cliff core
    ],
)
def test_untilize_multi_core_interleaved_to_interleaved(device, dtype, use_pack_untilize, tensor_shape):
    # Input memory config
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Output memory config
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True])
@pytest.mark.parametrize(
    "tensor_shape",
    [
        [2, 256, 512],
        [4128, 512],  # multiple blocks per core and a cliff core
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
def test_untilize_multi_core_interleaved_to_sharded(
    device,
    dtype,
    use_pack_untilize,
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
    input_shard_memory_layout = shard_memory_layout_map[output_memory_layout]
    output_shard_spec = ttnn.ShardSpec(
        input_shard_memory_layout["shard_grid"], input_shard_memory_layout["shard_shape"], output_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    # Test
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True])
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
    use_pack_untilize,
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
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True, False])
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
    use_pack_untilize,
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
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True])
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
    use_pack_untilize,
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
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True, False])
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
    ],
)
@pytest.mark.parametrize("input_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("output_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_untilize_multi_core_sharded_to_sharded_different_shard_types_uneven_input_shard_spec(
    device,
    dtype,
    use_pack_untilize,
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
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True])
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
    use_pack_untilize,
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
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True, False])
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
    use_pack_untilize,
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
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True])
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
    use_pack_untilize,
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
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=memory_config, use_multicore=True, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True, False])
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
    use_pack_untilize,
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
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=memory_config, use_multicore=True, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True])
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
    use_pack_untilize,
    tensor_shape,
    input_buffer_type,
    output_buffer_type,
    input_memory_layout,
    output_memory_layout,
):
    if input_buffer_type == ttnn.BufferType.DRAM and input_memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        pytest.skip("Untilize multicore does not support input DRAM sharded")

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
    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True, use_pack_untilize=use_pack_untilize
    )

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)
