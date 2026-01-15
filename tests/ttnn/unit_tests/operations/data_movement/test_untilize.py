# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
import numpy as np
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
@pytest.mark.parametrize("use_pack_untilize", [True, False])
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
    use_pack_untilize,
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
    print(f"Output ND shard shape: {nd_shard_spec.shard_shape}, tensor shape: {tensor_shape}")
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
@pytest.mark.parametrize(
    "tensor_shape",
    [
        [2, 64, 64],
        [1, 64, 64],
        [2, 32, 32],
        # [2, 256, 512],
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
    "num_shard_cores, shard_core_grid",
    [
        [
            4,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))
                    # ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 2))
                    # ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
                }
            ),
        ],
        [
            4,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 2))}),
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
def test_untilize_multi_core_nd_sharded_to_interleaved(
    device,
    dtype,
    use_pack_untilize,
    tensor_shape,
    input_shard_orientation,
    num_shard_cores,
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

    input_memory_config = tensor_spec.memory_config
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, spec=tensor_spec, device=device)

    ttnn_output_tensor = ttnn.untilize(
        input_ttnn_tensor, memory_config=output_memory_config, use_multicore=True, use_pack_untilize=use_pack_untilize
    )

    # torch_np = input_torch_tensor.to(torch.float32).cpu().numpy().reshape(-1)
    # ttnn_np = ttnn.to_torch(ttnn_output_tensor).to(torch.float32).cpu().numpy().reshape(-1)
    # np.savetxt("torch.txt", torch_np, fmt="%.8f")
    # np.savetxt("ttnn.txt", ttnn_np, fmt="%.8f")

    #    def dump_faces(tensor: torch.Tensor, path: str):
    #        arr = tensor.to(torch.float32).cpu().numpy()
    #        if arr.ndim >= 3 and arr.shape[0] >= 2 and arr.shape[-2] == 64 and arr.shape[-1] == 64:
    #            with open(path, "w") as f:
    #             # np.savetxt(f, arr[0].reshape(64, 64), fmt="%.8f")
    #             #    f.write("\nSECOND FACE\n")
    #             #    np.savetxt(f, arr[1].reshape(64, 64), fmt="%.8f")
    #                for face_idx, face in enumerate([arr[0], arr[1]]):
    #                    mat = face.reshape(64, 64)
    #                    for row in mat:
    #                        formatted = [f"{v:.8f}" for v in row]
    #                        left = " ".join(formatted[:32])
    #                        right = " ".join(formatted[32:])
    #                        f.write(f"{left}   {right}\n")
    #                    if face_idx == 0:
    #                        f.write("\nSECOND FACE\n")
    #        else:
    #            np.savetxt(path, arr.reshape(-1), fmt="%.8f")

    #    dump_faces(input_torch_tensor, "torch_2.txt")
    #    dump_faces(ttnn.to_torch(ttnn_output_tensor), "ttnn_2.txt")

    assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("use_pack_untilize", [True, False])
@pytest.mark.parametrize(
    "tensor_shape, shard_shape",
    [
        ([3, 128, 160], ttnn.Shape([2, 64, 64])),
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
    ],
)
def test_untilize_multi_core_nd_shard_to_interleaved_uneven_input_shard_spec(
    device,
    dtype,
    use_pack_untilize,
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
    untilized = ttnn.untilize(tile_tensor, memory_config=output_memory_config, use_pack_untilize=True)
    result = ttnn.to_torch(untilized)

    assert torch.equal(result, torch_tensor), f"untilize lost FP32 precision"


@pytest.mark.xfail(
    reason="has bad precision with use_pack_untilize=False for fp32 because kernel uses FP32 #30400, #33795"
)
@pytest.mark.parametrize(
    "tensor_shape",
    [
        [32, 32],
        [1, 1, 512, 512],
        [2, 256, 512],
        [1, 1, 128, 7328],
    ],
)
def test_untilize_fp32_not_use_pack_untilize(device, tensor_shape):
    torch.manual_seed(42)
    torch_tensor = torch.rand(tensor_shape, dtype=torch.float32)

    tile_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    untilized = ttnn.untilize(tile_tensor, use_pack_untilize=False)
    result = ttnn.to_torch(untilized)

    assert torch.equal(result, torch_tensor), f"untilize lost FP32 precision"


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

        assert_with_pcc(input_torch_tensor, ttnn.to_torch(ttnn_output_tensor), 0.9999)


def untilize_nd_shard_spec_test_helper(
    device, shape, dtype, core_start, core_end, shard_across_dims, use_pack_untilize
):
    """
    Test untilize with ND shard spec.
    """
    torch.manual_seed(0)
    core_ranges = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(core_start), ttnn.CoreCoord(core_end))})

    nd_spec = ttnn.TensorSpec(
        shape=shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1
    ).sharded_across_dims(shard_across_dims, core_ranges)
    print("Shard shape:", nd_spec.memory_config.nd_shard_spec.shard_shape)

    torch_tensor = torch.randn(tuple(nd_spec.shape))
    ttnn_tensor = ttnn.from_torch(torch_tensor, spec=nd_spec, device=device)

    print("is_sharded():", ttnn_tensor.is_sharded())
    print("memory config:", ttnn_tensor.memory_config())
    print("shard spec:", ttnn_tensor.memory_config().shard_spec)
    print("ND shard spec:", ttnn_tensor.memory_config().nd_shard_spec)
    print("Attempting ttnn.untilize on ND sharded tensor...")

    untilized_tensor = ttnn.untilize(ttnn_tensor, use_pack_untilize=use_pack_untilize)
    print("Successfully untilized ND sharded tensor!")
    print("Output layout:", untilized_tensor.layout)
    assert_with_pcc(torch_tensor, ttnn.to_torch(untilized_tensor), 0.9999)


@pytest.mark.parametrize("shape", [[4, 512, 768]])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    ("core_start", "core_end"),
    [((0, 0), (1, 3)), ((0, 0), (1, 2)), ((0, 0), (3, 3)), ((0, 0), (4, 4)), ((1, 1), (2, 3))],
)
@pytest.mark.parametrize("shard_across_dims", [[0, 1], [0, 1, 2], [1, 2]])
@pytest.mark.parametrize("use_pack_untilize", [True, False])
def test_untilize_nd_shard_spec_3D(device, shape, dtype, core_start, core_end, shard_across_dims, use_pack_untilize):
    untilize_nd_shard_spec_test_helper(device, shape, dtype, core_start, core_end, shard_across_dims, use_pack_untilize)


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
@pytest.mark.parametrize("use_pack_untilize", [True, False])
def test_untilize_nd_shard_spec_4D(device, shape, dtype, core_start, core_end, shard_across_dims, use_pack_untilize):
    untilize_nd_shard_spec_test_helper(device, shape, dtype, core_start, core_end, shard_across_dims, use_pack_untilize)
