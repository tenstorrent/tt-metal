# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
from functools import partial

from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.pytorch_ops import (
    tilize_with_val_padding as pytorch_tilize_with_val_padding,
)

torch.manual_seed(0)

params = [
    pytest.param(
        [[1, 1, 50, 50]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            "output_tensor_shape": [1, 1, 64, 64],
            "pad_value": 10.0,
        },
    )
]

params += [
    pytest.param(
        [[1, 1, 50, 50]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            "output_tensor_shape": [1, 1, 64, 64],
            "pad_value": 10.0,
        },
    )
]

params += [
    pytest.param(
        [[1, 1, 50, 50]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            "output_tensor_shape": [1, 1, 64, 64],
            "pad_value": 50.0,
        },
    )
]

params += [
    pytest.param(
        [[1, 1, 50, 50]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            "output_tensor_shape": [1, 1, 64, 64],
            "pad_value": -18.0,
        },
    )
]


@pytest.mark.parametrize("input_shapes, tilize_with_val_padding_args", params)
def test_run_tilize_with_val_padding_test(input_shapes, tilize_with_val_padding_args, device, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
    ]
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test(
        "tilize_with_val_padding",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        tilize_with_val_padding_args,
    )


@pytest.mark.parametrize("input_shape", [(32, 15916), (16, 5210112), (48, 5210112), (180, 5210116)])
def test_run_tilize_large_row_input(device, input_shape):
    orig_shape = input_shape

    input = torch.randn(orig_shape, dtype=torch.bfloat16)
    halos = ttnn.from_torch(input, dtype=ttnn.bfloat16, device=device)
    halos_tile = ttnn.to_layout(halos, layout=ttnn.TILE_LAYOUT)
    halos_rm = ttnn.to_layout(halos_tile, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = ttnn.to_torch(halos_rm)
    assert_equal(input, output)


@pytest.mark.parametrize(
    "tensor_shape, input_shard_shape, output_padded_shape, output_shard_shape, shard_core_grid",
    [
        (
            [3, 50, 96],
            [2, 50, 96],
            [3, 64, 96],
            [1, 64, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            [3, 100, 158],
            [2, 64, 96],
            [3, 128, 160],
            [2, 96, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            [4, 100, 160],
            [2, 100, 160],
            [4, 128, 160],
            [2, 128, 160],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            [3, 100, 158],
            [2, 64, 96],
            [4, 128, 160],
            [3, 96, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            [5, 3, 100, 158],
            [4, 2, 64, 96],
            [8, 4, 128, 160],
            [5, 3, 96, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
        ),
    ],
)
@pytest.mark.parametrize("pad_value", [10.2, 0.0])
@pytest.mark.parametrize("input_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("output_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_tilize_with_val_padding_nd_sharded(
    device,
    tensor_shape,
    input_shard_shape,
    output_padded_shape,
    output_shard_shape,
    shard_core_grid,
    pad_value,
    input_shard_orientation,
    output_shard_orientation,
):
    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=input_shard_orientation
    )
    input_tensor_spec = ttnn.TensorSpec(
        shape=tensor_shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        nd_shard_spec=input_nd_shard_spec,
        buffer_type=ttnn.BufferType.L1,
    )

    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, spec=input_tensor_spec, device=device)

    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_shard_shape, grid=shard_core_grid, orientation=output_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)

    ttnn_output_tensor = ttnn.tilize_with_val_padding(
        input_ttnn_tensor,
        output_padded_shape,
        pad_value,
        memory_config=output_memory_config,
    )

    output_torch_tensor = ttnn_output_tensor.cpu().to_torch_with_padded_shape()  # ttnn.to_torch(ttnn_output_tensor)
    expected_torch_tensor = pytorch_tilize_with_val_padding(input_torch_tensor, output_padded_shape, pad_value)
    assert_equal(expected_torch_tensor, output_torch_tensor)


_ND_TO_INTERLEAVED_PARAMS = [
    (
        [3, 50, 96],
        [2, 50, 96],
        [3, 64, 96],
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
    ),
    (
        [3, 100, 158],
        [2, 64, 96],
        [3, 128, 160],
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
    ),
]


@pytest.mark.parametrize(
    "tensor_shape, input_shard_shape, output_padded_shape, shard_core_grid",
    _ND_TO_INTERLEAVED_PARAMS,
)
@pytest.mark.parametrize("pad_value", [10.2, 0.0])
@pytest.mark.parametrize("input_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_tilize_with_val_padding_nd_sharded_to_interleaved(
    device, tensor_shape, input_shard_shape, output_padded_shape, shard_core_grid, pad_value, input_shard_orientation
):
    """tilize_with_val_padding: nd_sharded input -> interleaved output."""
    torch.manual_seed(0)
    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=input_shard_orientation
    )
    input_tensor_spec = ttnn.TensorSpec(
        shape=tensor_shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        nd_shard_spec=input_nd_shard_spec,
        buffer_type=ttnn.BufferType.L1,
    )
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, spec=input_tensor_spec, device=device)

    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    ttnn_output_tensor = ttnn.tilize_with_val_padding(
        input_ttnn_tensor, output_padded_shape, pad_value, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn_output_tensor.cpu().to_torch_with_padded_shape()
    expected_torch_tensor = pytorch_tilize_with_val_padding(input_torch_tensor, output_padded_shape, pad_value)
    assert_equal(expected_torch_tensor, output_torch_tensor)


@pytest.mark.parametrize(
    "tensor_shape, output_padded_shape, output_shard_shape, shard_core_grid",
    [
        (
            [3, 50, 96],
            [3, 64, 96],
            [1, 64, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            [3, 100, 158],
            [3, 128, 160],
            [2, 96, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
    ],
)
@pytest.mark.parametrize("pad_value", [10.2, 0.0])
@pytest.mark.parametrize("output_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_tilize_with_val_padding_interleaved_to_nd_sharded(
    device, tensor_shape, output_padded_shape, output_shard_shape, shard_core_grid, pad_value, output_shard_orientation
):
    """tilize_with_val_padding: interleaved input -> nd_sharded output."""
    torch.manual_seed(0)
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_shard_shape, grid=shard_core_grid, orientation=output_shard_orientation
    )
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)
    ttnn_output_tensor = ttnn.tilize_with_val_padding(
        input_ttnn_tensor, output_padded_shape, pad_value, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn_output_tensor.cpu().to_torch_with_padded_shape()
    expected_torch_tensor = pytorch_tilize_with_val_padding(input_torch_tensor, output_padded_shape, pad_value)
    assert_equal(expected_torch_tensor, output_torch_tensor)


@pytest.mark.parametrize(
    "tensor_shape, output_padded_shape, output_shard_shape, shard_core_grid",
    [
        (
            [3, 100, 128],
            [4, 128, 160],
            [3, 96, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
    ],
)
@pytest.mark.parametrize("pad_value", [10.2])
@pytest.mark.parametrize(
    "input_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize("input_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "num_shard_cores, standard_shard_core_grid, block_shard_core_grid",
    [
        (
            2,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
    ],
)
def test_tilize_with_val_padding_legacy_sharded_to_nd_sharded(
    device,
    tensor_shape,
    output_padded_shape,
    output_shard_shape,
    shard_core_grid,
    pad_value,
    input_memory_layout,
    input_shard_orientation,
    num_shard_cores,
    standard_shard_core_grid,
    block_shard_core_grid,
):
    """tilize_with_val_padding: legacy 2D sharded input -> nd_sharded output."""
    torch.manual_seed(0)
    num_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[-1]

    height_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    block_shard_shape = (
        tensor_height // int(math.sqrt(num_shard_cores)),
        tensor_width // int(math.sqrt(num_shard_cores)),
    )
    shard_layout_map = {
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED: {
            "shard_grid": standard_shard_core_grid,
            "shard_shape": height_shard_shape,
        },
        ttnn.TensorMemoryLayout.WIDTH_SHARDED: {
            "shard_grid": standard_shard_core_grid,
            "shard_shape": width_shard_shape,
        },
        ttnn.TensorMemoryLayout.BLOCK_SHARDED: {"shard_grid": block_shard_core_grid, "shard_shape": block_shard_shape},
    }
    layout_info = shard_layout_map[input_memory_layout]
    input_shard_spec = ttnn.ShardSpec(layout_info["shard_grid"], layout_info["shard_shape"], input_shard_orientation)
    input_memory_config = ttnn.MemoryConfig(input_memory_layout, ttnn.BufferType.L1, input_shard_spec)

    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)

    output_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=output_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    output_memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd_shard_spec)
    ttnn_output_tensor = ttnn.tilize_with_val_padding(
        input_ttnn_tensor, output_padded_shape, pad_value, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn_output_tensor.cpu().to_torch_with_padded_shape()
    expected_torch_tensor = pytorch_tilize_with_val_padding(input_torch_tensor, output_padded_shape, pad_value)
    assert_equal(expected_torch_tensor, output_torch_tensor)


# nd_sharded -> legacy: use tile-aligned output (H,W multiple of 32) and grid matching layout.
# Physical output 2D = (3*64, 64) = (192, 64).
# height: 2 cores in column, shard (96, 64) -> 2 height shards
# width:  2 cores in row,    shard (192, 32) -> 2 width shards, full height per shard
# block:  2 rows x 1 col,    shard (96, 64)  -> 2 height shards, 1 width shard
_ND_TO_LEGACY_PARAMS = [
    (
        [3, 50, 64],
        [2, 50, 64],
        [3, 64, 64],
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        (96, 64),
    ),
    (
        [3, 50, 64],
        [2, 50, 64],
        [3, 64, 64],
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        (192, 32),
    ),
    (
        [3, 50, 64],
        [2, 50, 64],
        [3, 64, 64],
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        (96, 64),
    ),
]


@pytest.mark.parametrize(
    "tensor_shape, input_shard_shape, output_padded_shape, shard_core_grid, output_memory_layout, output_shard_shape_legacy",
    _ND_TO_LEGACY_PARAMS,
)
@pytest.mark.parametrize("pad_value", [10.2])
@pytest.mark.parametrize("input_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize("output_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
def test_tilize_with_val_padding_nd_sharded_to_legacy_sharded(
    device,
    tensor_shape,
    input_shard_shape,
    output_padded_shape,
    shard_core_grid,
    output_memory_layout,
    output_shard_shape_legacy,
    pad_value,
    input_shard_orientation,
    output_shard_orientation,
):
    """tilize_with_val_padding: nd_sharded input -> legacy 2D sharded output."""
    torch.manual_seed(0)
    output_shard_spec = ttnn.ShardSpec(shard_core_grid, output_shard_shape_legacy, output_shard_orientation)
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)

    input_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=input_shard_orientation
    )
    input_tensor_spec = ttnn.TensorSpec(
        shape=tensor_shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        nd_shard_spec=input_nd_shard_spec,
        buffer_type=ttnn.BufferType.L1,
    )
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, spec=input_tensor_spec, device=device)

    ttnn_output_tensor = ttnn.tilize_with_val_padding(
        input_ttnn_tensor, output_padded_shape, pad_value, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn_output_tensor.cpu().to_torch_with_padded_shape()
    expected_torch_tensor = pytorch_tilize_with_val_padding(input_torch_tensor, output_padded_shape, pad_value)
    assert_equal(expected_torch_tensor, output_torch_tensor)


@pytest.mark.parametrize(
    "tensor_shape, output_padded_shape",
    [
        # Input smaller than output (padding in both dims). Shard row size must be multiple of 8 (16-byte alignment).
        ([3, 100, 128], [3, 128, 160]),
    ],
)
@pytest.mark.parametrize("pad_value", [10.2])
@pytest.mark.parametrize(
    "input_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize("input_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
@pytest.mark.parametrize(
    "num_shard_cores, standard_shard_core_grid, block_shard_core_grid",
    [
        (
            2,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
    ],
)
def test_tilize_with_val_padding_legacy_sharded_to_interleaved(
    device,
    tensor_shape,
    output_padded_shape,
    pad_value,
    input_memory_layout,
    input_shard_orientation,
    num_shard_cores,
    standard_shard_core_grid,
    block_shard_core_grid,
):
    """tilize_with_val_padding: legacy 2D sharded input -> interleaved output."""
    torch.manual_seed(0)
    num_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[-1]

    height_shard_shape = (tensor_height // num_shard_cores, tensor_width)
    width_shard_shape = (tensor_height, tensor_width // num_shard_cores)
    block_shard_shape = (
        tensor_height // int(math.sqrt(num_shard_cores)),
        tensor_width // int(math.sqrt(num_shard_cores)),
    )
    shard_layout_map = {
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED: {
            "shard_grid": standard_shard_core_grid,
            "shard_shape": height_shard_shape,
        },
        ttnn.TensorMemoryLayout.WIDTH_SHARDED: {
            "shard_grid": standard_shard_core_grid,
            "shard_shape": width_shard_shape,
        },
        ttnn.TensorMemoryLayout.BLOCK_SHARDED: {"shard_grid": block_shard_core_grid, "shard_shape": block_shard_shape},
    }
    layout_info = shard_layout_map[input_memory_layout]
    input_shard_spec = ttnn.ShardSpec(layout_info["shard_grid"], layout_info["shard_shape"], input_shard_orientation)
    input_memory_config = ttnn.MemoryConfig(input_memory_layout, ttnn.BufferType.L1, input_shard_spec)

    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    input_ttnn_tensor = ttnn.to_device(input_ttnn_tensor, device, memory_config=input_memory_config)

    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    ttnn_output_tensor = ttnn.tilize_with_val_padding(
        input_ttnn_tensor, output_padded_shape, pad_value, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn_output_tensor.cpu().to_torch_with_padded_shape()
    expected_torch_tensor = pytorch_tilize_with_val_padding(input_torch_tensor, output_padded_shape, pad_value)
    assert_equal(expected_torch_tensor, output_torch_tensor)


@pytest.mark.parametrize(
    "tensor_shape, output_padded_shape, output_memory_layout, output_shard_shape, output_shard_core_grid",
    [
        # height sharded: physical (3*128, 64) = (384, 64), 2 cores in column -> (192, 64) per core (tile-aligned)
        (
            [3, 100, 64],
            [3, 128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (192, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
        ),
        # width sharded: physical 2D is (3*128, 64) = (384, 64), 2 cores -> (384, 32) per core (tile-aligned)
        (
            [3, 100, 64],
            [3, 128, 64],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (384, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
    ],
)
@pytest.mark.parametrize("pad_value", [10.2])
@pytest.mark.parametrize("output_shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
def test_tilize_with_val_padding_interleaved_to_legacy_sharded(
    device,
    tensor_shape,
    output_padded_shape,
    output_memory_layout,
    output_shard_shape,
    output_shard_core_grid,
    pad_value,
    output_shard_orientation,
):
    """tilize_with_val_padding: interleaved input -> legacy 2D sharded output."""
    torch.manual_seed(0)
    input_torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_ttnn_tensor = ttnn.from_torch(
        input_torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    output_shard_spec = ttnn.ShardSpec(output_shard_core_grid, output_shard_shape, output_shard_orientation)
    output_memory_config = ttnn.MemoryConfig(output_memory_layout, ttnn.BufferType.L1, output_shard_spec)
    ttnn_output_tensor = ttnn.tilize_with_val_padding(
        input_ttnn_tensor, output_padded_shape, pad_value, memory_config=output_memory_config
    )
    output_torch_tensor = ttnn_output_tensor.cpu().to_torch_with_padded_shape()
    expected_torch_tensor = pytorch_tilize_with_val_padding(input_torch_tensor, output_padded_shape, pad_value)
    assert_equal(expected_torch_tensor, output_torch_tensor)


@pytest.mark.parametrize(
    "dtype, scalar_val, pad_value",
    [
        (ttnn.bfloat16, 1.5, 0.0),
        (ttnn.bfloat16, 1.5, 42.0),
        (ttnn.bfloat16, 1.5, -32.5),
        (ttnn.float32, 1.5, 0.0),
        (ttnn.float32, 1.5, -0.0),
        (ttnn.float32, 1.5, 42.0),
        (ttnn.float32, 1.5, -32.5),
        (ttnn.int32, 7, 0),
        (ttnn.int32, 7, -32),
        (ttnn.int32, 7, -0),
        (ttnn.uint32, 7, 0),
        (ttnn.uint32, 7, 42),
    ],
)
def test_tilize_with_val_padding_scalar(device, dtype, scalar_val, pad_value):
    """tilize_with_val_padding: scalar (rank-0) input."""
    torch.manual_seed(0)
    torch_dtype = {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.int32: torch.int32,
        ttnn.uint32: torch.int32,
    }[dtype]

    input_torch_tensor = torch.tensor(scalar_val, dtype=torch_dtype)
    input_ttnn_tensor = ttnn.from_torch(input_torch_tensor, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    output_padded_shape = [32, 32]
    ttnn_output_tensor = ttnn.tilize_with_val_padding(input_ttnn_tensor, output_padded_shape, pad_value)
    output_torch_tensor = ttnn_output_tensor.cpu().to_torch_with_padded_shape()

    ref_input = input_torch_tensor.reshape(1, 1)
    expected_torch_tensor = pytorch_tilize_with_val_padding(ref_input, output_padded_shape, pad_value)
    expected_torch_tensor = expected_torch_tensor.to(output_torch_tensor.dtype)
    assert_equal(expected_torch_tensor, output_torch_tensor)


@pytest.mark.parametrize("use_multicore", [False, True])
def test_tilize_with_val_padding_fp32_truncation(device, use_multicore):
    """Regression test: FP32 must not be truncated to TF32 during tilize_with_val_padding (issue #39310)."""
    input_shape = [1, 1, 50, 50]
    output_shape = [1, 1, 64, 64]
    torch_input = torch.full(input_shape, 0.1, dtype=torch.float32)
    tt_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_tiled = ttnn.tilize_with_val_padding(tt_input, output_shape, 0.0, use_multicore=use_multicore)
    tt_output = ttnn.untilize(tt_tiled)
    torch_output = ttnn.to_torch(tt_output)
    assert torch.equal(torch_input, torch_output[..., :50, :50])


@pytest.mark.parametrize(
    "hw, kernel, stride, pad",
    [
        ((64, 64), (2, 2), (2, 2), 0),
        ((32, 32), (3, 3), (2, 2), 1),
    ],
)
def test_tilize_with_val_padding_tilize_after_avg_pool2d_sum_input_interleaved_rm_tensor_has_larger_padded_width_than_logical_width(
    device, hw, kernel, stride, pad
):
    """
    Tests avg_pool2d -> to_layout(TILE) -> multiply
    This isolates the to_layout(TILE) step on the avg_pool2d output.

    The key for this test is that the output from avg_pool2d, which is the input to to_layout, is a row-major interleaved tensor with a larger padded_shape width than logical_shape width.
    This test aims to test such a scenario where there is a mismatch between the padded_shape width and the logical_shape width for interleaved row major tensors.
    """
    h, w = hw
    kh, kw = kernel
    sh, sw = stride

    torch_input = torch.randn(1, 1, h, w, dtype=torch.bfloat16)
    ref = torch.nn.functional.avg_pool2d(
        torch_input, kernel_size=(kh, kw), stride=(sh, sw), padding=pad, count_include_pad=True
    )

    mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    x_tt = ttnn.from_torch(
        torch_input.reshape(h, w),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_cfg,
    )
    x_flat = ttnn.reshape(x_tt, [1, 1, h * w, 1], memory_config=mem_cfg)
    x_rm = ttnn.to_layout(x_flat, ttnn.ROW_MAJOR_LAYOUT, None, memory_config=None)

    y = ttnn.avg_pool2d(
        x_rm,
        1,
        h,
        w,
        1,
        [kh, kw],
        [sh, sw],
        [pad, pad],
        False,
        True,
        None,
        memory_config=mem_cfg,
        applied_shard_scheme=None,
        compute_kernel_config=None,
        reallocate_halo_output=False,
        config_tensor_in_dram=True,
    )

    # At this point, for the testcase with "(hw, kernel, stride, pad)" == "((64, 64), (2, 2), (2, 2), 0)", y is a row-major interleaved tensor with logical_shape [1, 1, 1024, 1] and padded_shape [1, 1, 1024, 16].

    y_torch_before_tile = ttnn.to_torch(y)

    y_tiled = ttnn.to_layout(
        y, ttnn.TILE_LAYOUT, None, memory_config=None
    )  # This should call tilize_with_val_padding internally

    y_torch_after_tile = ttnn.to_torch(y_tiled)

    assert_with_pcc(y_torch_before_tile, y_torch_after_tile, pcc=0.999)

    ref_flat = ref.reshape(-1)
    result_flat = y_torch_after_tile.reshape(-1)[: ref_flat.numel()]

    assert_with_pcc(ref_flat, result_flat, pcc=0.999)
