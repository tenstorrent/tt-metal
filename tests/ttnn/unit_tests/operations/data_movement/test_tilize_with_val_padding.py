# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import os
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc
from tests.ttnn.python_api_testing.sweep_tests.ttnn_pytorch_ops import (
    tilize_with_val_padding as pytorch_tilize_with_val_padding,
)
from models.common.utility_functions import is_blackhole, skip_for_blackhole

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

# pad_value == 0.0 exercises the zero-fill path in the row-major tilize_with_val_padding
# reader kernel (the scalar fill_l1_range store loop that also handles non-zero pads).
# Only the multicore reader is covered: the single-core reader path is a broken legacy
# path (garbage output on device, unsupported oversized reads on the simulator) and is
# never used in production, where tilize_with_val_padding always runs multicore.
params += [
    pytest.param(
        [[1, 1, 50, 50]],
        {
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            "output_tensor_shape": [1, 1, 64, 64],
            "pad_value": 0.0,
            "use_multicore": True,
        },
    )
]


@pytest.mark.parametrize("input_shapes, tilize_with_val_padding_args", params)
def test_run_tilize_with_val_padding_test(input_shapes, tilize_with_val_padding_args, device, function_level_defaults):
    shape = input_shapes[0]
    torch_input = (torch.rand(shape) * 200 - 100).to(torch.bfloat16)

    output_tensor_shape = tilize_with_val_padding_args["output_tensor_shape"]
    pad_value = tilize_with_val_padding_args["pad_value"]
    use_multicore = tilize_with_val_padding_args.get("use_multicore", True)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=tilize_with_val_padding_args["dtype"][0],
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=tilize_with_val_padding_args["input_mem_config"][0],
    )
    tt_output = ttnn.tilize_with_val_padding(
        tt_input,
        output_tensor_shape,
        pad_value,
        memory_config=tilize_with_val_padding_args["output_mem_config"],
        use_multicore=use_multicore,
    )
    torch_output = tt_output.cpu().to_torch_with_padded_shape()

    torch_golden = pytorch_tilize_with_val_padding(torch_input, output_tensor_shape, pad_value)
    assert_equal(torch_golden, torch_output)


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
            [3, 128, 160],
            [3, 96, 96],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        (
            [5, 3, 100, 158],
            [4, 2, 64, 96],
            [5, 3, 128, 160],
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
            [3, 128, 128],
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
        ([3, 100, 128], [3, 128, 128]),
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


# Only the multicore reader is covered: the single-core reader is a broken legacy path whose
# stale TensorAccessorArgs contract fails to JIT-compile on a cold cache, and it is never used
# in production (tilize_with_val_padding always runs multicore).
@pytest.mark.parametrize("use_multicore", [True])
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
def test_tilize_with_val_padding_tilize_after_avg_pool2d_sum(device, hw, kernel, stride, pad):
    """
    Tests avg_pool2d followed by to_layout(TILE) on the avg_pool2d output.
    This isolates and validates the to_layout(TILE) step on the avg_pool2d result against a PyTorch avg_pool2d reference.
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

    y_torch_before_tile = ttnn.to_torch(y)

    y_tiled = ttnn.to_layout(
        y, ttnn.TILE_LAYOUT, None, memory_config=None
    )  # This should call tilize_with_val_padding internally

    y_torch_after_tile = ttnn.to_torch(y_tiled)

    assert_with_pcc(y_torch_before_tile, y_torch_after_tile, pcc=0.999)

    ref_flat = ref.reshape(-1)
    result_flat = y_torch_after_tile.reshape(-1)[: ref_flat.numel()]

    assert_with_pcc(ref_flat, result_flat, pcc=0.999)


# Regression test for issue #51215 bug 4: column/row padding stores must handle a row byte size
# that is not a multiple of 4.  The reader's fill_l1_range<elem_size> helper does element-sized
# head/tail stores around the 4-byte-aligned interior; a naive 4-byte-only fill would clobber
# adjacent input bytes and leave the last 1-3 pad bytes unfilled.  Uses the multicore reader
# (the production path); the single-core reader is a broken legacy path (garbage output, and a
# stale TensorAccessorArgs contract that fails to JIT-compile) and is not exercised here.
@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        # bfloat16: row_bytes = width * 2; odd width → row_bytes % 4 == 2 (unaligned)
        ([1, 1, 32, 31], [1, 1, 32, 32]),
        ([1, 1, 32, 33], [1, 1, 32, 64]),
        ([1, 1, 64, 29], [1, 1, 64, 32]),
        # Partial height: exercises both column-pad and row-pad paths together
        ([1, 1, 30, 31], [1, 1, 32, 32]),
    ],
    ids=["bf16_w31", "bf16_w33", "bf16_w29", "bf16_w31_h30"],
)
def test_tilize_with_val_padding_unaligned_row_width(device, input_shape, output_shape):
    """tilize_with_val_padding with row bytes not a multiple of 4 (issue #51215 bug 4)."""
    pad_value = 7.0
    torch_input = (torch.rand(input_shape) * 200 - 100).to(torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    tt_output = ttnn.tilize_with_val_padding(tt_input, output_shape, pad_value, use_multicore=True)
    torch_output = tt_output.cpu().to_torch_with_padded_shape()

    torch_golden = pytorch_tilize_with_val_padding(torch_input, output_shape, pad_value)
    assert_equal(torch_golden, torch_output)


# FP8_E4M3 is a 1-byte-per-element format. A row width of 63 → 63 bytes/row, which is
# not a multiple of 4. This covers the head/tail element-sized stores in
# fill_l1_range<1> that handle the unaligned bytes at the start and end of each
# padding region; an incorrect 4-byte-only fill would corrupt adjacent real data.
# The output must be wider than a single tile (>32): the Blackhole 8-bit unpack-tilize
# path corrupts odd rows when the block is a single column tile (ct_dim == 1), so that
# case is rejected by the op guard (tt-llk narrow 8-bit tilize bug).
# Uses the multicore reader (production path); the single-core reader is broken legacy.
# Runs on Blackhole hardware only: it is a Blackhole-specific path, and the simulator does
# not implement the 8-bit tensix elementwise/unpack ops it exercises
# (UnimplementedFunctionality: tensix_elw_op src_fmt=10).
@pytest.mark.skipif(
    not is_blackhole() or bool(os.environ.get("TT_METAL_SIMULATOR")),
    reason="FP8_E4M3 tilize runs only on Blackhole hardware (not implemented in the simulator)",
)
def test_tilize_with_val_padding_fp8_unaligned_row_width(device):
    """FP8_E4M3 tilize_with_val_padding where the row byte size (width × 1) is not a
    multiple of 4: verifies that padding is written exactly and real data is preserved."""
    input_shape = [1, 1, 32, 63]
    output_shape = [1, 1, 32, 64]
    # Use a nonzero FP8-exact pad value so the padding assertion fails loudly if the
    # fill is reverted — 0.0 can pass silently when the CB happens to be zeroed.
    pad_value = 2.0

    torch_input = (torch.rand(input_shape) * 2 - 1).to(torch.float32)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    tt_tiled = ttnn.tilize_with_val_padding(tt_input, output_shape, pad_value, use_multicore=True)
    tt_untiled = ttnn.untilize(tt_tiled)
    torch_output = ttnn.to_torch(tt_untiled).float()

    # Real data region: use the exact device-quantised FP8 values as golden (converted to
    # float32 host-side via to_dtype, since single-device FP8→torch is unsupported). This
    # avoids the lossy float32→FP8→float32 round-trip that weakens detection of the
    # misaligned-store bug (which clobbers the last real byte of each row).
    torch_golden = ttnn.to_torch(ttnn.to_dtype(tt_input.cpu(), ttnn.float32)).float()
    assert_equal(torch_golden, torch_output[..., :63])
    # Padding column: FP8 2.0 (0x40) round-trips to exactly 2.0.
    assert torch.all(
        torch_output[..., 63:] == pad_value
    ), f"Expected pad_value={pad_value} in padding column, got unique values: {torch_output[..., 63:].unique()}"


def _assert_program_cache_reuse_across_new_allocations(
    device, tensor_shape, output_padded_shape, in_mem_cfg, out_mem_cfg, use_multicore
):
    """Run tilize_with_val_padding four times, allocating a fresh input and output every
    iteration so each dispatch sees a DIFFERENT buffer address, and assert that the results stay
    correct while the program cache is only ever populated once.

    A tensor address reaches this op's kernels one of two ways, and both are re-resolved on every
    cache hit rather than baked into the program: the sharded factory backs borrowed-memory
    dataflow buffers with the input and output shards (their L1 addresses are reattached per
    dispatch), and the interleaved factory carries the addresses on typed tensor bindings. A stale
    address on the second dispatch corrupts results silently, which a single-shot test cannot
    catch."""
    torch.manual_seed(0)
    pad_value = 3.5

    device.enable_program_cache()
    device.clear_program_cache()

    keep_alive = []  # retain prior tensors so each iteration allocates at a NEW address
    entries = None
    for i in range(4):
        torch_input = (torch.rand(tensor_shape) * 200 - 100).to(torch.bfloat16)
        tt_input = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem_cfg
        )
        tt_output = ttnn.tilize_with_val_padding(
            tt_input,
            ttnn.Shape(output_padded_shape),
            pad_value,
            memory_config=out_mem_cfg,
            use_multicore=use_multicore,
        )
        keep_alive += [tt_input, tt_output]

        readback = tt_output
        if tt_output.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            readback = ttnn.sharded_to_interleaved(
                tt_output, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
            )

        torch_golden = pytorch_tilize_with_val_padding(torch_input, output_padded_shape, pad_value)
        assert_equal(torch_golden, readback.cpu().to_torch_with_padded_shape())

        if i == 0:
            entries = device.num_program_cache_entries()
            assert entries >= 1, "the first invocation should have populated the program cache"
        else:
            # Not an exact count: the sharded case also runs from_torch / sharded_to_interleaved,
            # which cache programs of their own. What matters is that iteration 2+ adds none.
            assert (
                device.num_program_cache_entries() == entries
            ), "tilize_with_val_padding must reuse the cached program on a hit"


@skip_for_blackhole("BH LLK Issue with tilize, #14609")
def test_tilize_with_val_padding_program_cache_addr_change_sharded(device):
    """Sharded factory: the input and output shards back borrowed-memory dataflow buffers."""
    tensor_shape = (50, 256)
    num_cores = 4
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))}),
        [tensor_shape[0], tensor_shape[1] // num_cores],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    _assert_program_cache_reuse_across_new_allocations(
        device,
        tensor_shape=tensor_shape,
        output_padded_shape=[64, 256],
        in_mem_cfg=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec),
        # The op derives the output shard spec itself (height becomes the padded height), so pass
        # the layout only -- supplying a stale shard shape here would fight that.
        out_mem_cfg=ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer_type=ttnn.BufferType.L1
        ),
        use_multicore=True,
    )


# Only the multicore reader is covered here, matching the rest of this file: the single-core reader
# is a broken legacy path (garbage output for pad_value == 0) that production never selects, since
# tilize_with_val_padding always runs multicore. Its factory is ported, so the tensor-binding refresh
# it relies on is the same one this test exercises through the multicore path.
def test_tilize_with_val_padding_program_cache_addr_change_interleaved_multicore(device):
    """Multicore-default factory: addresses ride typed tensor bindings."""
    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    _assert_program_cache_reuse_across_new_allocations(
        device,
        tensor_shape=(1, 1, 100, 128),
        output_padded_shape=[1, 1, 128, 128],
        in_mem_cfg=dram,
        out_mem_cfg=dram,
        use_multicore=True,
    )
