# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn
from models.utility_functions import get_devices_for_t3000
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from loguru import logger
import tt_lib as ttl
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.test_all_gather import (
    is_unsupported_case,
    generate_input_tensor_sharded,
    generate_config_sharded,
    generate_tt_tensors,
    compare_results,
)
from ttnn import ShardTensorToMesh

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.

height_shard_shapes = []
width_shard_shapes = []
block_shard_shapes = []

tile_batch_size = [1, 2, 3, 7, 8, 57, 127, 128]
row_batch_size = [1, 3, 96]

width_shard_Y = [1, 2, 3, 8, 7, 40, 128, 512]
height_shard_Y = [1, 2, 3, 4, 6, 7, 8, 16, 40, 31, 32, 33, 67]
block_shard_Y = height_shard_Y

width_shard_X = height_shard_Y
height_shard_X = width_shard_Y
block_shard_X = block_shard_Y


def generate_shard_input_shapes(batch_sizes, shard_Y, shard_X):
    input_shapes = []
    for batch_size in batch_sizes:
        for channels in batch_sizes:
            for height in shard_Y:
                for width in shard_X:
                    input_shapes.append([batch_size, channels, height * batch_size * channels, width])
    return input_shapes


tile_width_shard_input_shapes = generate_shard_input_shapes(tile_batch_size, width_shard_Y, width_shard_X)
row_width_shard_input_shapes = generate_shard_input_shapes(row_batch_size, width_shard_Y, width_shard_X)
tile_height_shard_input_shapes = generate_shard_input_shapes(tile_batch_size, height_shard_Y, height_shard_X)
row_height_shard_input_shapes = generate_shard_input_shapes(row_batch_size, height_shard_Y, height_shard_X)
tile_block_shard_input_shapes = generate_shard_input_shapes(tile_batch_size, block_shard_Y, block_shard_X)
row_block_shard_input_shapes = generate_shard_input_shapes(row_batch_size, block_shard_Y, block_shard_X)


for height in range(32, 128, 32):  # Increment by 32
    height_shard_shapes.append([height, 32])

for width in range(32, 1024, 32):  # Increment by 32
    width_shard_shapes.append([32, width])

for height in range(32, 128, 32):  # Increment by 32
    for width in range(32, 1024, 32):  # Increment by 32
        block_shard_shapes.append([height, width])

shard_grid = [
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7))}),
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 4))}),
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 4))}),
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7))}),
]

parameters = {
    "all_gather_tile_height_sharded": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": tile_height_shard_input_shapes,
        "input_shard_shape": height_shard_shapes,
        "shard_grid": shard_grid,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ],
        "all_gather_operation": ["all_gather", "line_all_gather"],
    },
    "all_gather_tile_width_sharded": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": tile_width_shard_input_shapes,
        "input_shard_shape": width_shard_shapes,
        "shard_grid": shard_grid,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ],
        "all_gather_operation": ["all_gather", "line_all_gather"],
    },
    "all_gather_tile_block_sharded": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": tile_block_shard_input_shapes,
        "input_shard_shape": block_shard_shapes,
        "shard_grid": shard_grid,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ],
        "all_gather_operation": ["all_gather", "line_all_gather"],
    },
    "all_gather_row_height_sharded": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": row_height_shard_input_shapes,
        "input_shard_shape": height_shard_shapes,
        "shard_grid": shard_grid,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ],
        "all_gather_operation": ["all_gather", "line_all_gather"],
    },
    "all_gather_row_width_sharded": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": row_width_shard_input_shapes,
        "input_shard_shape": width_shard_shapes,
        "shard_grid": shard_grid,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ],
        "all_gather_operation": ["all_gather", "line_all_gather"],
    },
    "all_gather_row_block_sharded": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": row_block_shard_input_shapes,
        "input_shard_shape": block_shard_shapes,
        "shard_grid": shard_grid,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ],
        "all_gather_operation": ["all_gather", "line_all_gather"],
    },
    "all_gather_non_sharded": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": [
            [1, 1, 32, 1024],
            [4, 1, 32, 128],
            [8, 1, 32, 1024],
            [4, 1, 32, 32],
            [1, 8, 32, 1024],
        ],
        "input_shard_shape": [
            [32, 1024],
            [32, 128],
            [32, 32],
        ],
        "shard_grid": [
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
        ],
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.SINGLE_BANK,
            ttnn.TensorMemoryLayout.INTERLEAVED,
        ],
        "all_gather_operation": ["all_gather", "line_all_gather"],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    unchunked_input_shape = list(test_vector["input_shape"])
    unchunked_input_shape[test_vector["dim"]] *= test_vector["num_devices"]
    if test_vector["num_devices"] < 2:
        return True, f"Requires multiple devices to run"

    if unchunked_input_shape[test_vector["dim"]] % test_vector["num_devices"] != 0 or (
        test_vector["dim"] == 3 and unchunked_input_shape[test_vector["dim"]] // test_vector["num_devices"] % 32 != 0
    ):
        return True, f"Unsupported test case"
    if test_vector["tensor_layout"] == ttnn.ROW_MAJOR_LAYOUT and test_vector["input_dtype"] == ttnn.bfloat8_b:
        return True, f"bfloat8_b/4_b only supports TILE layout"
    if test_vector["num_links"] == 2 and test_vector["num_devices"] == 8:
        return True, f"8 devices and 2 links are not supported on t3000 devices"
    return False, None


def device_mesh_fixture():
    import tt_lib as ttl

    assert ttnn.get_num_devices() >= 8, "Not T3000!"
    device_ids = [0, 4, 5, 1, 2, 6, 7, 3]
    num_devices_requested = len(device_ids)
    device_mesh = ttnn.open_device_mesh(ttnn.DeviceGrid(1, num_devices_requested), device_ids[:num_devices_requested])
    print("ALL GATHER: Opened device mesh")

    yield (device_mesh, "T3000 Mesh")

    print("ALL GATHER: Closing device mesh")
    for device in device_mesh.get_devices():
        ttl.device.DumpDeviceProfiler(device)
    ttnn.close_device_mesh(device_mesh)
    del device_mesh


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.


def run(
    num_devices,
    input_shape,
    input_shard_shape,
    shard_grid,
    dim,
    num_links,
    orientation,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    # num_cores,
    all_gather_operation,
    *,
    device,
):
    all_devices = device

    unchunked_input_shape, unchunked_input_tensor, input_tensors = generate_input_tensor_sharded(
        input_shape, num_devices, dim
    )

    # devices = get_devices_for_t3000(all_devices, num_devices)
    t3k_device = []

    for device in all_devices.get_devices():
        t3k_device.append(device)
    # devices = all_devices

    input_mem_config, output_mem_config = generate_config_sharded(
        input_shape,
        unchunked_input_shape,
        dim,
        num_devices,
        num_links,
        input_dtype,
        tensor_layout,
        tensor_mem_layout,
        orientation,
        shard_grid,
        input_shard_shape,
    )

    input_tensor_mesh = generate_tt_tensors(input_tensors, input_dtype, tensor_layout, devices, input_mem_config)

    ## Run the actual allgather operation
    gather_function = ttnn.all_gather if all_gather_operation == "all_gather" else ttnn.line_all_gather

    start_time = start_measuring_time()
    tt_out_tensor = gather_function(input_tensor_mesh, dim, num_links=num_links, memory_config=output_mem_config)
    e2e_perf = stop_measuring_time(start_time)

    all_eq, output = compare_results(input_dtype, tt_out_tensor, unchunked_input_tensor, input_shape)

    return [(all_eq, f"{i} FAILED: {output}"), e2e_perf]
