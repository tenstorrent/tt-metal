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
TIMEOUT = 60

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.

TILE_HEIGHT = 32
TILE_WIDTH = 32

height_shard_shapes = []

tile_batch_size = [1, 2, 3, 7, 8, 57, 127, 128]
row_batch_size = [1, 3, 96]

width_shard_Y = [1, 2, 3, 8, 7, 40, 128, 512]
height_shard_Y = [1, 2, 3, 4, 6, 7, 8, 16, 40, 31, 32, 33, 67]

height_shard_X = width_shard_Y


def generate_shard_input_shapes(batch_sizes, shard_Y, shard_X, x=1, y=1):
    input_shapes = []
    for W in batch_sizes:
        for Z in batch_sizes:
            for height in shard_Y:
                for width in shard_X:
                    input_shapes.append([W, Z, height * x, width * y])
    input_shapes = input_shapes[:50] + input_shapes[-50:]
    return input_shapes


tile_height_shard_input_shapes = generate_shard_input_shapes(
    tile_batch_size, height_shard_Y, height_shard_X, x=TILE_HEIGHT, y=TILE_WIDTH
)
row_height_shard_input_shapes = generate_shard_input_shapes(row_batch_size, height_shard_Y, height_shard_X)

for height in range(32, 1024, 32):  # Increment by 32
    height_shard_shapes.append([height, 32])

shard_grid_map = {
    "shard_grid_1": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
    "shard_grid_2": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))}),
    "shard_grid_3": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
    "shard_grid_4": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
    "shard_grid_5": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
    "shard_grid_6": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3))}),
    "shard_grid_7": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
    "shard_grid_8": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
    "shard_grid_9": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
    "shard_grid_10": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 4))}),
    "shard_grid_11": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 4))}),
}


dtype_map = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b}

orientation_map = {"row_major": ttnn.ShardOrientation.ROW_MAJOR, "col_major": ttnn.ShardOrientation.COL_MAJOR}

gather_op_map = {"line": ttnn.Topology.Linear, "ring": ttnn.Topology.Ring}

tensor_layouts = {"row": ttnn.ROW_MAJOR_LAYOUT, "tile": ttnn.TILE_LAYOUT}

tensor_mem_layouts = {
    "height_sharded": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
}

input_shape_map = {
    (ttnn.TILE_LAYOUT, ttnn.TensorMemoryLayout.HEIGHT_SHARDED): tile_height_shard_input_shapes,
    (ttnn.ROW_MAJOR_LAYOUT, ttnn.TensorMemoryLayout.HEIGHT_SHARDED): row_height_shard_input_shapes,
}

input_shard_shape_map = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: height_shard_shapes,
}

dims = [0, 1, 2, 3]
num_devices = [4, 8]
num_links = [1, 2]


def generate_params(
    dtype_str,
    orientation_str,
    gather_op_str,
    tensor_layout_key,
    tensor_mem_layout_key,
    shard_grid_str,
    dim,
    num_device,
    num_link,
):
    dtype = dtype_map[dtype_str]
    orientation = orientation_map[orientation_str]
    gather_op = gather_op_map[gather_op_str]
    tensor_layout_value = tensor_layouts[tensor_layout_key]
    tensor_mem_layout_value = tensor_mem_layouts[tensor_mem_layout_key]
    shard_grid = shard_grid_map[shard_grid_str]

    # Retrieve the appropriate input shape based on tensor layout and tensor memory layout
    input_shape = input_shape_map.get((tensor_layout_value, tensor_mem_layout_value), tile_height_shard_input_shapes)
    input_shard_shape = input_shard_shape_map.get(tensor_mem_layout_value, height_shard_shapes)

    param_name = f"all_gather_{tensor_layout_key}_{tensor_mem_layout_key}_{gather_op_str}_{dtype_str}_{orientation_str}_dim{dim}_{num_device}devices_{num_link}links_{shard_grid_str}"
    return {
        "name": param_name,
        "config": {
            "num_devices": [num_device],
            "num_links": [num_link],
            "input_shape": input_shape,
            "input_shard_shape": input_shard_shape,
            "dim": [dim],
            "tensor_layout": [tensor_layout_value],
            "input_dtype": [dtype],
            "orientation": [orientation],
            "tensor_mem_layout": [tensor_mem_layout_value],
            "shard_grid": [shard_grid],
            "all_gather_topology": [gather_op],
        },
    }


parameters = {}

for dtype_str in dtype_map.keys():
    for orientation_str in orientation_map.keys():
        for gather_op_str in gather_op_map.keys():
            for tensor_layout_key in tensor_layouts:
                for tensor_mem_layout_key in tensor_mem_layouts:
                    for shard_grid_str in shard_grid_map.keys():
                        for dim in dims:
                            for num_device in num_devices:
                                for num_link in num_links:
                                    if num_link != 2 or num_device != 8:  # Not for T3K devices
                                        param = generate_params(
                                            dtype_str,
                                            orientation_str,
                                            gather_op_str,
                                            tensor_layout_key,
                                            tensor_mem_layout_key,
                                            shard_grid_str,
                                            dim,
                                            num_device,
                                            num_link,
                                        )
                                        parameters[param["name"]] = param["config"]


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


def mesh_device_fixture():
    assert ttnn.get_num_devices() >= 8, "Not T3000!"
    device_ids = ttnn.get_t3k_physical_device_ids_ring()
    num_devices_requested = len(device_ids)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, num_devices_requested), mesh_type=ttnn.MeshType.Ring)
    print("ALL GATHER: Opened device mesh")

    yield (mesh_device, "T3000 Mesh")

    print("ALL GATHER: Closing device mesh")
    for device in mesh_device.get_devices():
        ttnn.DumpDeviceProfiler(device)
    ttnn.close_mesh_device(mesh_device)
    del mesh_device


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
    all_gather_topology,
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

    input_tensor_mesh = generate_tt_tensors(input_tensors, input_dtype, tensor_layout, t3k_device, input_mem_config)

    ## Run the actual allgather operation

    start_time = start_measuring_time()
    tt_out_tensor = ttnn.all_gather(
        input_tensor_mesh, dim, num_links=num_links, memory_config=output_mem_config, topology=all_gather_topology
    )
    e2e_perf = stop_measuring_time(start_time)

    all_eq, output = compare_results(input_dtype, tt_out_tensor, unchunked_input_tensor, input_shape)

    return [(all_eq, output), e2e_perf]
