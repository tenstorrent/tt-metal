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

TILE_HEIGHT = 32

tile_batch_size = [1, 2, 3, 7, 8, 57, 127, 128]
row_batch_size = [1, 3, 96]

n_pages = [1, 2, 3, 4, 7, 16, 17, 124, 231, 1024, 2048]

tile_Y = [n * TILE_HEIGHT for n in n_pages]
row_Y = n_pages + [16384, 17000, 32768, 65535]

n_pages = [1, 2, 3, 6, 7, 8, 16, 17, 128, 231, 2048]

tile_X = [n * 32 for n in n_pages]
row_X = [n * 32 for n in n_pages]


def generate_input_shapes(batch_sizes, Y, X):
    input_shapes = []
    for W in batch_sizes:
        for Z in batch_sizes:
            for height in Y:
                for width in X:
                    input_shapes.append([W, Z, height, width])
    return input_shapes


tile_input_shapes = generate_input_shapes(tile_batch_size, tile_Y, tile_X)
row_input_shapes = generate_input_shapes(row_batch_size, row_Y, row_X)

parameters = {
    "all_gather_tile_non_sharded_single_bank_ring_bf16": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": tile_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.SINGLE_BANK,
        ],
        "all_gather_operation": ["all_gather"],
        "num_iters": [1],
    },
    "all_gather_tile_non_sharded_single_bank_ring_bf8": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": tile_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat8_b],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.SINGLE_BANK,
        ],
        "all_gather_operation": ["all_gather"],
        "num_iters": [1],
    },
    "all_gather_tile_non_sharded_single_bank_line_bf16": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": tile_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.SINGLE_BANK,
        ],
        "all_gather_operation": ["line_all_gather"],
        "num_iters": [1],
    },
    "all_gather_tile_non_sharded_single_bank_line_bf8": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": tile_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat8_b],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.SINGLE_BANK,
        ],
        "all_gather_operation": ["line_all_gather"],
        "num_iters": [1],
    },
    "all_gather_tile_non_sharded_interleaved_ring_bf16": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": tile_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.INTERLEAVED,
        ],
        "all_gather_operation": ["all_gather"],
        "num_iters": [1],
    },
    "all_gather_tile_non_sharded_interleaved_ring_bf8": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": tile_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat8_b],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.INTERLEAVED,
        ],
        "all_gather_operation": ["all_gather"],
        "num_iters": [1],
    },
    "all_gather_tile_non_sharded_interleaved_line_bf16": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": tile_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.INTERLEAVED,
        ],
        "all_gather_operation": ["line_all_gather"],
        "num_iters": [1],
    },
    "all_gather_tile_non_sharded_interleaved_line_bf8": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": tile_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat8_b],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.INTERLEAVED,
        ],
        "all_gather_operation": ["line_all_gather"],
        "num_iters": [1],
    },
    "all_gather_row_non_sharded_single_bank_ring_bf16": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": row_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.SINGLE_BANK,
        ],
        "all_gather_operation": ["all_gather"],
        "num_iters": [1],
    },
    "all_gather_row_non_sharded_single_bank_ring_bf8": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": row_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat8_b],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.SINGLE_BANK,
        ],
        "all_gather_operation": ["all_gather"],
        "num_iters": [1],
    },
    "all_gather_row_non_sharded_single_bank_line_bf16": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": row_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.SINGLE_BANK,
        ],
        "all_gather_operation": ["line_all_gather"],
        "num_iters": [1],
    },
    "all_gather_row_non_sharded_single_bank_line_bf8": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": row_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat8_b],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.SINGLE_BANK,
        ],
        "all_gather_operation": ["line_all_gather"],
        "num_iters": [1],
    },
    "all_gather_row_non_sharded_interleaved_ring_bf16": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": row_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.INTERLEAVED,
        ],
        "all_gather_operation": ["all_gather"],
        "num_iters": [1],
    },
    "all_gather_row_non_sharded_interleaved_ring_bf8": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": row_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat8_b],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.INTERLEAVED,
        ],
        "all_gather_operation": ["all_gather"],
        "num_iters": [1],
    },
    "all_gather_row_non_sharded_interleaved_line_bf16": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": row_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.INTERLEAVED,
        ],
        "all_gather_operation": ["line_all_gather"],
        "num_iters": [1],
    },
    "all_gather_row_non_sharded_interleaved_line_bf8": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": row_input_shapes,
        "dim": [0, 1, 2, 3],
        "tensor_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_dtype": [ttnn.bfloat8_b],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.INTERLEAVED,
        ],
        "all_gather_operation": ["line_all_gather"],
        "num_iters": [1],
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
    dim,
    num_links,
    input_dtype,
    tensor_layout,
    tensor_mem_layout,
    # num_cores,
    num_iters,
    all_gather_operation,
    *,
    device,
):
    t3k_mesh_device = device

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    input_tensor = torch.rand(input_shape).bfloat16()

    ttnn_tensor = ttnn.from_torch(input_tensor, mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=dim))
    input_tensor_mesh = ttnn.to_device(ttnn_tensor, t3k_mesh_device)

    ## Run the actual allgather operation
    gather_function = ttnn.all_gather if all_gather_operation == "all_gather" else ttnn.line_all_gather

    for i in range(num_iters):
        start_time = start_measuring_time()
        tt_out_tensor = gather_function(input_tensor_mesh, dim, num_links=num_links, memory_config=mem_config)
        e2e_perf = stop_measuring_time(start_time)

        logger.info(f"Done iteration {i}")

    for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
        tt_output_tensor = t.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        if input_dtype == ttl.tensor.DataType.BFLOAT16:
            eq, output = comp_equal(tt_output_tensor, input_tensor)
        else:
            eq, output = comp_pcc(tt_output_tensor, input_tensor)
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
        return [(eq, output), e2e_perf]
