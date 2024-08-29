# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn
from models.utility_functions import get_devices_for_t3000
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from loguru import logger
import tt_lib as ttl
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.test_all_gather import is_unsupported_case
from ttnn import ShardTensorToMesh

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "line_all_gather": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": [
            # [1, 1, 32, 32], # throwing python segmentation fault
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
        "orientation": [ttnn.ShardOrientation.ROW_MAJOR],
        "tensor_mem_layout": [
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.TensorMemoryLayout.INTERLEAVED,
            ttnn.TensorMemoryLayout.SINGLE_BANK,
        ],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    # if test_vector["num_links"] == 2:
    #     return True, f"test cases with num_links = 2 is currently not supported by new mesh fixture"
    unchunked_input_shape = list(test_vector["input_shape"])
    unchunked_input_shape[test_vector["dim"]] *= test_vector["num_devices"]
    if test_vector["num_devices"] < 2:
        return True, f"Requires multiple devices to run"
    elif test_vector["num_devices"] == 2 and test_vector["num_links"] == 2:
        return True, f"Not enough links to run"

    if unchunked_input_shape[test_vector["dim"]] % test_vector["num_devices"] != 0 or (
        test_vector["dim"] == 3 and unchunked_input_shape[test_vector["dim"]] // test_vector["num_devices"] % 32 != 0
    ):
        return True, f"Unsupported test case"
    if test_vector["tensor_layout"] == ttnn.ROW_MAJOR_LAYOUT and test_vector["input_dtype"] == ttnn.bfloat8_b:
        return True, f"bfloat8_b/4_b only supports TILE layout"
    if test_vector["tensor_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, f"ROW_MAJOR_LAYOUT not supported"
    if test_vector["tensor_mem_layout"] == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        return True, f"Output mismatch"
    if test_vector["num_links"] == 2 and test_vector["num_devices"] == 8:
        return True, f"8 devices and 2 links are not supported"
    if test_vector["input_shard_shape"][1] != test_vector["input_shape"][-1]:
        return True, f"Shard width must be equal to last dim of shape"
    if test_vector["tensor_mem_layout"] == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        return True, f"BLOCK_SHARDED not supported"
    if test_vector["tensor_mem_layout"] == ttnn.TensorMemoryLayout.SINGLE_BANK:
        return True, f"SINGLE_BANK is not supported"
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
    *,
    device,
):
    all_devices = device

    numel = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * num_devices
    unchunked_input_shape = list(input_shape)
    unchunked_input_shape[dim] *= num_devices

    unchunked_input_tensor = torch.rand(unchunked_input_shape).bfloat16()

    debug = False
    if debug:
        tile_id = 0
        for w in range(unchunked_input_shape[0]):
            for z in range(unchunked_input_shape[1]):
                for y in range(0, unchunked_input_shape[2], 32):
                    for x in range(0, unchunked_input_shape[3], 32):
                        for yy in range(32):
                            for xx in range(32):
                                unchunked_input_tensor[w][z][y + yy][x + xx] = tile_id
                        tile_id += 1

    unchunked_input_tensor = unchunked_input_tensor.bfloat16()

    input_tensors = torch.chunk(unchunked_input_tensor, num_devices, dim)
    # devices = get_devices_for_t3000(all_devices, num_devices)
    t3k_device = []

    for device in all_devices.get_devices():
        t3k_device.append(device)
    # devices = all_devices

    # num_cores =
    # compute_grid_size = devices[0].compute_with_storage_grid_size()

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"unchunked_input_shape: {unchunked_input_shape}")
    logger.info(f"dim: {dim}")
    logger.info(f"num_devices: {num_devices}")
    logger.info(f"num_links: {num_links}")
    logger.info(f"input_dtype: {input_dtype}")
    logger.info(f"tensor_layout: {tensor_layout}")
    logger.info(f"tensor_mem_layout: {tensor_mem_layout}")
    logger.info(f"orientation: {orientation}")
    # logger.info(f"num_cores: {num_cores}")
    logger.info(f"shard_grid: {shard_grid}")
    logger.info(f"input_shard_shape: {input_shard_shape}")

    input_shard_spec = ttnn.ShardSpec(
        shard_grid,
        input_shard_shape,
        orientation,
        False,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)
    output_shard_shape = list(input_shard_shape)
    if dim == 3:
        output_shard_shape[1] *= num_devices
    else:
        output_shard_shape[0] *= num_devices
    output_shard_spec = ttnn.ShardSpec(
        shard_grid,
        output_shard_shape,
        orientation,
        False,
    )
    output_mem_config = ttnn.MemoryConfig(
        tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
    )

    tt_input_tensors_dups = []
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        tt_input_tensors_dups.append(ttnn.Tensor(t, input_dtype).to(tensor_layout).to(t3k_device[i], input_mem_config))
        tt_input_tensors.append(ttnn.Tensor(t, input_dtype).to(tensor_layout).to(t3k_device[i], input_mem_config))

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)

    ## Run the actual allgather operation
    start_time = start_measuring_time()
    tt_out_tensor = ttnn.line_all_gather(input_tensor_mesh, dim, num_links=num_links, memory_config=output_mem_config)
    e2e_perf = stop_measuring_time(start_time)

    torch.set_printoptions(sci_mode=False)
    all_eq = True
    reported_mismatch = False
    for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
        tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if input_dtype == ttnn.bfloat16:
            eq, output = comp_equal(tt_output_tensor, unchunked_input_tensor)
        else:
            eq, output = comp_pcc(tt_output_tensor, unchunked_input_tensor)
        if not eq:
            all_eq = False
            logger.error(f"output mismatch for tensor {i}")
            for w in range(input_shape[0]):
                for z in range(input_shape[1]):
                    for y in range(0, input_shape[2], 32):
                        for x in range(0, input_shape[3], 32):
                            xx = 0
                            yy = 0
                            # for yy in range(32):
                            #     for xx in range(32):
                            if tt_output_tensor[w, z, y + yy, x + xx] != unchunked_input_tensor[w, z, y + yy, x + xx]:
                                logger.error(
                                    f"mismatch at {w}, {z}, {y + yy}, {x + xx}: {tt_output_tensor[w, z, y + yy, x + xx]} != {unchunked_input_tensor[w, z, y + yy, x + xx]}"
                                )
                                # if not reported_mismatch:
                                #     reported_mismatch = True

    return [(all_eq, f"{i} FAILED: {output}"), e2e_perf]
