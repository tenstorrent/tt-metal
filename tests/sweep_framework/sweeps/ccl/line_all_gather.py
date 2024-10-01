# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from loguru import logger
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
    "suite_1": {
        "num_devices": [4, 8],
        "num_links": [1, 2],
        "input_shape": [
            [8, 1, 33, 256],
            [8, 1, 256, 32],
            [8, 8, 256, 384],
            [8, 5, 13, 512],
            [8, 5, 32, 512],
            [1, 1, 32, 16384],
        ],
        "dim": [0, 1, 3],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "mem_config": [ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)],
        "enable_async": [True, False],
        "num_iters": [1],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    (is_known_failure, message) = is_unsupported_case(
        test_vector["input_shape"],
        test_vector["dim"],
        test_vector["mem_config"],
        test_vector["num_devices"],
        test_vector["num_links"],
        test_vector["input_dtype"],
        test_vector["layout"],
    )
    if is_known_failure:
        return True, f"Skipping unsupported case {message}."
    if test_vector["num_links"] == 2:
        return True, f"test cases with num_links = 2 is currently not supported by new mesh fixture"
    return False, None


def mesh_device_fixture():
    assert ttnn.get_num_devices() >= 8, "Not T3000!"
    device_ids = ttnn.get_t3k_physical_device_ids_ring()
    num_devices_requested = len(device_ids)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, num_devices_requested), device_ids[:num_devices_requested])
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
    dim,
    num_links,
    input_dtype,
    layout,
    mem_config,
    enable_async,
    num_iters,
    *,
    device,
) -> list:
    t3k_mesh_device = device
    for device in t3k_mesh_device.get_devices():
        device.enable_async(enable_async)

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    input_tensor = torch.rand(input_shape).bfloat16()

    ttnn_tensor = ttnn.from_torch(input_tensor, mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=dim))
    input_tensor_mesh = ttnn.to_device(ttnn_tensor, t3k_mesh_device)

    for i in range(num_iters):
        start_time = start_measuring_time()
        tt_out_tensor = ttnn.all_gather(
            input_tensor_mesh, dim, num_links=num_links, memory_config=mem_config, topology=ttnn.Topology.Linear
        )
        e2e_perf = stop_measuring_time(start_time)

        logger.info(f"Done iteration {i}")

    for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
        tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        if input_dtype == ttnn.bfloat16:
            eq, output = comp_equal(tt_output_tensor, input_tensor)
        else:
            eq, output = comp_pcc(tt_output_tensor, input_tensor)
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
        return [(eq, output), e2e_perf]
