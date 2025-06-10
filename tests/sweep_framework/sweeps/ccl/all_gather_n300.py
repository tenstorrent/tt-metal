# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import is_unsupported_case_n300

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.

batch_sizes = [1, 2]
heights = [32, 64, 128]
widths = [32 * (2**i) for i in range(10)]  # [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
input_shapes = []
for W in batch_sizes:
    for Z in batch_sizes:
        for height in heights:
            for width in widths:
                input_shapes.append([W, Z, height, width])

parameters = {
    "all_gather_n300": {
        "num_devices": [2],
        "num_links": [1],
        "input_shape": input_shapes,
        "dim": [0, 1, 2, 3],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "mem_config": [
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1),
        ],
        "num_iters": [1],
        "tile": [(32, 32)],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    (is_known_failure, message) = is_unsupported_case_n300(
        test_vector["input_shape"],
        test_vector["dim"],
        test_vector["mem_config"],
        test_vector["num_devices"],
        test_vector["num_links"],
        test_vector["input_dtype"],
        test_vector["layout"],
        test_vector["tile"],
    )
    if is_known_failure:
        return True, f"Skipping unsupported case {message}."
    return False, None


def mesh_device_fixture():
    assert ttnn.get_num_devices() == 2, "Not N300!"
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, num_devices_requested))

    yield (mesh_device, "N300 Fixture")

    ttnn.close_mesh_device(mesh_device)


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
    num_iters,
    tile,
    *,
    device,
) -> list:
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    input_tensor = torch.rand(input_shape).bfloat16()

    input_tensors = torch.chunk(input_tensor, num_devices, dim)
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        tt_input_tensors.append(ttnn.Tensor(t, input_dtype, {}, ttnn.Tile(tile)).to(layout))

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors).to(device, mem_config)
    for i in range(num_iters):
        start_time = start_measuring_time()
        tt_out_tensor = ttnn.all_gather(input_tensor_mesh, dim, num_links=num_links, memory_config=mem_config)
        e2e_perf = stop_measuring_time(start_time)

        ttnn.synchronize_device(device)
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
