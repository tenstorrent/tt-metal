# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.test_all_gather import is_unsupported_case_n300

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.

n_chips = 2
TILE_HEIGHT = 32
TILE_WIDTH = 32

input_shapes = [
    (1, 1, 1, 1),
    (1, 1, 1, 16),
    (1, 1, 1, 32),
    (1, 1, 32, 1),
    (1, 1, 32, 17),
    (1, 17, 1, 1),
    (1, 17, 1, 16),
    (1, 17, 1, 32),
    (1, 17, 32, 1),
    (1, 17, 32, 17),
    (17, 1, 1, 1),
    (17, 1, 1, 16),
    (17, 1, 1, 32),
    (17, 1, 32, 1),
    (17, 1, 32, 17),
    (17, 17, 17, 17),
]

rands = [0, 13, 97, 37]

for w in rands:
    for z in rands:
        for y in rands:
            for x in rands:
                if w != 0 and z != 0:
                    input_shapes.append((w, z, y * TILE_HEIGHT + 17, x * TILE_WIDTH + 19))

rands = [1, 13, 97, 37]

for w in rands:
    for z in rands:
        for y in rands:
            for x in rands:
                input_shapes.append((w, z, y * 17, x * 19))

max_tensor_size = int(1024 * 1024 * 1024 * 11 * (1 / (n_chips + 1)) / 2)
max_size = int(max_tensor_size / 2)
input_shapes_large = [
    (max_size, 1, 1, 1),
    (1, max_size, 1, 1),
    (1, 1, max_size, 1),
    (1, 1, 1, max_size),
    (1, 1, 32, max_size // 32),
    (1, 1, max_size // 32, 32),
]

rand_divisors = [1, 13, 97, 37, 101]

# Rotate the remainder through each dim
for z in rand_divisors:
    for y in rand_divisors:
        for x in rand_divisors:
            remaining_wzy = max_size // x
            remaining_wz = remaining_wzy // y
            remaining_w = remaining_wz // z
            input_shapes_large.append((remaining_w, z, y, x))


for w in rand_divisors:
    for y in rand_divisors:
        for x in rand_divisors:
            remaining_wzy = max_size // x
            remaining_wz = remaining_wzy // y
            remaining_z = remaining_wz // w
            input_shapes_large.append((w, remaining_z, y, x))


for w in rand_divisors:
    for z in rand_divisors:
        for x in rand_divisors:
            remaining_wzy = max_size // x
            remaining_wy = remaining_wzy // z
            remaining_y = remaining_wy // w
            input_shapes_large.append((w, z, remaining_y, x))


for w in rand_divisors:
    for z in rand_divisors:
        for y in rand_divisors:
            remaining_wzx = max_size // y
            remaining_wx = remaining_wzx // z
            remaining_x = remaining_wx // w
            input_shapes_large.append((w, z, y, remaining_x))

# input_shapes = [(shape[0], shape[1], shape[2] * 32, shape[3] * 32) for shape in input_shapes]
parameters = {
    "all_gather_n300_focused": {
        "num_devices": [2],
        "num_links": [1],
        "input_shape": input_shapes,
        "dim": [0, 1, 2, 3],
        "layout": [ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "mem_config": [
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ],
        "enable_async": [True],
        "num_iters": [1],
    },
    "all_gather_n300_focused_large": {
        "num_devices": [2],
        "num_links": [1],
        "input_shape": input_shapes_large,
        "dim": [0, 1, 2, 3],
        "layout": [ttnn.TILE_LAYOUT],
        "input_dtype": [ttnn.bfloat16],
        "mem_config": [
            ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        ],
        "enable_async": [True],
        "num_iters": [1],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    # (is_known_failure, message) = is_unsupported_case_n300(
    #     test_vector["input_shape"],
    #     test_vector["dim"],
    #     test_vector["mem_config"],
    #     test_vector["num_devices"],
    #     test_vector["num_links"],
    #     test_vector["input_dtype"],
    #     test_vector["layout"],
    # )
    # if is_known_failure:
    #     return True, f"Skipping unsupported case {message}."
    return False, None


def mesh_device_fixture():
    assert ttnn.get_num_devices() == 2, "Not N300!"

    num_devices = ttnn.GetNumAvailableDevices()
    device_ids = [i for i in range(num_devices)]

    devices = ttnn.CreateDevices(device_ids)

    yield ([devices[i] for i in range(num_devices)], "N300 Fixture")

    ttnn.close_device(devices[0])


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
    all_devices = device

    for device in all_devices:
        device.enable_async(enable_async)

    if enable_async:
        logger.info(f"Using Async Mode for All Gather Op Dispatch")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    # for device in devices:
    #    device.disable_and_clear_program_cache()

    input_tensor = torch.rand(input_shape).bfloat16()

    input_tensors = torch.chunk(input_tensor, num_devices, dim)
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        tt_input_tensors.append(ttnn.Tensor(t, input_dtype).to(layout).to(all_devices[i], mem_config))

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)
    for i in range(num_iters):
        start_time = start_measuring_time()
        tt_out_tensor = ttnn.all_gather(input_tensor_mesh, dim, num_links=num_links, memory_config=mem_config)
        e2e_perf = stop_measuring_time(start_time)

        for d in all_devices:
            ttnn.synchronize_device(d)
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
