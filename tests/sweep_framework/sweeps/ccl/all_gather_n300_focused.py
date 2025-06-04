# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc

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

rands = [0, 3, 7, 13, 97, 37]

for w in rands:
    for z in rands:
        for y in rands:
            for x in rands:
                input_shapes.append((w, z, y * TILE_HEIGHT + 17, x * TILE_WIDTH + 19))

rands = [1, 5, 13, 17, 97, 37]

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
        "num_iters": [1],
        "tile": [(32, 32)],
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
        "num_iters": [1],
        "tile": [(32, 32)],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    n_chips = 2
    max_tensor_size = int(1024 * 1024 * 1024 * 11 * (1 / (n_chips + 1)) / 2)
    input_tensor_size = (
        test_vector["input_shape"][0]
        * test_vector["input_shape"][1]
        * test_vector["input_shape"][2]
        * test_vector["input_shape"][3]
    )

    if input_tensor_size > max_tensor_size:
        return True, f"Not enough memory to allocate."
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
    num_iters,
    tile,
    *,
    device,
) -> list:
    all_devices = device

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"dim: {dim}")

    # for device in devices:
    #    device.disable_and_clear_program_cache()

    input_shape_list = list(input_shape)
    input_shape_list[dim] *= num_devices
    input_shape = tuple(input_shape_list)

    input_tensor = torch.rand(input_shape).bfloat16()

    input_tensors = torch.chunk(input_tensor, num_devices, dim)
    tt_input_tensors = []
    for i, t in enumerate(input_tensors):
        t = ttnn.from_torch(t, input_dtype, tile=ttnn.Tile(tile), layout=ttnn.Layout.TILE)
        t = t.to(all_devices[i], mem_config)
        tt_input_tensors.append(t)

    input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors)
    for i in range(num_iters):
        start_time = start_measuring_time()
        tt_out_tensor = ttnn.all_gather(input_tensor_mesh, dim, num_links=num_links, memory_config=mem_config)
        e2e_perf = stop_measuring_time(start_time)

        for d in all_devices:
            ttnn.synchronize_device(d)
        logger.info(f"Done iteration {i}")

    for i, t in enumerate(ttnn.get_device_tensors(tt_out_tensor)):
        tt_output_tensor = ttnn.to_torch(t)
        if input_dtype == ttnn.bfloat16:
            eq, output = comp_equal(tt_output_tensor, input_tensor)
        else:
            eq, output = comp_pcc(tt_output_tensor, input_tensor)
        if not eq:
            logger.error(f"output mismatch for tensor {i}")
        return [(eq, output), e2e_perf]
