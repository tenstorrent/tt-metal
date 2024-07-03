# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn
import time
from serialize import deserialize_vector

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type batches that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each batch has a key name (in this case "dram" and "l1") which will associate the test vectors to this specific batch of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "dram": {
        "batch_sizes": [(1,)],
        "height": [384, 1024],
        "width": [1024, 4096],
        "broadcast": [None, "h", "w", "hw"],
        "input_a_dtype": [ttnn.bfloat16],
        "input_b_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
    "l1": {
        "batch_sizes": [(1,)],
        "height": [1024, 1024],
        "width": [1024, 2048],
        "broadcast": [None, "h", "hw"],
        "input_a_dtype": [ttnn.bfloat16],
        "input_b_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_b_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be expected to fail.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector.broadcast in {"w", "hw"} and test_vector.input_b_layout == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Broadcasting along width is not supported for row major layout"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
def run(
    batch_sizes,
    height,
    width,
    broadcast,
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_b_memory_config,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    input_shape_a = (*batch_sizes, height, width)
    input_shape_b = (*batch_sizes, height, width)
    if broadcast == "hw":
        input_shape_b = (*batch_sizes, 1, 1)
    elif broadcast == "h":
        input_shape_b = (*batch_sizes, 1, width)
    elif broadcast == "w":
        input_shape_b = (*batch_sizes, height, 1)

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_b_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor, 0.999)
