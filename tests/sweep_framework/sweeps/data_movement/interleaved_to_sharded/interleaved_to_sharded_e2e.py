# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 15
# seed for random
random.seed(0)

parameters = {
    "nightly": {
        "shard_specs": [
            {"shape": [1, 1, 32], "shard_shape": None},
        ],
        "strategy": [ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.HEIGHT],
        "orientation": [ttnn.ShardOrientation.ROW_MAJOR],
        "core_grid": [ttnn.CoreGrid(y=1, x=1)],
        "dtype": [ttnn.bfloat16],
        "layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_buffer_type": [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
        "output_buffer_type": [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],
    }
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        if test_vector["dtype"] == ttnn.bfloat8_b:
            return True, "bfloat8_b not supported with ROW_MAJOR_LAYOUT"

    return False, None


def run(
    shard_specs,
    strategy,
    orientation,
    core_grid,
    dtype,
    layout,
    input_buffer_type,
    output_buffer_type,
    *,
    device,
):
    device.enable_async(False)
    print("shard_specs:", shard_specs)
    # Ensure shard_specs is a dictionary
    if not isinstance(shard_specs, dict):
        print("shard_specs should be a dictionary")
        raise ValueError("shard_specs should be a dictionary")
    else:
        print("shard_specs is a dictionary")
    # Debug print to check the type and content of shard_specs
    print("Debug: shard_specs:", shard_specs)

    # Extract the shape specs from the input vector
    shape = shard_specs["shape"]
    shard_shape = shard_specs["shard_shape"]

    # Debug prints to check the values of the variables
    print("Debug: shape:", shape)
    print("Debug: shard_shape:", shard_shape)
    print("Debug: strategy:", strategy)
    print("Debug: orientation:", orientation)
    print("Debug: core_grid:", core_grid)

    # Prepare the shard config accordingly
    if shard_shape is None:
        shard_config = ttnn.create_sharded_memory_config(
            shape=shape,
            core_grid=core_grid,
            strategy=strategy,
            orientation=orientation,
            use_height_and_width_as_shard_shape=False,
        )
    else:
        shard_config = ttnn.create_sharded_memory_config(
            shape=shard_shape,
            core_grid=core_grid,
            strategy=strategy,
            orientation=orientation,
            use_height_and_width_as_shard_shape=True,
        )

    # Create a random tensor of the specified shape
    torch.manual_seed(0)
    input_data = torch.randn(shape, dtype=torch.bfloat16)
    interleaved_data = ttnn.from_torch(
        input_data,
        device=device,
        layout=layout,
        memory_config=input_buffer_type,
        dtype=ttnn.bfloat16,
    )

    # Measure performance of the split operation in ttnn
    start_time = start_measuring_time()

    sharded_data = ttnn.to_memory_config(interleaved_data, shard_config)
    interleaved_output = ttnn.to_memory_config(sharded_data, output_buffer_type)

    e2e_perf = stop_measuring_time(start_time)

    output_data = ttnn.from_device(interleaved_output)
    output_data = ttnn.to_torch(output_data)
    # Compare the concatenated tensors and return performance and accuracy check
    result = check_with_pcc(input_data, output_data, 0.999)

    return [result, e2e_perf]
