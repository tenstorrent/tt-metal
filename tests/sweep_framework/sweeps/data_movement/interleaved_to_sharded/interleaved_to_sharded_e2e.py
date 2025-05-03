# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn
import traceback

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 15
TILE_HEIGHT = TILE_WIDTH = 32
# seed for random
random.seed(0)

parameters = {
    "nightly": {
        "shard_specs": [
            {"shape": [1, 1, 1, 16], "shard_shape": None},
            {"shape": [1, 1, 32, 16], "shard_shape": None},
            {"shape": [1, 1, 16, 32], "shard_shape": None},
            {"shape": [1, 1, 128, 32], "shard_shape": None},
            {"shape": [1, 1, 64, 64], "shard_shape": None},
            {"shape": [1, 1, 128, 128], "shard_shape": None},
            {"shape": [1, 1, 1, 16], "shard_shape": [1, 1, 1, 16]},
            {"shape": [1, 1, 32, 16], "shard_shape": [1, 1, 16, 16]},
            {"shape": [1, 1, 16, 32], "shard_shape": [1, 1, 1, 16]},
            {"shape": [1, 1, 32, 32], "shard_shape": [1, 1, 16, 16]},
            {"shape": [1, 1, 64, 64], "shard_shape": [1, 1, 16, 16]},
            {"shape": [1, 1, 128, 128], "shard_shape": [1, 1, 32, 16]},
            {"shape": [1, 1, 128, 128], "shard_shape": [1, 1, 32, 32]},
        ],
        "strategy": [ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.HEIGHT],
        "orientation": [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
        "core_grid": [
            ttnn.CoreGrid(y=1, x=1),
            ttnn.CoreGrid(y=2, x=1),
            ttnn.CoreGrid(y=1, x=2),
            ttnn.CoreGrid(y=2, x=2),
        ],
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
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
    elif test_vector["layout"] == ttnn.TILE_LAYOUT:
        if test_vector["shard_specs"]["shard_shape"] is not None and (
            test_vector["shard_specs"]["shard_shape"][-2] % TILE_HEIGHT != 0
            or test_vector["shard_specs"]["shard_shape"][-1] % TILE_WIDTH != 0
        ):
            return True, "shard_shape not supported with TILE_LAYOUT"
        elif test_vector["shard_specs"]["shard_shape"] is None:
            ncores = test_vector["core_grid"].x * test_vector["core_grid"].y
            sizey = (
                test_vector["shard_specs"]["shape"][-2] // ncores
                if test_vector["strategy"] == ttnn.ShardStrategy.HEIGHT
                else test_vector["shard_specs"]["shape"][-2]
            )
            sizex = (
                test_vector["shard_specs"]["shape"][-1] // ncores
                if test_vector["strategy"] == ttnn.ShardStrategy.WIDTH
                else test_vector["shard_specs"]["shape"][-1]
            )
            if sizex % TILE_HEIGHT != 0 or sizey % TILE_WIDTH != 0:
                return True, "shard_shape not supported with TILE_LAYOUT"
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
    shape = shard_specs["shape"]
    shard_shape = shard_specs["shard_shape"]

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
        dtype=dtype,
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
