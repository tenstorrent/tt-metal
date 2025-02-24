# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "log-survey": {
        "input_shape": [{"self": [1, 1, 1024, 1024]}],
        "input_dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b],
        "input_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [
            "l1_interleaved",
            "dram_interleaved",
            "l1_height_sharded_rm",
            "l1_height_sharded_cm",
            "l1_width_sharded_rm",
            "l1_width_sharded_cm",
            "l1_block_sharded_rm",
            "l1_block_sharded_cm",
        ],
    },
}


def return_mem_config(mem_config_string):
    if mem_config_string == "l1_interleaved":
        return ttnn.L1_MEMORY_CONFIG
    elif mem_config_string == "dram_interleaved":
        return ttnn.DRAM_MEMORY_CONFIG
    elif mem_config_string == "l1_height_sharded_rm":
        return ttnn.create_sharded_memory_config(
            shape=(1024 // 8, 1024),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_width_sharded_rm":
        return ttnn.create_sharded_memory_config(
            shape=(1024, 1024 // 8),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_block_sharded_rm":
        return ttnn.create_sharded_memory_config(
            shape=(1024 // 2, 1024 // 4),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_height_sharded_cm":
        return ttnn.create_sharded_memory_config(
            shape=(1024, 1024 // 8),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_width_sharded_cm":
        return ttnn.create_sharded_memory_config(
            shape=(1024 // 8, 1024),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_block_sharded_cm":
        return ttnn.create_sharded_memory_config(
            shape=(1024 // 2, 1024 // 4),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    raise ("Input mem_config_string is not valid!")


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a device_mesh_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_dtype,
    input_layout,
    input_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=1, high=100, dtype=torch.float32), input_dtype
    )(input_shape["self"])

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=return_mem_config(input_memory_config),
    )

    torch_input_tensor = ttnn.to_torch(input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.log)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)
    start_time = start_measuring_time()
    result = ttnn.log(input_tensor)
    output_tensor = ttnn.to_torch(result)
    print("torch_output_tensor", torch_output_tensor)
    print("output_tensor", output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, pcc=0.999), e2e_perf]
