# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
# TIMEOUT = 30

# random.seed(0)

# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "bitwise_right_shift_1": {
        # "input_shape": [{"self": [1, 1, 1024, 1024], "other": [1, 1, 1024, 1024]}],
        "input_shape": [{"self": [1, 1, 512, 512], "other": [1, 1, 512, 512]}],  # for float32 and int32 dtypes
        "input_a_dtype": [ttnn.int32],
        "input_b_dtype": [ttnn.int32],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_mem_config": [
            {"a_mem": "l1_interleaved", "b_mem": "l1_interleaved"},
            {"a_mem": "l1_interleaved", "b_mem": "dram_interleaved"},
            {"a_mem": "dram_interleaved", "b_mem": "l1_interleaved"},
            {"a_mem": "dram_interleaved", "b_mem": "dram_interleaved"},  # l1 - dram combination
            {"a_mem": "l1_height_sharded_rm", "b_mem": "l1_height_sharded_rm"},
            {"a_mem": "dram_interleaved", "b_mem": "l1_height_sharded_rm"},
            {"a_mem": "l1_height_sharded_rm", "b_mem": "dram_interleaved"},  # HS
            {"a_mem": "l1_width_sharded_rm", "b_mem": "l1_width_sharded_rm"},
            {"a_mem": "dram_interleaved", "b_mem": "l1_width_sharded_rm"},
            {"a_mem": "l1_width_sharded_rm", "b_mem": "dram_interleaved"},  # WS
            {"a_mem": "l1_block_sharded_rm", "b_mem": "l1_block_sharded_rm"},
            {"a_mem": "dram_interleaved", "b_mem": "l1_block_sharded_rm"},
            {"a_mem": "l1_block_sharded_rm", "b_mem": "dram_interleaved"},  # BS #row_major orientation
            {"a_mem": "l1_height_sharded_cm", "b_mem": "l1_height_sharded_cm"},
            {"a_mem": "dram_interleaved", "b_mem": "l1_height_sharded_cm"},
            {"a_mem": "l1_height_sharded_cm", "b_mem": "dram_interleaved"},  # HS
            {"a_mem": "l1_width_sharded_cm", "b_mem": "l1_width_sharded_cm"},
            {"a_mem": "dram_interleaved", "b_mem": "l1_width_sharded_cm"},
            {"a_mem": "l1_width_sharded_cm", "b_mem": "dram_interleaved"},  # WS
            {"a_mem": "l1_block_sharded_cm", "b_mem": "l1_block_sharded_cm"},
            {"a_mem": "dram_interleaved", "b_mem": "l1_block_sharded_cm"},
            {"a_mem": "l1_block_sharded_cm", "b_mem": "dram_interleaved"},  # BS #col_major orientation
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
            shape=(512 // 8, 512),
            # shape=(1024 // 8, 1024),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_height_sharded_cm":
        return ttnn.create_sharded_memory_config(
            shape=(512, 512 // 8),
            # shape=(1024, 1024 // 8),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_width_sharded_rm":
        return ttnn.create_sharded_memory_config(
            shape=(512, 512 // 8),
            # shape=(1024, 1024 // 8),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_width_sharded_cm":
        return ttnn.create_sharded_memory_config(
            shape=(512 // 8, 512),
            # shape=(1024 // 8, 1024),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_block_sharded_rm":
        return ttnn.create_sharded_memory_config(
            shape=(512 // 2, 512 // 4),
            # shape=(1024 // 2, 1024 // 4),
            core_grid=ttnn.CoreGrid(y=2, x=4),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_block_sharded_cm":
        return ttnn.create_sharded_memory_config(
            shape=(512 // 2, 512 // 4),
            # shape=(1024 // 2, 1024 // 4),
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
    input_a_dtype,
    input_b_dtype,
    input_a_layout,
    input_b_layout,
    input_mem_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.int32), input_a_dtype
    )(input_shape["self"])

    if isinstance(input_shape["other"], list):
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=0, high=32, dtype=torch.int32), input_b_dtype
        )(input_shape["other"])
    else:
        torch_input_tensor_b = torch.tensor(input_shape["other"], dtype=torch.int32)

    input_a_memory_config = input_mem_config["a_mem"]
    input_b_memory_config = input_mem_config["b_mem"]

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=return_mem_config(input_a_memory_config),
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=return_mem_config(input_b_memory_config),
    )

    golden_function = ttnn.get_golden_function(ttnn.experimental.bitwise_right_shift)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    start_time = start_measuring_time()
    result = ttnn.experimental.bitwise_right_shift(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)
    return [check_with_pcc(torch_output_tensor, output_tensor, pcc=0.99), e2e_perf]
