# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

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
    "ne_rm_2": {
        "input_shape": [{"self": [1, 1, 512, 512], "other": [1, 1, 512, 512]}],  # no bcast
        # "input_shape": [
        #     {"self": [4, 8, 64, 512], "other": [1, 8, 64, 1]},  # col_b, N_b #0.98
        #     {"self": [4, 8, 64, 512], "other": [4, 1, 1, 512]},  # row_b, C_b
        #     {"self": [4, 8, 64, 512], "other": [4, 8, 1, 1]},  # B scalar
        #     {"self": [1, 8, 64, 1], "other": [4, 8, 64, 512]},  # col_a, N_a
        #     {"self": [4, 1, 1, 512], "other": [4, 8, 64, 512]},  # row_a, C_a
        #     {"self": [4, 8, 1, 1], "other": [4, 8, 64, 512]},  # A scalar
        #     {"self": [4, 8, 1, 512], "other": [4, 8, 64, 1]},  # row_a, col_b
        #     {"self": [4, 8, 64, 1], "other": [4, 8, 1, 512]},  # row_b, col_a
        # ],  # bcast
        "input_dtype": [
            {"input_a_dtype": "ttnn.bfloat16", "input_b_dtype": "ttnn.bfloat16"},
            {"input_a_dtype": "ttnn.float32", "input_b_dtype": "ttnn.float32"},
            {"input_a_dtype": "ttnn.bfloat8_b", "input_b_dtype": "ttnn.bfloat8_b"},
            {"input_a_dtype": "ttnn.bfloat4_b", "input_b_dtype": "ttnn.bfloat4_b"},  # same dtype
            {"input_a_dtype": "ttnn.bfloat16", "input_b_dtype": "ttnn.float32"},
            {"input_a_dtype": "ttnn.bfloat16", "input_b_dtype": "ttnn.bfloat8_b"},
            {"input_a_dtype": "ttnn.bfloat16", "input_b_dtype": "ttnn.bfloat4_b"},
            {"input_a_dtype": "ttnn.float32", "input_b_dtype": "ttnn.bfloat16"},
            {"input_a_dtype": "ttnn.float32", "input_b_dtype": "ttnn.bfloat8_b"},
            {"input_a_dtype": "ttnn.float32", "input_b_dtype": "ttnn.bfloat4_b"},
            {"input_a_dtype": "ttnn.bfloat8_b", "input_b_dtype": "ttnn.float32"},
            {"input_a_dtype": "ttnn.bfloat8_b", "input_b_dtype": "ttnn.bfloat16"},
            {"input_a_dtype": "ttnn.bfloat8_b", "input_b_dtype": "ttnn.bfloat4_b"},
            {"input_a_dtype": "ttnn.bfloat4_b", "input_b_dtype": "ttnn.float32"},
            {"input_a_dtype": "ttnn.bfloat4_b", "input_b_dtype": "ttnn.bfloat16"},
            {"input_a_dtype": "ttnn.bfloat4_b", "input_b_dtype": "ttnn.bfloat8_b"},  # mixed dtype
        ],
        # "input_a_layout": [ttnn.TILE_LAYOUT],
        # "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_a_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_b_layout": [ttnn.ROW_MAJOR_LAYOUT],
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


def return_dtype(dtype):
    if dtype == "ttnn.bfloat16":
        return ttnn.bfloat16
    elif dtype == "ttnn.float32":
        return ttnn.float32
    elif dtype == "ttnn.bfloat8_b":
        return ttnn.bfloat8_b
    elif dtype == "ttnn.bfloat4_b":
        return ttnn.bfloat4_b


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
    input_dtype,
    input_a_layout,
    input_b_layout,
    input_mem_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    input_a_dtype = return_dtype(input_dtype["input_a_dtype"])
    input_b_dtype = return_dtype(input_dtype["input_b_dtype"])

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape["self"])

    if isinstance(input_shape["other"], list):
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(input_shape["other"])
    else:
        torch_input_tensor_b = torch.tensor(input_shape["other"], dtype=torch.float32)

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

    torch_input_tensor_a = ttnn.to_torch(input_tensor_a)
    torch_input_tensor_b = ttnn.to_torch(input_tensor_b)

    golden_function = ttnn.get_golden_function(ttnn.experimental.ne)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    start_time = start_measuring_time()
    result = ttnn.experimental.ne(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, pcc=0.99), e2e_perf]
