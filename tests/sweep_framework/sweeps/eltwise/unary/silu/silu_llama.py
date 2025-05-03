# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

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
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.


parameters = {
    "nightly": {
        "input_spec": [
            {"self": [1, 1, 32, 3584], "input_dtype": "ttnn.bfloat16", "y": 2},
            {"self": [1, 1, 32, 14336], "input_dtype": "ttnn.bfloat16", "y": 4},
        ],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_a_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Row Major layout is not supported"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_spec,
    input_a_layout,
    input_b_layout,
    *,
    device,
) -> list:
    torch.manual_seed(0)
    if input_spec["input_dtype"] == "ttnn.bfloat16":
        input_dtype = ttnn.bfloat16
    elif input_spec["input_dtype"] == "ttnn.float32":
        input_dtype = ttnn.float32
    elif input_spec["input_dtype"] == "ttnn.int32":
        input_dtype = ttnn.int32

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(input_spec["self"])

    golden_function = ttnn.get_golden_function(ttnn.silu)
    torch_output_tensor = golden_function(torch_input_tensor_a)

    sharded_config = ttnn.create_sharded_memory_config_(
        shape=input_spec["self"],
        core_grid=ttnn.CoreGrid(y=input_spec["y"], x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
        halo=0,
        tile_layout=True,
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=sharded_config,
    )

    start_time = start_measuring_time()
    result = ttnn.silu(input_tensor_a, memory_config=sharded_config)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
