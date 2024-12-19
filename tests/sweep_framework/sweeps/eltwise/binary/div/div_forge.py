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
        "input_shape": [
            {"self": [1, 12, 1, 1], "other": [1, 12, 1, 1], "input_dtype": "ttnn.float32"},
            {"self": [1, 16, 1, 1], "other": [1, 16, 1, 1], "input_dtype": "ttnn.float32"},
            {"self": [1, 6, 1, 1], "other": [1, 6, 1, 1], "input_dtype": "ttnn.float32"},
            {"self": [1, 8, 1, 1], "other": [1, 8, 1, 1], "input_dtype": "ttnn.float32"},
            {"self": [1], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [1, 1024, 640], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 1280, 16, 16], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 1280, 8, 8], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 12, 197, 197], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 12, 201, 201], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 12, 8, 8], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 16, 197, 197], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 1], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [1, 1, 16384, 256], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 1, 19200, 300], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 256, 1280], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 2, 4096, 256], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 2, 4800, 300], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 320, 64, 64], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 4096, 320], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 512], "other": [1, 1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 5, 1024, 256], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 5, 1200, 300], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 640, 32, 32], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 64, 1280], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 8, 2048, 256], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 8, 256, 2048], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 8, 256, 256], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 8, 300, 300], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [2, 512], "other": [2, 1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 1024, 512], "other": [1, 1024, 512], "input_dtype": "ttnn.float32"},
            {"self": [1, 1024, 640], "other": [1, 1024, 640], "input_dtype": "ttnn.float32"},
            {"self": [1, 10, 3072], "other": [1, 10, 3072], "input_dtype": "ttnn.float32"},
            {"self": [1, 10, 768], "other": [1, 10, 768], "input_dtype": "ttnn.float32"},
            {"self": [1, 1200, 1280], "other": [1, 1200, 1280], "input_dtype": "ttnn.float32"},
            {"self": [1, 1445, 768], "other": [1, 1445, 768], "input_dtype": "ttnn.float32"},
            {"self": [1, 1536], "other": [1, 1536], "input_dtype": "ttnn.float32"},
            {"self": [1, 16384, 128], "other": [1, 16384, 128], "input_dtype": "ttnn.float32"},
            {"self": [1, 16, 3072], "other": [1, 16, 3072], "input_dtype": "ttnn.float32"},
            {"self": [1, 19200, 256], "other": [1, 19200, 256], "input_dtype": "ttnn.float32"},
            {"self": [1, 197, 3072], "other": [1, 197, 3072], "input_dtype": "ttnn.float32"},
            {"self": [1, 197, 4096], "other": [1, 197, 4096], "input_dtype": "ttnn.float32"},
            {"self": [1, 19, 4096], "other": [1, 19, 4096], "input_dtype": "ttnn.float32"},
            {"self": [1, 201, 3072], "other": [1, 201, 3072], "input_dtype": "ttnn.float32"},
            {"self": [1, 2048, 768], "other": [1, 2048, 768], "input_dtype": "ttnn.float32"},
            {"self": [1, 256, 1024], "other": [1, 256, 1024], "input_dtype": "ttnn.float32"},
            {"self": [1, 256, 1280], "other": [1, 256, 1280], "input_dtype": "ttnn.float32"},
            {"self": [1, 256, 256], "other": [1, 256, 256], "input_dtype": "ttnn.float32"},
            {"self": [1, 256, 4096], "other": [1, 256, 4096], "input_dtype": "ttnn.float32"},
            {"self": [1, 256, 5120], "other": [1, 256, 5120], "input_dtype": "ttnn.float32"},
            {"self": [1, 25, 3072], "other": [1, 25, 3072], "input_dtype": "ttnn.float32"},
            {"self": [1, 300, 2048], "other": [1, 300, 2048], "input_dtype": "ttnn.float32"},
            {"self": [1, 3072, 8], "other": [1, 3072, 8], "input_dtype": "ttnn.float32"},
            {"self": [1, 4096, 1280], "other": [1, 4096, 1280], "input_dtype": "ttnn.float32"},
            {"self": [1, 4096, 256], "other": [1, 4096, 256], "input_dtype": "ttnn.float32"},
            {"self": [1, 4800, 512], "other": [1, 4800, 512], "input_dtype": "ttnn.float32"},
            {"self": [1, 64, 5120], "other": [1, 64, 5120], "input_dtype": "ttnn.float32"},
            {"self": [1, 7, 18176], "other": [1, 7, 18176], "input_dtype": "ttnn.float32"},
            {"self": [1, 1024], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 768], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [1, 1, 1], "other": [1], "input_dtype": "ttnn.float32"},
            {"self": [1, 512], "other": [1], "input_dtype": "ttnn.bfloat16"},
            {"self": [112], "other": [112], "input_dtype": "ttnn.float32"},
            {"self": [116], "other": [116], "input_dtype": "ttnn.float32"},
            {"self": [120], "other": [120], "input_dtype": "ttnn.float32"},
            {"self": [128], "other": [128], "input_dtype": "ttnn.float32"},
            {"self": [1280], "other": [1280], "input_dtype": "ttnn.float32"},
            {"self": [134], "other": [134], "input_dtype": "ttnn.float32"},
            {"self": [14], "other": [14], "input_dtype": "ttnn.float32"},
            {"self": [144], "other": [144], "input_dtype": "ttnn.float32"},
            {"self": [16], "other": [16], "input_dtype": "ttnn.float32"},
            {"self": [160], "other": [160], "input_dtype": "ttnn.float32"},
            {"self": [168], "other": [168], "input_dtype": "ttnn.float32"},
            {"self": [184], "other": [184], "input_dtype": "ttnn.float32"},
            {"self": [192], "other": [192], "input_dtype": "ttnn.float32"},
            {"self": [196], "other": [196], "input_dtype": "ttnn.float32"},
            {"self": [20], "other": [20], "input_dtype": "ttnn.float32"},
            {"self": [200], "other": [200], "input_dtype": "ttnn.float32"},
            {"self": [2048], "other": [2048], "input_dtype": "ttnn.float32"},
            {"self": [24], "other": [24], "input_dtype": "ttnn.float32"},
            {"self": [240], "other": [240], "input_dtype": "ttnn.float32"},
            {"self": [256], "other": [256], "input_dtype": "ttnn.float32"},
            {"self": [272], "other": [272], "input_dtype": "ttnn.float32"},
            {"self": [28], "other": [28], "input_dtype": "ttnn.float32"},
            {"self": [32], "other": [32], "input_dtype": "ttnn.float32"},
            {"self": [320], "other": [320], "input_dtype": "ttnn.float32"},
            {"self": [334], "other": [334], "input_dtype": "ttnn.float32"},
            {"self": [34], "other": [34], "input_dtype": "ttnn.float32"},
            {"self": [384], "other": [384], "input_dtype": "ttnn.float32"},
            {"self": [40], "other": [40], "input_dtype": "ttnn.float32"},
            {"self": [46], "other": [46], "input_dtype": "ttnn.float32"},
            {"self": [462], "other": [462], "input_dtype": "ttnn.float32"},
            {"self": [480], "other": [480], "input_dtype": "ttnn.float32"},
            {"self": [512], "other": [512], "input_dtype": "ttnn.float32"},
            {"self": [576], "other": [576], "input_dtype": "ttnn.float32"},
            {"self": [58], "other": [58], "input_dtype": "ttnn.float32"},
            {"self": [64], "other": [64], "input_dtype": "ttnn.float32"},
            {"self": [640], "other": [640], "input_dtype": "ttnn.float32"},
            {"self": [672], "other": [672], "input_dtype": "ttnn.float32"},
            {"self": [68], "other": [68], "input_dtype": "ttnn.float32"},
            {"self": [72], "other": [72], "input_dtype": "ttnn.float32"},
            {"self": [78], "other": [78], "input_dtype": "ttnn.float32"},
            {"self": [80], "other": [80], "input_dtype": "ttnn.float32"},
            {"self": [96], "other": [96], "input_dtype": "ttnn.float32"},
            {"self": [960], "other": [960], "input_dtype": "ttnn.float32"},
            {"self": [98], "other": [98], "input_dtype": "ttnn.float32"},
            # {"self": [1, 12, 10, 10], "other": [1, 12, 10, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 12, 197, 197], "other": [1, 12, 197, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 12, 1, 10], "other": [1, 12, 1, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 12, 201, 201], "other": [1, 12, 201, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 12, 8, 8], "other": [1, 12, 8, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 16, 10, 10], "other": [1, 16, 10, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 16, 197, 197], "other": [1, 16, 197, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 16, 1, 10], "other": [1, 16, 1, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 16, 32, 32], "other": [1, 16, 32, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 16, 5, 5], "other": [1, 16, 5, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 1, 16384, 256], "other": [1, 1, 16384, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 1, 19200, 300], "other": [1, 1, 19200, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 2, 4096, 256], "other": [1, 2, 4096, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 2, 4800, 300], "other": [1, 2, 4800, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 5, 1024, 256], "other": [1, 5, 1024, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 5, 1200, 300], "other": [1, 5, 1200, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 6, 15, 15], "other": [1, 6, 15, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 6, 1, 15], "other": [1, 6, 1, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 8, 10, 10], "other": [1, 8, 10, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 8, 1, 10], "other": [1, 8, 1, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 8, 2048, 256], "other": [1, 8, 2048, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 8, 256, 2048], "other": [1, 8, 256, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 8, 256, 256], "other": [1, 8, 256, 1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 8, 300, 300], "other": [1, 8, 300, 1], "input_dtype": "ttnn.float32"},
            # {"self": [8, 100, 100], "other": [8, 100, 1], "input_dtype": "ttnn.float32"},
            # {"self": [8, 100, 920], "other": [8, 100, 1], "input_dtype": "ttnn.float32"},
            # {"self": [8, 920, 920], "other": [8, 920, 1], "input_dtype": "ttnn.float32"},
            # {"self": [10, 10], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [128], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [15, 15], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [160], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 16, 5, 5], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 23, 40], "other": [1, 1, 40], "input_dtype": "ttnn.float32"},
            # {"self": [1, 23, 40], "other": [1, 23, 1], "input_dtype": "ttnn.float32"},
            # {"self": [20], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [2], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [3234, 1], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [3], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [3, 320, 320], "other": [3, 1, 1], "input_dtype": "ttnn.float32"},
            # {"self": [5], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [197], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [19], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 480, 1, 1], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 672, 1, 1], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 72, 1, 1], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 184, 20, 20], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 200, 20, 20], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 240, 20, 20], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 240, 40, 40], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 480, 10, 10], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 480, 20, 20], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 672, 10, 10], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 672, 20, 20], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 10, 1], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 120, 1, 1], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 1280, 1, 1], "other": [1], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1, 15, 1], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 2048, 1, 1], "other": [1], "input_dtype": "ttnn.bfloat16"},
            # {"self": [1, 32, 1], "other": [1], "input_dtype": "ttnn.float32"},
            # {"self": [1, 512, 1, 1], "other": [1], "input_dtype": "ttnn.bfloat16"},
        ],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_a_layout"] == ttnn.ROW_MAJOR_LAYOUT or test_vector["input_b_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Row Major layout is not supported"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_layout,
    input_b_layout,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)
    if input_shape["input_dtype"] == "ttnn.bfloat16":
        input_dtype = ttnn.bfloat16
    elif input_shape["input_dtype"] == "ttnn.float32":
        input_dtype = ttnn.float32
    elif input_shape["input_dtype"] == "ttnn.int32":
        input_dtype = ttnn.int32

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(input_shape["self"])
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(input_shape["other"])

    golden_function = ttnn.get_golden_function(ttnn.div)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    start_time = start_measuring_time()
    result = ttnn.div(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
    # ToDo: Update it once the tensor layout support with rank < 2 is supported in mid of Jan
    output_tensor = ttnn.to_torch(result, torch_rank=len(input_shape["self"]))
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
