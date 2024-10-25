# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

random.seed(0)


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.
parameters = {
    "nightly": {
        "embedding_args": gen_shapes([1, 32, 32, 128], [4, 2080, 4128, 550], [1, 32, 32, 32], 32),
        "input_dtype": [ttnn.uint32],
        "weight_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "output_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "weight_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "weight_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_layout"] == ttnn.TILE_LAYOUT:
        return True, "Input must be in row major layout"
    if test_vector["weight_layout"] == ttnn.TILE_LAYOUT:
        return True, "Weights must in row major layout"
    if test_vector["output_dtype"] == ttnn.bfloat8_b:
        return True, "bloat8_b is not supported for output tensor"
    if test_vector["weight_layout"] == ttnn.ROW_MAJOR_LAYOUT and test_vector["weight_dtype"] == ttnn.bfloat8_b:
        return True, "bfloat8_b is only supported on tiled layout"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    embedding_args,
    input_dtype,
    weight_dtype,
    output_dtype,
    input_layout,
    weight_layout,
    input_memory_config,
    weight_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    batch_size, seq_length, embeddings_dim, num_embeddings = embedding_args

    input_shape = (batch_size, seq_length)
    weight_shape = (num_embeddings, embeddings_dim)

    torch_input_tensor = torch_random(input_shape, 0, num_embeddings, torch.int64)
    torch_weight_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), weight_dtype
    )(weight_shape)

    golden_function = ttnn.get_golden_function(ttnn.embedding)
    torch_output_tensor = golden_function(torch_input_tensor, torch_weight_tensor).squeeze(dim=0)
    # torch_output_tensor = torch.nn.functional.embedding(torch_input_tensor, torch_weight_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_memory_config,
    )
    weight_tensor = ttnn.from_torch(
        torch_weight_tensor,
        dtype=weight_dtype,
        layout=weight_layout,
        device=device,
        memory_config=weight_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.embedding(input_tensor, weight_tensor, dtype=output_dtype, memory_config=output_memory_config)
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(output_tensor).squeeze()

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
