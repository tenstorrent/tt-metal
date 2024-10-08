# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.utils import gen_shapes
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
        "grad_dtype": [ttnn.bfloat16],
        "weight_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "output_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "grad_layout": [ttnn.TILE_LAYOUT],
        "weight_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "grad_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "weight_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
    "xfail": {
        "embedding_args": gen_shapes([1, 32, 32, 128], [4, 2080, 4128, 550], [1, 32, 32, 32], 32),
        "input_dtype": [ttnn.bfloat16],
        "grad_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "weight_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "output_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "grad_layout": [ttnn.TILE_LAYOUT],
        "weight_layout": [ttnn.TILE_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "grad_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "weight_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
    },
}


# TO-DO: Create an issue on this, since these constrictions are not mentioned in the documentation
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["output_dtype"] != test_vector["grad_dtype"]:
        return True, "Output and input gradient tensors must have the same dtype"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    embedding_args,
    input_dtype,
    grad_dtype,
    weight_dtype,
    output_dtype,
    input_layout,
    grad_layout,
    weight_layout,
    input_memory_config,
    grad_memory_config,
    weight_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    batch_size, seq_length, embeddings_dim, num_embeddings = embedding_args

    input_shape = (batch_size, seq_length)
    grad_shape = (1, 1, batch_size * seq_length, embeddings_dim)
    weight_shape = (num_embeddings, embeddings_dim)

    torch_input_tensor = torch_random(input_shape, 0, num_embeddings, torch.int64)
    torch_grad_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), grad_dtype
    )(grad_shape)
    torch_weight_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), weight_dtype
    )(weight_shape)

    torch_weight_tensor.requires_grad = True
    torch_weight_tensor.retain_grad()

    intermediate_result = torch.nn.functional.embedding(torch_input_tensor, torch_weight_tensor).reshape(grad_shape)
    intermediate_result.backward(gradient=torch_grad_tensor)
    torch_output_tensor = torch_weight_tensor.grad

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_memory_config,
    )
    grad_tensor = ttnn.from_torch(
        torch_grad_tensor,
        dtype=grad_dtype,
        layout=grad_layout,
        device=device,
        memory_config=grad_memory_config,
    )
    weight_tensor = ttnn.from_torch(
        torch_weight_tensor,
        dtype=weight_dtype,
        layout=weight_layout,
        device=device,
        memory_config=weight_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.embedding_bw(
        input_tensor, weight_tensor, grad_tensor, dtype=output_dtype, memory_config=output_memory_config
    )
    output_tensor = ttnn.to_torch(output_tensor).squeeze()
    e2e_perf = stop_measuring_time(start_time)

    info_string = f"{input_dtype}. {weight_dtype}, {grad_dtype}"

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf, info_string]
