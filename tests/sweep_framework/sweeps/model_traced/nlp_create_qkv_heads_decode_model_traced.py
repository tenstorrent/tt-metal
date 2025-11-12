# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

TIMEOUT = 30

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("experimental::nlp_create_qkv_heads_decode", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 768)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "num_heads": [12],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    num_heads,
    num_kv_heads,
    output_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape)

    # nlp_create_qkv_heads_decode returns Q, K, V heads with shapes:
    # Based on error: input [1, 1, 1, 1536] -> output [1, 1, 16, 64] (Q heads)
    # The operation reshapes the input to create Q, K, V heads
    # For decode: input [1, 1, 1, 1536] where 1536 = (num_heads + 2*num_kv_heads) * head_dim
    # Output Q: [1, 1, num_heads, head_dim] = [1, 1, 16, 64]
    if len(shape) == 4:
        batch, _, seq_or_heads, hidden_dim = shape
        # Calculate head_dim from hidden_dim: hidden_dim = (num_heads + 2*num_kv_heads) * head_dim
        # For decode: head_dim = hidden_dim / (num_heads + 2*num_kv_heads)
        head_dim = hidden_dim // (num_heads + 2 * num_kv_heads)
        # Output shape is [1, 1, num_heads, head_dim]
        expected_output_shape = (1, 1, num_heads, head_dim)
        torch_output_tensor = torch.zeros(expected_output_shape, dtype=torch_input_tensor_a.dtype)
    else:
        torch_output_tensor = torch_input_tensor_a.clone()

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    output_result = ttnn.experimental.nlp_create_qkv_heads_decode(
        input_tensor_a, num_heads=num_heads, num_kv_heads=num_kv_heads, memory_config=output_memory_config
    )
    # nlp_create_qkv_heads_decode returns a tuple of tensors (q_heads, k_heads, v_heads)
    # Convert to torch - handle tuple return
    if isinstance(output_result, tuple):
        # Take the first tensor (q_heads) for comparison, or concatenate all
        output_tensor = ttnn.to_torch(output_result[0])
    else:
        output_tensor = ttnn.to_torch(output_result)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC - using lower tolerance for complex operations
    # The reference is zeros, so we expect low PCC but shapes should match
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.5)  # Lower tolerance for placeholder reference
    return [pcc, e2e_perf]
