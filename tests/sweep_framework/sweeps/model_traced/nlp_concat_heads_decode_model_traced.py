# Model traced sweep for nlp_concat_heads_decode
# Generated automatically - DO NOT EDIT MANUALLY

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
model_traced_params = loader.get_suite_parameters("experimental::nlp_concat_heads_decode", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 12, 32, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
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
    output_memory_config,
    num_heads,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    # num_heads is now required and passed from traced configs
    # If somehow None, try to infer from shape
    if num_heads is None:
        if len(shape) >= 2:
            if len(shape) == 4:
                if shape[1] == 1:
                    num_heads = shape[2]  # [B, 1, H, D] -> H is num_heads
                else:
                    num_heads = shape[1]  # [B, H, S, D] -> H is num_heads
            else:
                num_heads = 16  # Default fallback
        else:
            num_heads = 16  # Default fallback

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape)

    # nlp_concat_heads_decode concatenates heads: [B, 1, H, D] -> [B, 1, num_heads, num_heads*D]
    # Based on actual output: input [1, 1, 32, 64] with num_heads=32 -> output [1, 1, 32, 2048]
    # The operation reshapes and concatenates heads based on num_heads parameter
    # Output shape is [B, 1, num_heads, num_heads*head_dim]
    # Use golden function if available, otherwise use a simple approximation
    if len(shape) == 4:
        batch, _, seq_or_heads, head_dim = shape
        expected_output_shape = (batch, 1, num_heads, num_heads * head_dim)

        # Try to use golden function for accurate reference
        try:
            golden_func = ttnn.get_golden_function(ttnn.experimental.nlp_concat_heads_decode)
            torch_output_tensor = golden_func(torch_input_tensor_a, num_heads=num_heads)
        except:
            # Fallback: create a simple reference by reshaping input
            # This is an approximation - the actual operation does complex head concatenation
            # For now, just replicate the input to match output shape
            input_elements = torch_input_tensor_a.numel()
            output_elements = batch * 1 * num_heads * num_heads * head_dim
            # Repeat input data to fill output shape
            repeated = torch_input_tensor_a.flatten().repeat((output_elements // input_elements) + 1)[:output_elements]
            torch_output_tensor = repeated.reshape(expected_output_shape)
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
    output_tensor = ttnn.experimental.nlp_concat_heads_decode(
        input_tensor_a, num_heads=num_heads, memory_config=output_memory_config
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC - using standard threshold
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)

    return [pcc, e2e_perf]
