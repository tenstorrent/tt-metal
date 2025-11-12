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

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("reshape", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "target_shape": [(1, 32, 1, 32)],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    target_shape,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    # Handle tuple input_shape for sample suite
    if isinstance(input_shape, (tuple, list)):
        input_shape_tuple = tuple(input_shape)
    else:
        input_shape_tuple = input_shape

    # Handle target_shape - convert to tuple if needed
    if isinstance(target_shape, (tuple, list)):
        target_shape_tuple = tuple(target_shape)
    else:
        target_shape_tuple = target_shape

    # Validate that target_shape matches input_shape in total elements
    import math

    input_elements = math.prod(input_shape_tuple)
    # Handle -1 in target_shape (means infer from other dimensions)
    if -1 in target_shape_tuple:
        # Calculate what -1 should be
        known_product = math.prod([d for d in target_shape_tuple if d != -1])
        if known_product == 0:
            raise ValueError(
                f"Invalid target_shape {target_shape_tuple}: cannot infer -1 with zero in other dimensions"
            )
        inferred_dim = input_elements // known_product
        target_shape_tuple = tuple([inferred_dim if d == -1 else d for d in target_shape_tuple])

    target_elements = math.prod(target_shape_tuple)
    if input_elements != target_elements:
        raise ValueError(
            f"Invalid reshape: input_shape {input_shape_tuple} has {input_elements} elements, "
            f"but target_shape {target_shape_tuple} has {target_elements} elements"
        )

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape_tuple)

    torch_output_tensor = torch.reshape(torch_input_tensor_a, target_shape_tuple)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.reshape(input_tensor_a, target_shape_tuple, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
