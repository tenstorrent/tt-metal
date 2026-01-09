# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.master_config_loader import MasterConfigLoader

TIMEOUT = 30

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("gt", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
        "scalar": [0.0],
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
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    scalar=None,
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    is_binary = input_b_dtype is not None
    shape_a = input_shape if isinstance(input_shape, (tuple, list)) else input_shape

    # Tensor creation
    torch_input_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)
    input_tensor_a = ttnn.from_torch(
        torch_input_a, dtype=input_a_dtype, layout=input_a_layout, device=device, memory_config=input_a_memory_config
    )

    if is_binary:
        torch_input_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(shape_a)
        torch_output = ttnn.get_golden_function(ttnn.gt)(torch_input_a, torch_input_b)
        input_tensor_b = ttnn.from_torch(
            torch_input_b,
            dtype=input_b_dtype,
            layout=input_b_layout,
            device=device,
            memory_config=input_b_memory_config,
        )

        # Op call
        start_time = start_measuring_time()
        output_tensor = ttnn.gt(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
        output_tensor = ttnn.to_torch(output_tensor)
        e2e_perf = stop_measuring_time(start_time)
    else:
        scalar_value = scalar if scalar is not None else 0
        torch_output = ttnn.get_golden_function(ttnn.gt)(torch_input_a, scalar_value)

        # Op call
        start_time = start_measuring_time()
        output_tensor = ttnn.gt(input_tensor_a, scalar_value, memory_config=output_memory_config)
        output_tensor = ttnn.to_torch(output_tensor)
        e2e_perf = stop_measuring_time(start_time)

    # Comparison
    return [check_with_pcc(torch_output.float(), output_tensor.float(), 0.999), e2e_perf]
