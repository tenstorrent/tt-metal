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

TIMEOUT = 60

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("where", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
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
    input_c_dtype=None,
    input_c_layout=None,
    input_c_memory_config=None,
    scalar_if_true=None,
    scalar_if_false=None,
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    is_ternary_tensor = input_b_dtype is not None and input_c_dtype is not None
    shape_a = input_shape if isinstance(input_shape, (tuple, list)) else input_shape

    if is_ternary_tensor:
        # Tensor creation
        torch_condition = torch.randint(0, 2, shape_a, dtype=torch.float32)
        torch_input_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(shape_a)
        torch_input_c = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_c_dtype
        )(shape_a)
        torch_output = torch.where(torch_condition > 0, torch_input_b, torch_input_c)

        condition_tensor = ttnn.from_torch(
            torch_condition,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=input_a_memory_config,
        )
        input_tensor_b = ttnn.from_torch(
            torch_input_b,
            dtype=input_b_dtype,
            layout=input_b_layout,
            device=device,
            memory_config=input_b_memory_config,
        )
        input_tensor_c = ttnn.from_torch(
            torch_input_c,
            dtype=input_c_dtype,
            layout=input_c_layout,
            device=device,
            memory_config=input_c_memory_config,
        )

        # Op call
        start_time = start_measuring_time()
        output_tensor = ttnn.where(condition_tensor, input_tensor_b, input_tensor_c, memory_config=output_memory_config)
        output_tensor = ttnn.to_torch(output_tensor)
        e2e_perf = stop_measuring_time(start_time)
    else:
        # Tensor creation
        try:
            scalar_true = float(scalar_if_true) if scalar_if_true is not None else 1.0
        except (ValueError, TypeError):
            scalar_true = 1.0
        try:
            scalar_false = float(scalar_if_false) if scalar_if_false is not None else 0.0
        except (ValueError, TypeError):
            scalar_false = 0.0
        torch_condition = torch.randint(0, 2, shape_a, dtype=torch.float32)
        torch_output = torch.where(torch_condition > 0, scalar_true, scalar_false)

        condition_tensor = ttnn.from_torch(
            torch_condition,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=input_a_memory_config,
        )

        # Op call
        start_time = start_measuring_time()
        output_tensor = ttnn.where(condition_tensor, scalar_true, scalar_false, memory_config=output_memory_config)
        output_tensor = ttnn.to_torch(output_tensor)
        e2e_perf = stop_measuring_time(start_time)

    # Comparison
    return [check_with_pcc(torch_output, output_tensor, 0.999), e2e_perf]
