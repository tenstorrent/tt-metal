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
model_traced_params = loader.get_suite_parameters("where", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
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
    storage_type="StorageType::DEVICE",
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    input_c_dtype=None,
    input_c_layout=None,
    input_c_memory_config=None,
    scalar_if_true=None,  # Scalar value from traced configs (arg1)
    scalar_if_false=None,  # Scalar value from traced configs (arg2)
    *,
    device,
) -> list:
    torch.manual_seed(0)

    # The where operation signature: where(condition, value_if_true, value_if_false)
    # Traced configs show: arg0=Tensor, arg1=scalar, arg2=scalar
    # So this is: where(tensor, scalar, scalar)

    # Determine operation mode based on optional parameters
    # If input_b_dtype and input_c_dtype provided, it's tensor-tensor-tensor mode
    # Otherwise, it's tensor-scalar-scalar mode (from traced configs)
    is_ternary_tensor = input_b_dtype is not None and input_c_dtype is not None

    # Handle input shapes
    if isinstance(input_shape, (tuple, list)):
        shape_a = tuple(input_shape)
    else:
        shape_a = input_shape

    # Check if storage_type is HOST
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments for tensor A (condition)
    from_torch_kwargs_a = {
        "dtype": input_a_dtype,
        "layout": input_a_layout,
    }
    if not is_host:
        from_torch_kwargs_a["device"] = device
        from_torch_kwargs_a["memory_config"] = input_a_memory_config

    if is_ternary_tensor:
        # Tensor-tensor-tensor mode: where(condition_tensor, x_tensor, y_tensor)
        if isinstance(input_shape, dict) and "self" in input_shape:
            shape_a = input_shape["self"]
            shape_b = input_shape.get("input_b", shape_a)
            shape_c = input_shape.get("input_c", shape_a)
        else:
            shape_b = shape_a
            shape_c = shape_a

        # Generate condition tensor (boolean-like: 0 or 1)
        torch_condition = torch.randint(0, 2, shape_a, dtype=torch.float32)

        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(shape_b)

        torch_input_tensor_c = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_c_dtype
        )(shape_c)

        # PyTorch reference
        torch_output_tensor = torch.where(torch_condition > 0, torch_input_tensor_b, torch_input_tensor_c)

        condition_tensor = ttnn.from_torch(torch_condition, **from_torch_kwargs_a)

        # Build from_torch arguments for tensor B
        from_torch_kwargs_b = {
            "dtype": input_b_dtype,
            "layout": input_b_layout,
        }
        if not is_host:
            from_torch_kwargs_b["device"] = device
            from_torch_kwargs_b["memory_config"] = input_b_memory_config

        input_tensor_b = ttnn.from_torch(torch_input_tensor_b, **from_torch_kwargs_b)

        # Build from_torch arguments for tensor C
        from_torch_kwargs_c = {
            "dtype": input_c_dtype,
            "layout": input_c_layout,
        }
        if not is_host:
            from_torch_kwargs_c["device"] = device
            from_torch_kwargs_c["memory_config"] = input_c_memory_config

        input_tensor_c = ttnn.from_torch(torch_input_tensor_c, **from_torch_kwargs_c)

        start_time = start_measuring_time()
        output_tensor = ttnn.where(condition_tensor, input_tensor_b, input_tensor_c, memory_config=output_memory_config)
        output_tensor = ttnn.to_torch(output_tensor)
        e2e_perf = stop_measuring_time(start_time)

        pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    else:
        # Tensor-scalar-scalar mode: where(condition_tensor, scalar_if_true, scalar_if_false)
        # Use scalar values from traced configs if provided, otherwise use defaults
        if scalar_if_true is None:
            scalar_if_true = 1.0  # Default fallback
        if scalar_if_false is None:
            scalar_if_false = 0.0  # Default fallback

        # Generate condition tensor (boolean-like: 0 or 1)
        torch_condition = torch.randint(0, 2, shape_a, dtype=torch.float32)

        # PyTorch reference: where with scalars
        torch_output_tensor = torch.where(torch_condition > 0, scalar_if_true, scalar_if_false)

        condition_tensor = ttnn.from_torch(torch_condition, **from_torch_kwargs_a)

        start_time = start_measuring_time()
        # TTNN where with condition tensor and two scalars
        output_tensor = ttnn.where(
            condition_tensor, scalar_if_true, scalar_if_false, memory_config=output_memory_config
        )
        output_tensor = ttnn.to_torch(output_tensor)
        e2e_perf = stop_measuring_time(start_time)

        pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
