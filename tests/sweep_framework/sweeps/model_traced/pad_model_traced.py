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
model_traced_params = loader.get_suite_parameters("pad", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "padding": [
            ((0, 1), (0, 1), (0, 2), (0, 2))
        ],  # padding as tuple of tuples: ((dim0_left, dim0_right), (dim1_left, dim1_right), ...)
        "value": [0.0],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
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
    padding=None,
    value=0.0,
    output_padded_shape=None,
    input_tensor_start=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
) -> list:
    torch.manual_seed(0)

    # Handle tuple input_shape for sample suite
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # Determine which pad format is being used
    if output_padded_shape is not None and input_tensor_start is not None:
        # Using output_padded_shape format from traced JSON
        # Calculate padding for PyTorch reference from logical shapes
        calculated_padding = []
        for i in range(len(shape)):
            start = input_tensor_start[i] if i < len(input_tensor_start) else 0
            end = output_padded_shape[i] - shape[i] - start
            # Ensure padding is non-negative (if output < input, this is invalid for padding)
            if end < 0:
                # This means output is smaller than input - invalid for padding, use 0
                calculated_padding.append([0, 0])
            else:
                calculated_padding.append([start, end])

        # Convert to torch padding format (reverse order and flatten) for reference output
        torch_padding = []
        for i in range(len(calculated_padding) - 1, -1, -1):
            for p in calculated_padding[i]:
                torch_padding.append(p)
        torch_output_tensor = torch.nn.functional.pad(torch_input_tensor_a, torch_padding, mode="constant", value=value)

        # Convert to padding format for ttnn.pad
        padding = tuple(tuple(p) for p in calculated_padding)
    else:
        # Using padding format (default)
        if padding is None:
            # Fallback: no padding
            padding = [[0, 0]] * len(shape)

        # Convert padding format for PyTorch (reverse order and flatten)
        torch_padding = []
        for i in range(len(padding) - 1, -1, -1):  # go through each dim of padding
            for p in padding[i]:
                torch_padding.append(p)  # each dim has 2 padding values
        torch_output_tensor = torch.nn.functional.pad(torch_input_tensor_a, torch_padding, mode="constant", value=value)

        if isinstance(padding, list):
            padding = tuple(tuple(p) if isinstance(p, (list, tuple)) else p for p in padding)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments based on storage_type
    from_torch_kwargs = {
        "dtype": input_a_dtype,
        "layout": input_a_layout,
    }

    # Only add device and memory_config if not HOST storage
    if not is_host:
        from_torch_kwargs["device"] = device
        from_torch_kwargs["memory_config"] = input_a_memory_config

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs)

    start_time = start_measuring_time()
    # Call ttnn.pad directly - this will fail for HOST tensors (padding will be ignored)
    output_tensor = ttnn.pad(input_tensor_a, padding=padding, value=value)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
