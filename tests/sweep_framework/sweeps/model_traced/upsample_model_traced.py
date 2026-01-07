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
model_traced_params = loader.get_suite_parameters("upsample", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Note: upsample requires scale_factor and mode from JSON
    # Sample test skipped - use model_traced suite only
}

# Only add model_traced suite if it has valid configurations
if model_traced_params and any(len(v) > 0 for v in model_traced_params.values() if isinstance(v, list)):
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    scale_factor=None,
    mode=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    """
    Run upsample test with parameters extracted from traced JSON.
    All parameters are now extracted from JSON including scale_factor.
    """
    torch.manual_seed(0)

    # Handle tuple input_shape for sample suite
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # scale_factor must be extracted from JSON - no fallbacks
    if scale_factor is None:
        raise ValueError(f"Missing scale_factor from JSON")

    # Handle scale_factor - can be int or list [H, W]
    if isinstance(scale_factor, list):
        # If array format [H, W], use first element if both are same, otherwise use tuple
        if len(scale_factor) == 2:
            if scale_factor[0] == scale_factor[1]:
                scale_factor = scale_factor[0]
            else:
                scale_factor = tuple(scale_factor)
        else:
            raise ValueError(f"Invalid scale_factor format from JSON: {scale_factor}")
    elif not isinstance(scale_factor, (int, tuple)):
        raise ValueError(f"Invalid scale_factor type from JSON: {type(scale_factor)}, value: {scale_factor}")

    # mode must be extracted from JSON - no fallbacks
    if mode is None:
        raise ValueError(f"Missing mode from JSON")
    upsample_mode = mode

    torch_output_tensor = ttnn.get_golden_function(ttnn.upsample)(torch_input_tensor_a, scale_factor=scale_factor)

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
    # Handle scale_factor - can be int or tuple
    if isinstance(scale_factor, (tuple, list)) and len(scale_factor) == 2:
        output_tensor = ttnn.upsample(
            input_tensor_a, scale_factor=tuple(scale_factor), memory_config=output_memory_config
        )
    else:
        output_tensor = ttnn.upsample(input_tensor_a, scale_factor=scale_factor, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
