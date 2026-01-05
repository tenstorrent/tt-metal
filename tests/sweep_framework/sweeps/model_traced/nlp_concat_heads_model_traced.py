# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from typing import Optional, Tuple

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("experimental::nlp_concat_heads", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 12, 32, 64)],  # Batch, heads, seq, head_dim
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    """
    Invalidate test vectors that might cause bad optional access.
    This can happen with None memory configs or invalid shapes.
    """
    output_memory_config = test_vector.get("output_memory_config")
    input_memory_config = test_vector.get("input_a_memory_config")
    input_shape = test_vector.get("input_shape")

    # Check for None memory configs
    if output_memory_config is None:
        return True, "output_memory_config is None"

    if input_memory_config is None:
        return True, "input_a_memory_config is None"

    # Check for invalid shape (must be 4D)
    if input_shape is None:
        return True, "input_shape is None"

    if isinstance(input_shape, (tuple, list)):
        if len(input_shape) != 4:
            return True, "Input shape must be 4D"
    elif isinstance(input_shape, str):
        # Try to parse string shape
        try:
            shape = eval(input_shape)
            if len(shape) != 4:
                return True, "Input shape must be 4D"
        except:
            return True, "Could not parse input_shape string"

    return False, None


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,  # Accept traced_source, traced_machine_info, config_id, etc.
) -> list:
    torch.manual_seed(0)

    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape)

    # nlp_concat_heads concatenates heads: [B, H, S, D] -> [B, 1, S, H*D]
    # So we need to compute the expected output shape
    if len(shape) == 4:
        batch, num_heads, seq_len, head_dim = shape
        # Reshape input to match expected output for comparison
        torch_output_tensor = (
            torch_input_tensor_a.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch, seq_len, num_heads * head_dim)
            .unsqueeze(1)
        )
    else:
        # Fallback: just clone the input
        torch_output_tensor = torch_input_tensor_a.clone()

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments based on storage_type
    from_torch_kwargs = {
        "dtype": input_a_dtype,
        "layout": input_a_layout,
    }

    # Only add device and memory_config if not HOST storage
    # Always use DRAM to avoid shard width mismatches
    if not is_host:
        from_torch_kwargs["device"] = device
        from_torch_kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs)

    start_time = start_measuring_time()
    output_tensor = ttnn.experimental.nlp_concat_heads(input_tensor_a, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)
    return [pcc, e2e_perf]
