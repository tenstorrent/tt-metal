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
model_traced_params = loader.get_suite_parameters("experimental::nlp_concat_heads_decode", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 12, 32, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
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
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    # num_heads is required - try to infer from shape if missing
    if num_heads is None:
        # Try to infer from input shape: [B, 1, H, D] where H might be num_heads or head_dim
        # For nlp_concat_heads_decode, input is typically [1, 1, num_heads, head_dim]
        if len(shape) == 4 and shape[1] == 1:
            # Use shape[2] as num_heads (third dimension)
            num_heads = shape[2]
        else:
            # Default fallback
            num_heads = 16

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape)

    # Proper torch reference from test_nlp_concat_heads_decode.py (line 95)
    # Input shape: [1, batch, padded_heads, head_dim]
    # Output shape: [1, 1, batch, head_dim * num_heads]
    # The operation takes first num_heads from padded_heads dimension and concatenates them

    if len(shape) == 4:
        _, batch, padded_heads, head_dim = shape
        # Take first num_heads from the padded_heads dimension and reshape
        # Input: (1, batch, padded_heads, head_dim) -> Output: (1, 1, batch, head_dim * num_heads)
        torch_output_tensor = torch_input_tensor_a[:, :, :num_heads, :].reshape(1, 1, batch, head_dim * num_heads)
    else:
        torch_output_tensor = torch_input_tensor_a.clone()

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
    output_tensor = ttnn.experimental.nlp_concat_heads_decode(
        input_tensor_a, num_heads=num_heads, memory_config=output_memory_config
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Unpad the output - TTNN output may be padded to tile size (32)
    # We need to extract only the actual batch size
    if len(shape) == 4:
        _, batch, _, _ = shape
        output_tensor = output_tensor[:, :, :batch, :]

    # Check with PCC - using standard threshold
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)

    return [pcc, e2e_perf]
