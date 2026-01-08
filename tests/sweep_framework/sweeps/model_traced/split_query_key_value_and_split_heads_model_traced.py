# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
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
model_traced_params = loader.get_suite_parameters("split_query_key_value_and_split_heads", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 96)],  # Must be divisible for QKV split
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
        "num_heads": [1],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    """Custom device fixture with DispatchCoreConfig to free up more compute cores"""
    device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.device.DispatchCoreConfig())
    device_name = ttnn.get_arch_name()

    yield (device, device_name)

    ttnn.close_device(device)
    del device


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    num_heads=1,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,  # Accept traced_source, traced_machine_info, etc.
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

    # Get golden function for split_query_key_value_and_split_heads
    golden_function = ttnn.get_golden_function(ttnn.transformer.split_query_key_value_and_split_heads)

    # Golden function expects 3D input [batch, seq_len, hidden_dim]
    # but traced configs have 4D [batch, 1, seq_len, hidden_dim]
    if len(shape) == 4 and shape[1] == 1:
        # Squeeze out the second dimension for golden function
        torch_input_3d = torch_input_tensor_a.squeeze(1)
        (
            torch_query_tensor,
            torch_key_tensor,
            torch_value_tensor,
        ) = golden_function(torch_input_3d, num_heads=num_heads)
        # Unsqueeze back to 4D to match ttnn output
        torch_query_tensor = torch_query_tensor.unsqueeze(1)
        torch_key_tensor = torch_key_tensor.unsqueeze(1)
        torch_value_tensor = torch_value_tensor.unsqueeze(1)
    else:
        (
            torch_query_tensor,
            torch_key_tensor,
            torch_value_tensor,
        ) = golden_function(torch_input_tensor_a, num_heads=num_heads)

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

    # The operation expects 3D input [batch, seq_len, hidden_dim]
    # but traced configs have 4D [batch, 1, seq_len, hidden_dim]
    needs_unsqueeze = False
    if len(shape) == 4 and shape[1] == 1:
        # Squeeze out the second dimension for ttnn operation
        torch_input_tensor_a = torch_input_tensor_a.squeeze(1)
        needs_unsqueeze = True

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs)

    # Check if input has sharded memory - if so, convert to interleaved first
    # The operation internally uses create_qkv_heads which requires tile-aligned shard shapes
    # If traced config has non-tile-aligned shards, convert to interleaved
    if hasattr(input_tensor_a, "memory_config"):
        mem_config = input_tensor_a.memory_config()
        if mem_config.is_sharded():
            input_tensor_a = ttnn.to_memory_config(input_tensor_a, ttnn.DRAM_MEMORY_CONFIG)

    start_time = start_measuring_time()
    # This operation splits QKV and heads - returns tuple of (Q, K, V)
    query_tensor, key_tensor, value_tensor = ttnn.transformer.split_query_key_value_and_split_heads(
        input_tensor_a, num_heads=num_heads
    )
    query_tensor = ttnn.to_torch(query_tensor)
    key_tensor = ttnn.to_torch(key_tensor)
    value_tensor = ttnn.to_torch(value_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Unsqueeze back to 4D if we squeezed earlier
    if needs_unsqueeze:
        query_tensor = query_tensor.unsqueeze(1)
        key_tensor = key_tensor.unsqueeze(1)
        value_tensor = value_tensor.unsqueeze(1)

    # Check with PCC for all three outputs
    # check_with_pcc returns (bool, str) tuple
    pcc_q = check_with_pcc(torch_query_tensor, query_tensor, 0.999)
    pcc_k = check_with_pcc(torch_key_tensor, key_tensor, 0.999)
    pcc_v = check_with_pcc(torch_value_tensor, value_tensor, 0.999)

    # All three must pass for overall success
    all_pass = pcc_q[0] and pcc_k[0] and pcc_v[0]
    # Use minimum PCC value as the reported value
    min_pcc_value = min(float(pcc_q[1]), float(pcc_k[1]), float(pcc_v[1]))
    pcc_result = (all_pass, str(min_pcc_value))

    return [pcc_result, e2e_perf]
