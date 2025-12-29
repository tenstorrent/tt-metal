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
    output_memory_config,
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

    # Check if traced config requires sharding that exceeds device capabilities
    if hasattr(input_a_memory_config, "shard_spec") and input_a_memory_config.shard_spec is not None:
        shard_spec = input_a_memory_config.shard_spec
        if hasattr(shard_spec, "grid"):
            grid = shard_spec.grid
            if hasattr(grid, "ranges"):
                for core_range in grid.ranges():
                    end_coord = core_range.end  # CoreRange uses .end not .end_coord
                    required_x = end_coord.x + 1
                    required_y = end_coord.y + 1
                    if required_x > device.core_grid.x or required_y > device.core_grid.y:
                        import pytest

                        pytest.skip(
                            f"Insufficient device cores: requires {required_x}x{required_y}, "
                            f"but device has {device.core_grid.x}x{device.core_grid.y}"
                        )

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # Get golden function for split_query_key_value_and_split_heads
    golden_function = ttnn.get_golden_function(ttnn.transformer.split_query_key_value_and_split_heads)
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

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs)

    # Check if input has sharded memory - if so, convert to interleaved first
    # The operation internally uses create_qkv_heads which requires tile-aligned shard shapes
    # If traced config has non-tile-aligned shards, convert to interleaved
    needs_conversion = False
    if hasattr(input_tensor_a, "memory_config"):
        mem_config = input_tensor_a.memory_config()
        if mem_config.is_sharded():
            needs_conversion = True
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

    # Check with PCC for all three outputs
    pcc_q = check_with_pcc(torch_query_tensor, query_tensor, 0.999)
    pcc_k = check_with_pcc(torch_key_tensor, key_tensor, 0.999)
    pcc_v = check_with_pcc(torch_value_tensor, value_tensor, 0.999)

    # Return minimum PCC of the three outputs
    pcc = min(pcc_q, pcc_k, pcc_v)

    return [pcc, e2e_perf]
