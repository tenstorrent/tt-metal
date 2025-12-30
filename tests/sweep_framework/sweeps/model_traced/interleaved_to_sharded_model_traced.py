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
model_traced_params = loader.get_suite_parameters("interleaved_to_sharded", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector):
    """
    Validate that the output memory config is sharded.
    interleaved_to_sharded requires a sharded output memory config.
    """
    from typing import Tuple, Optional

    output_memory_config = test_vector.get("output_memory_config")

    # Check if output is sharded
    is_output_sharded = False

    # Handle dict memory configs (from JSON deserialization in pipeline)
    if isinstance(output_memory_config, dict):
        # Check dict structure for memory_layout
        mem_layout = output_memory_config.get("memory_layout") or output_memory_config.get("data", {}).get(
            "memory_layout"
        )
        if mem_layout and "INTERLEAVED" in str(mem_layout):
            return True, "Output memory config must be sharded for interleaved_to_sharded operation"
        # If it's a dict with SHARDED layout, consider it valid
        if mem_layout and "SHARDED" in str(mem_layout):
            is_output_sharded = True
    else:
        # Handle proper ttnn MemoryConfig objects
        if hasattr(output_memory_config, "is_sharded") and callable(getattr(output_memory_config, "is_sharded", None)):
            is_output_sharded = output_memory_config.is_sharded()
        elif hasattr(output_memory_config, "memory_layout"):
            is_output_sharded = output_memory_config.memory_layout in [
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ]

    if not is_output_sharded:
        return True, "Output memory config must be sharded for interleaved_to_sharded operation"

    return False, None


def mesh_device_fixture():
    """
    Override default device fixture for interleaved_to_sharded operation.
    Using explicit DispatchCoreConfig to handle sharded memory configs.
    """
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
    storage_type="StorageType::DEVICE",
    *,
    device,
) -> list:
    torch.manual_seed(0)

    # Handle tuple input_shape
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # For interleaved_to_sharded, the output is the same tensor but in sharded memory layout
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

    # Use the traced output_memory_config directly - no hardcoding
    # The traced config contains the exact memory layout and shard spec from real model runs
    if output_memory_config is None:
        raise ValueError("output_memory_config is None - required parameter missing from traced config")

    start_time = start_measuring_time()
    output_tensor = ttnn.interleaved_to_sharded(input_tensor_a, output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC - should be identical since it's just a memory layout change
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
