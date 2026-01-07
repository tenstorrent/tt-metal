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
model_traced_params = loader.get_suite_parameters("sharded_to_interleaved", all_cases=False)

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


def mesh_device_fixture():
    """
    Override default device fixture for sharded_to_interleaved operation.
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
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Handle input_shape - ensure it's always a tuple
    # This is a simple operation: convert sharded tensor to interleaved
    # Convert input_shape to tuple - handle all possible types
    if input_shape is None:
        raise ValueError("input_shape cannot be None")

    # Handle list/tuple
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    # Handle single int/float (invalid, but provide clear error)
    elif isinstance(input_shape, (int, float)):
        raise ValueError(f"input_shape must be a list or tuple, got {type(input_shape).__name__}: {input_shape}")
    # Handle other iterables (numpy arrays, etc.)
    else:
        try:
            shape = tuple(input_shape)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Cannot convert input_shape to tuple: {type(input_shape).__name__} = {input_shape}. Error: {e}"
            )

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # Check if input is already interleaved or needs to be sharded
    is_input_sharded = (
        hasattr(input_a_memory_config, "memory_layout")
        and input_a_memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    )

    if is_input_sharded:
        # Input should be sharded - create interleaved first, then convert to sharded
        input_tensor_interleaved = ttnn.from_torch(
            torch_input_tensor_a,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Convert to sharded using the traced config
        try:
            input_tensor = ttnn.interleaved_to_sharded(input_tensor_interleaved, input_a_memory_config)
        except (RuntimeError, ValueError) as e:
            error_msg = str(e)
            if "No core coordinate found" in error_msg or "core coordinate" in error_msg.lower():
                raise ValueError(
                    f"Invalid core coordinates in sharding config: {error_msg}. "
                    f"This traced config uses cores that don't exist on this device."
                )
            raise
    else:
        # Input is interleaved - use the traced config directly (op supports this)
        input_tensor = ttnn.from_torch(
            torch_input_tensor_a,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=input_a_memory_config,
        )

    # Run sharded_to_interleaved
    start_time = start_measuring_time()
    output_tensor = ttnn.sharded_to_interleaved(input_tensor, memory_config=output_memory_config)
    e2e_perf = stop_measuring_time(start_time)

    # Verify output is interleaved
    output_mem_config = output_tensor.memory_config()
    if output_mem_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(
            f"sharded_to_interleaved should produce interleaved output, but got {output_mem_config.memory_layout}"
        )

    # Verify correctness by comparing with original torch tensor
    output_torch = ttnn.to_torch(output_tensor)
    pcc = check_with_pcc(torch_input_tensor_a, output_torch, 0.999)

    return [pcc, e2e_perf]
