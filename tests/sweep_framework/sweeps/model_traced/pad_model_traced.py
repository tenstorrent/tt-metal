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
        "storage_type": [
            "StorageType::DEVICE"
        ],  # NOTE: HOST storage does not work properly for pad - always use DEVICE
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> tuple:
    """
    Invalidate test vectors that will fail due to memory or resource constraints.
    Also skips HOST operations (weight/bias padding done on CPU during model init).
    """
    # Skip all HOST operations - these are CPU-side preprocessing, not device operations
    storage_type = test_vector.get("storage_type")
    if storage_type and "HOST" in str(storage_type):
        return True, "HOST storage operation: CPU-side preprocessing, not a device operation to test"

    # All DEVICE operations pass - MasterConfigLoader bug is fixed
    return False, None


def mesh_device_fixture():
    """
    Override default device fixture for pad operation.
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
    padding=None,
    value=0.0,
    output_padded_shape=None,
    input_tensor_start=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
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

    # Calculate padding tuples and PyTorch reference
    # Handle both parameter formats from loader:
    # 1. padding + value (direct format)
    # 2. output_padded_shape + input_tensor_start + value (alternative format)
    # PREFER padding if provided directly, as it's more reliable
    if padding is not None:
        # Use provided padding directly (preferred format)
        pass
    elif output_padded_shape is not None and input_tensor_start is not None:
        # Calculate padding from output_padded_shape (alternative format)
        calculated_padding = []
        for i in range(len(shape)):
            start = input_tensor_start[i] if i < len(input_tensor_start) else 0
            end = output_padded_shape[i] - shape[i] - start
            calculated_padding.append([start, max(0, end)])
        padding = calculated_padding
    else:
        # No padding parameters provided - use default no padding
        padding = [[0, 0]] * len(shape)

    # Calculate PyTorch reference
    torch_padding = []
    for i in range(len(padding) - 1, -1, -1):
        for p in padding[i]:
            torch_padding.append(p)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor_a, torch_padding, mode="constant", value=value)

    # Convert padding to tuple format for ttnn.pad
    if isinstance(padding, list):
        padding = tuple(tuple(p) if isinstance(p, (list, tuple)) else p for p in padding)

    # NOTE: HOST storage does not work properly for pad operation - always use DEVICE
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.pad(input_tensor_a, padding=padding, value=value)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
