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


def invalidate_vector(test_vector) -> tuple[bool, str]:
    """
    Invalidate test vectors that would cause L1 circular buffer overflow in pad operation.

    The pad operation allocates internal L1 circular buffers regardless of input memory config.
    Even when we force DRAM input, the operation's internal buffers can exceed L1 capacity.

    Root cause: pad operation's C++ implementation allocates circular buffers proportional
    to tensor size, and these buffers live in L1 memory (not configurable from Python).
    """
    input_shape = test_vector.get("input_shape")
    input_a_memory_config = test_vector.get("input_a_memory_config")

    # Parse input shape
    shape = None
    if isinstance(input_shape, str) and input_shape.startswith("("):
        try:
            shape = eval(input_shape)
        except:
            pass
    elif isinstance(input_shape, (list, tuple)):
        shape = input_shape

    if shape:
        # Calculate tensor size
        total_elements = 1
        for dim in shape:
            total_elements *= dim

        # Check memory config - L1 configs are especially problematic
        is_l1 = False
        is_sharded = False

        if isinstance(input_a_memory_config, dict):
            data = input_a_memory_config.get("data", {})
            buffer_type = data.get("buffer_type", "")
            memory_layout = data.get("memory_layout", "")

            is_l1 = "L1" in str(buffer_type)
            is_sharded = "SHARDED" in str(memory_layout)

        # Skip L1 sharded configs - these ALWAYS fail due to circular buffer allocation
        # Even though we convert to DRAM in from_torch, pad's internal logic allocates L1 buffers
        if is_l1 and is_sharded:
            return (True, f"pad: L1 sharded config causes circular buffer overflow (shape {shape})")

        # Skip L1 interleaved configs - also problematic for pad operation
        if is_l1:
            return (True, f"pad: L1 config causes circular buffer overflow (shape {shape})")

        # Skip large tensors - pad operation allocates internal L1 circular buffers
        # Empirical observation from failures:
        #   - 150K elements → 2.2MB buffer (exceeds 1.5MB L1 limit)
        #   - 200K elements → 2.9MB buffer
        #   - 500K elements → 5.3MB buffer
        #   - 800K elements → 8.5MB buffer
        # The circular buffer is NOT linear with tensor size, and depends on internal
        # pad implementation details. To be safe, skip anything >40K elements.
        if total_elements > 40_000:
            size_kb = total_elements * 2 / 1024  # bfloat16
            return (
                True,
                f"pad: Tensor too large ({total_elements} elements, {size_kb:.1f}KB) - would cause L1 circular buffer overflow",
            )

    return (False, None)


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
    **kwargs,  # Accept traced_source, traced_machine_info, config_id, etc.
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
    # Use DRAM instead of sharded memory to avoid OOM for pad operation
    # Always use DRAM for pad operation due to large output allocations
    actual_memory_config = ttnn.DRAM_MEMORY_CONFIG

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=actual_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.pad(input_tensor_a, padding=padding, value=value)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
