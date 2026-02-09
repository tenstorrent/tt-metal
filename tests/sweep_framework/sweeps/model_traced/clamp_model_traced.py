# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    get_mesh_shape_from_machine_info,
    mesh_tensor_to_torch,
)

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("clamp")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "min": [-10.0],
        "max": [10.0],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> tuple:
    """
    Validate test vector (non-mesh related checks).

    Note: Mesh shape filtering happens at runtime in run() function,
    not during vector generation, since MESH_DEVICE_SHAPE env var
    is only set during test execution on specific runners.

    Returns:
        Tuple of (is_invalid: bool, reason: str or None)
    """
    # Add any static validation logic here (e.g., parameter constraints)
    # Mesh filtering is handled in run() function at execution time
    return False, None


def mesh_device_fixture():
    """
    Override default device fixture.
    Creates mesh device if MESH_DEVICE_SHAPE is set, otherwise single device.
    """
    import ttnn

    mesh_shape = get_mesh_shape()

    if mesh_shape:
        try:
            device = create_mesh_device(mesh_shape)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_mesh_device(device)
        except Exception as e:
            print(f"⚠️ Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
            device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def run(
    input_a_shape,
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

    # Extract min/max from kwargs (avoid shadowing Python built-ins)
    min_val = kwargs.get("min", None)
    max_val = kwargs.get("max", None)

    # Extract placement information from kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)

    # Check if device is mesh device
    is_mesh_device = hasattr(device, "get_num_devices")

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # V2 format provides input_a_shape
    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    torch_output_tensor = torch.clamp(torch_input_tensor_a, min=min_val, max=max_val)

    # Convert to ttnn tensor (mesh or single device)
    if is_host:
        # HOST storage - no device or memory config
        input_tensor_a = ttnn.from_torch(
            torch_input_tensor_a,
            dtype=input_a_dtype,
            layout=input_a_layout,
        )
    elif is_mesh_device and input_a_tensor_placement:
        # Mesh device with placement
        input_tensor_a = create_tensor_on_mesh(
            torch_input_tensor_a,
            device,
            input_a_dtype,
            input_a_layout,
            input_a_memory_config,
            input_a_tensor_placement,
        )
    else:
        # Single device
        input_tensor_a = ttnn.from_torch(
            torch_input_tensor_a,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=input_a_memory_config,
        )

    start_time = start_measuring_time()
    output_tensor = ttnn.clamp(input_tensor_a, min=min_val, max=max_val, memory_config=output_memory_config)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
