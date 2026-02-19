# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("concat")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "arg0": [
            [
                {"shape": (1, 1, 32, 16), "dtype": "ttnn.bfloat16", "layout": "ttnn.TILE_LAYOUT"},
                {"shape": (1, 1, 32, 8), "dtype": "ttnn.bfloat16", "layout": "ttnn.TILE_LAYOUT"},
            ]
        ],
        "dim": [3],
        "memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    """
    Override default device fixture.
    Creates mesh device if MESH_DEVICE_SHAPE is set, otherwise single device.
    """
    mesh_shape = get_mesh_shape()

    if mesh_shape:
        # Create mesh device based on env var
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
        # Single device (default)
        device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def run(
    arg0,  # List of tensor specs: [{"shape": ..., "dtype": ..., "layout": ..., "tensor_placement": ...}, ...]
    arg1=None,  # dim value (positional in JSON)
    output_memory_config=None,
    memory_config=None,
    storage_type="StorageType::DEVICE",
    dim=None,  # dim as kwarg (fallback)
    *,
    device,
    **kwargs,  # Accept traced_source, traced_machine_info, etc.
) -> list:
    torch.manual_seed(0)

    # Handle dim parameter - can be arg1 (positional) or dim (kwarg)
    dim_value = arg1 if arg1 is not None else dim
    if dim_value is None:
        dim_value = -1  # Default concat dimension

    # Handle memory_config - prefer output_memory_config, fallback to memory_config
    mem_config = output_memory_config if output_memory_config is not None else memory_config

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")  # MeshDevice has this method

    # Check if storage_type is HOST
    is_host = storage_type and "HOST" in str(storage_type)

    # Process arg0 - it's a list of tensor specifications
    if not isinstance(arg0, list):
        raise ValueError(f"arg0 must be a list of tensor specs, got {type(arg0)}")

    torch_tensors = []
    ttnn_tensors = []

    for i, tensor_spec in enumerate(arg0):
        # Extract tensor spec
        shape = tensor_spec.get("shape")
        dtype_str = tensor_spec.get("dtype", "ttnn.bfloat16")
        layout_str = tensor_spec.get("layout", "ttnn.TILE_LAYOUT")
        tensor_placement = tensor_spec.get("tensor_placement", None)

        # Convert strings to actual ttnn objects if needed
        if isinstance(dtype_str, str):
            dtype = eval(dtype_str) if "ttnn." in dtype_str else ttnn.bfloat16
        else:
            dtype = dtype_str

        if isinstance(layout_str, str):
            layout = eval(layout_str) if "ttnn." in layout_str else ttnn.TILE_LAYOUT
        else:
            layout = layout_str

        # Generate torch tensor
        torch_tensor = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), dtype)(
            tuple(shape) if isinstance(shape, list) else shape
        )
        torch_tensors.append(torch_tensor)

        # Create ttnn tensor
        if not is_host:
            if is_mesh_device and tensor_placement:
                # Use mesh with placement
                ttnn_tensor = create_tensor_on_mesh(
                    torch_tensor,
                    device,
                    dtype,
                    layout,
                    mem_config,
                    tensor_placement,
                )
            else:
                # Regular single-device tensor
                ttnn_tensor = ttnn.from_torch(
                    torch_tensor,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    memory_config=mem_config,
                )
        else:
            # Host storage
            ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout)

        ttnn_tensors.append(ttnn_tensor)

    # Compute expected output with torch
    torch_output_tensor = torch.cat(torch_tensors, dim=dim_value)

    start_time = start_measuring_time()
    output_tensor = ttnn.concat(ttnn_tensors, dim=dim_value, memory_config=mem_config)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
