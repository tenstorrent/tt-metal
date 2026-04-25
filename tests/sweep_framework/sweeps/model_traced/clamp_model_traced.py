# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_named_tensor_kwargs, parse_dict_value
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_model_traced_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    get_mesh_composer,
)

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

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


def mesh_device_fixture():
    mesh_shape = get_model_traced_mesh_shape()
    device = create_mesh_device(mesh_shape)
    device_name = ttnn.get_arch_name()
    yield (device, device_name)
    ttnn.close_mesh_device(device)

def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Extract placement information from kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)

    # Check if device is mesh device
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    # Check for output_tensor named tensor kwarg (pre-allocated output tensor)
    output_tensor_info = extract_named_tensor_kwargs(kwargs, "output_tensor")

    # The master trace may record output_tensor=None and min=None as explicit kwargs.
    # build_op_kwargs filters None values, so add them back when present in the test vector.
    for key in ("output_tensor", "min"):
        if key in kwargs and kwargs[key] is None and key not in op_kwargs:
            op_kwargs[key] = None

    # Extract min/max from op_kwargs for golden computation (avoid shadowing Python built-ins)
    min_val = op_kwargs.get("min", None)
    max_val = op_kwargs.get("max", None)

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

    # Create pre-allocated output tensor if the master config specifies one
    if output_tensor_info is not None and not is_host:
        ot_shape = tuple(output_tensor_info["shape"]) if output_tensor_info["shape"] else shape
        ot_dtype = output_tensor_info.get("dtype") or input_a_dtype
        ot_layout = output_tensor_info.get("layout") or input_a_layout
        ot_mem_cfg = output_tensor_info.get("memory_config") or input_a_memory_config
        ot_dtype = parse_dict_value("output_tensor_dtype", ot_dtype) if isinstance(ot_dtype, dict) else ot_dtype
        ot_layout = parse_dict_value("output_tensor_layout", ot_layout) if isinstance(ot_layout, dict) else ot_layout
        ot_mem_cfg = parse_dict_value("output_tensor_memory_config", ot_mem_cfg) if isinstance(ot_mem_cfg, dict) else ot_mem_cfg
        ot_placement = output_tensor_info.get("tensor_placement")

        torch_preallocated = torch.zeros(shape, dtype=torch.float32)
        if is_mesh_device and ot_placement:
            preallocated_output = create_tensor_on_mesh(
                torch_preallocated, device, ot_dtype, ot_layout, ot_mem_cfg, ot_placement,
            )
        else:
            preallocated_output = ttnn.from_torch(
                torch_preallocated, dtype=ot_dtype, layout=ot_layout, device=device, memory_config=ot_mem_cfg,
            )
        op_kwargs["output_tensor"] = preallocated_output

    start_time = start_measuring_time()
    output_tensor = ttnn.clamp(input_tensor_a, **op_kwargs)
    mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
