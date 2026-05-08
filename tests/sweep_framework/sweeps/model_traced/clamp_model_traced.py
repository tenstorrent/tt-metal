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
from tests.sweep_framework.sweep_utils.op_kwargs_utils import (
    build_op_kwargs,
    extract_named_tensor_kwargs,
    parse_dict_value,
)
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    reconcile_golden_to_actual,
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
    """
    Override default device fixture.
    Creates mesh device if MESH_DEVICE_SHAPE is set, otherwise single device.
    """
    mesh_shape = get_mesh_shape()

    if mesh_shape:
        try:
            device = create_mesh_device(mesh_shape)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_mesh_device(device)
        except Exception as e:
            print(f"⚠️ Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
            device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


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

    # build_op_kwargs strips None values, but clamp needs min/max passed explicitly
    # even when None (the master trace records them). Re-inject from kwargs.
    if "min" not in op_kwargs and "min" in kwargs:
        op_kwargs["min"] = kwargs["min"]
    if "max" not in op_kwargs and "max" in kwargs:
        op_kwargs["max"] = kwargs["max"]

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

    # Pre-allocate output tensor if the master config recorded one
    output_tensor_info = extract_named_tensor_kwargs(kwargs, "output_tensor")
    if output_tensor_info and output_tensor_info.get("shape"):
        ot_shape = tuple(output_tensor_info["shape"])
        ot_dtype = output_tensor_info.get("dtype") or input_a_dtype
        if isinstance(ot_dtype, dict):
            ot_dtype = parse_dict_value("dtype", ot_dtype) or input_a_dtype
        ot_layout = output_tensor_info.get("layout") or input_a_layout
        if isinstance(ot_layout, dict):
            ot_layout = parse_dict_value("layout", ot_layout) or input_a_layout
        ot_mem_cfg_raw = output_tensor_info.get("memory_config")
        ot_mem_cfg = (
            parse_dict_value("memory_config", ot_mem_cfg_raw)
            if isinstance(ot_mem_cfg_raw, dict)
            else (ot_mem_cfg_raw or input_a_memory_config)
        )
        ot_placement = output_tensor_info.get("tensor_placement")
        torch_out_alloc = torch.zeros(ot_shape, dtype=torch.float32)
        if is_mesh_device and ot_placement:
            op_kwargs["output_tensor"] = create_tensor_on_mesh(
                torch_out_alloc, device, ot_dtype, ot_layout, ot_mem_cfg, ot_placement
            )
        elif not is_host:
            op_kwargs["output_tensor"] = ttnn.from_torch(
                torch_out_alloc, dtype=ot_dtype, layout=ot_layout, device=device, memory_config=ot_mem_cfg
            )

    start_time = start_measuring_time()
    output_tensor = ttnn.clamp(input_tensor_a, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(torch_output_tensor, output_tensor, input_a_tensor_placement)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
