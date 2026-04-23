# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_model_traced_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_named_tensor_kwargs, parse_dict_value

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("add")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_shape": [(1, 1, 32, 32)],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
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
    input_b_shape=None,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    arg1=None,  # May contain scalar value from V2 traced configs
    use_legacy=None,  # Legacy mode flag from V2 traced configs
    memory_config=None,  # Alternative memory_config parameter from V2 traced configs
    dtype=None,  # Output dtype from V2 traced configs
    *,
    device,
    **kwargs,  # Accept placements, traced_source, traced_machine_info, etc.
) -> list:
    torch.manual_seed(0)

    # Extract kwargs - arg1 is now a named param, use it as scalar fallback
    scalar = kwargs.get("scalar", arg1)
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")  # MeshDevice has this method
    op_kwargs = build_op_kwargs(kwargs, exclude={"scalar"}, output_memory_config=output_memory_config)

    # Pass through memory_config kwarg when present in traced config
    if memory_config is not None:
        parsed_mc = parse_dict_value("memory_config", memory_config)
        if parsed_mc is not None:
            op_kwargs["memory_config"] = parsed_mc

    # Pass through dtype kwarg when present in traced config
    if dtype is not None:
        parsed_dtype = parse_dict_value("dtype", dtype)
        if parsed_dtype is not None:
            op_kwargs["dtype"] = parsed_dtype

    # Check for output_tensor named tensor kwarg (pre-allocated output tensor)
    output_tensor_info = extract_named_tensor_kwargs(kwargs, "output_tensor")

    # V2 format provides separate shapes for each input
    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
    shape_b = tuple(input_b_shape) if input_b_shape and isinstance(input_b_shape, (list, tuple)) else input_b_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    # Check if this is a scalar add operation (shape_b is None or scalar is provided)
    if shape_b is None or scalar is not None:
        # Tensor-scalar add: use the scalar value directly
        # If scalar is None but shape_b is None, default to scalar=1.0
        scalar_value = scalar if scalar is not None else 1.0
        torch_output_tensor = torch.add(torch_input_tensor_a, scalar_value)
        is_scalar_add = True
    else:
        # Tensor-tensor add: generate second tensor
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(shape_b)
        torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)
        is_scalar_add = False

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Create first tensor (with mesh support if device is mesh)
    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            # Use mesh with placement
            input_tensor_a = create_tensor_on_mesh(
                torch_input_tensor_a,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            # Regular single-device tensor
            input_tensor_a = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        # Host storage
        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=input_a_layout)

    # Create pre-allocated output tensor if the master config specifies one
    preallocated_output = None
    if output_tensor_info is not None and not is_host:
        ot_shape = tuple(output_tensor_info["shape"]) if output_tensor_info["shape"] else shape_a
        ot_dtype = output_tensor_info.get("dtype") or input_a_dtype
        ot_layout = output_tensor_info.get("layout") or input_a_layout
        ot_mem_cfg = output_tensor_info.get("memory_config") or input_a_memory_config
        # Parse dict values if needed
        ot_dtype = parse_dict_value("output_tensor_dtype", ot_dtype) if isinstance(ot_dtype, dict) else ot_dtype
        ot_layout = parse_dict_value("output_tensor_layout", ot_layout) if isinstance(ot_layout, dict) else ot_layout
        ot_mem_cfg = parse_dict_value("output_tensor_memory_config", ot_mem_cfg) if isinstance(ot_mem_cfg, dict) else ot_mem_cfg
        ot_placement = output_tensor_info.get("tensor_placement")

        torch_preallocated = torch.zeros(ot_shape, dtype=torch.float32)
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

    if is_scalar_add:
        # Tensor-scalar add: pass scalar directly
        scalar_value = scalar if scalar is not None else 1.0
        output_tensor = ttnn.add(input_tensor_a, scalar_value, **op_kwargs)
    else:
        # Tensor-tensor add: convert second tensor and add
        if not is_host:
            if is_mesh_device and input_b_tensor_placement:
                # Use mesh with placement for second tensor
                input_tensor_b = create_tensor_on_mesh(
                    torch_input_tensor_b,
                    device,
                    input_b_dtype,
                    input_b_layout,
                    input_b_memory_config,
                    input_b_tensor_placement,
                )
            else:
                # Regular single-device tensor
                input_tensor_b = ttnn.from_torch(
                    torch_input_tensor_b,
                    dtype=input_b_dtype,
                    layout=input_b_layout,
                    device=device,
                    memory_config=input_b_memory_config,
                )
        else:
            # Host storage
            input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=input_b_dtype, layout=input_b_layout)

        output_tensor = ttnn.add(input_tensor_a, input_tensor_b, **op_kwargs)

    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
