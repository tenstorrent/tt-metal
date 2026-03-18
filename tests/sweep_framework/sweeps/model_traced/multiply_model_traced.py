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
    infer_mesh_shape_from_params,
    detect_mesh_shape_from_hardware,
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, parse_dict_value

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("multiply")

parameters = {
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
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def _parse_activations(activations_list):
    """Parse a list of UnaryOpType dicts into ttnn.UnaryOpType enum values."""
    if not activations_list:
        return None
    parsed = []
    for item in activations_list:
        if isinstance(item, dict) and item.get("type") == "UnaryOpType":
            op_name = item["repr"].split(".")[-1]
            op_type = getattr(ttnn.UnaryOpType, op_name, None)
            if op_type is not None:
                parsed.append(op_type)
        else:
            parsed.append(item)
    return parsed if parsed else None


def mesh_device_fixture():
    mesh_shape = get_mesh_shape()

    if not mesh_shape:
        mesh_shape = infer_mesh_shape_from_params(model_traced_params)
    if not mesh_shape:
        mesh_shape = detect_mesh_shape_from_hardware()

    if mesh_shape:
        try:
            device = create_mesh_device(mesh_shape)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_mesh_device(device)
        except Exception as e:
            print(f"Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
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
    input_b_shape=None,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    arg1=None,
    use_legacy=None,
    memory_config=None,
    dtype=None,
    input_tensor_a_activations=None,
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    scalar = kwargs.get("scalar", None)
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)

    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"scalar"})

    # Conditionally add memory_config (build_op_kwargs excludes it by default)
    if memory_config is not None:
        op_kwargs["memory_config"] = parse_dict_value("memory_config", memory_config)

    # Conditionally add dtype
    if dtype is not None:
        op_kwargs["dtype"] = parse_dict_value("dtype", dtype)

    # Conditionally add input_tensor_a_activations
    parsed_activations = _parse_activations(input_tensor_a_activations)
    if parsed_activations is not None:
        op_kwargs["input_tensor_a_activations"] = parsed_activations

    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
    shape_b = tuple(input_b_shape) if input_b_shape and isinstance(input_b_shape, (list, tuple)) else input_b_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    # Apply fused activation to golden if present (e.g., SiLU fused with multiply)
    torch_golden_a = torch_input_tensor_a
    if parsed_activations:
        for act in parsed_activations:
            if hasattr(ttnn.UnaryOpType, "SILU") and act == ttnn.UnaryOpType.SILU:
                torch_golden_a = torch.nn.functional.silu(torch_golden_a)

    if shape_b is None:
        scalar_value = arg1 if arg1 is not None else (scalar if scalar is not None else 2.0)
        torch_output_tensor = torch.mul(torch_golden_a, scalar_value)
        is_scalar_multiply = True
    else:
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(shape_b)
        torch_output_tensor = torch.mul(torch_golden_a, torch_input_tensor_b)
        is_scalar_multiply = False

    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor_a = create_tensor_on_mesh(
                torch_input_tensor_a,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            input_tensor_a = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=input_a_layout)

    start_time = start_measuring_time()

    if is_scalar_multiply:
        scalar_value = arg1 if arg1 is not None else (scalar if scalar is not None else 2.0)
        output_tensor = ttnn.multiply(input_tensor_a, scalar_value, **op_kwargs)
    else:
        if not is_host:
            if is_mesh_device and input_b_tensor_placement:
                input_tensor_b = create_tensor_on_mesh(
                    torch_input_tensor_b,
                    device,
                    input_b_dtype,
                    input_b_layout,
                    input_b_memory_config,
                    input_b_tensor_placement,
                )
            else:
                input_tensor_b = ttnn.from_torch(
                    torch_input_tensor_b,
                    dtype=input_b_dtype,
                    layout=input_b_layout,
                    device=device,
                    memory_config=input_b_memory_config,
                )
        else:
            input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=input_b_dtype, layout=input_b_layout)

        output_tensor = ttnn.multiply(input_tensor_a, input_tensor_b, **op_kwargs)

    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
