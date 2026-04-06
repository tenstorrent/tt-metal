# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("where")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    mesh_shape = get_mesh_shape()
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
    output_memory_config=None,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    input_c_dtype=None,
    input_c_layout=None,
    input_c_memory_config=None,
    scalar_if_true=None,
    scalar_if_false=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)
    input_c_tensor_placement = kwargs.get("input_c_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    is_ternary_tensor = input_b_dtype is not None and input_c_dtype is not None
    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (tuple, list)) else input_a_shape

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    if is_ternary_tensor:
        # Tensor creation
        torch_condition = torch.randint(0, 2, shape_a, dtype=torch.float32)
        torch_input_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(shape_a)
        torch_input_c = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_c_dtype
        )(shape_a)
        torch_output = torch.where(torch_condition > 0, torch_input_b, torch_input_c)

        if not is_host:
            if is_mesh_device and input_a_tensor_placement:
                condition_tensor = create_tensor_on_mesh(
                    torch_condition,
                    device,
                    input_a_dtype,
                    input_a_layout,
                    input_a_memory_config,
                    input_a_tensor_placement,
                )
            else:
                condition_tensor = ttnn.from_torch(
                    torch_condition,
                    dtype=input_a_dtype,
                    layout=input_a_layout,
                    device=device,
                    memory_config=input_a_memory_config,
                )

            if is_mesh_device and input_b_tensor_placement:
                input_tensor_b = create_tensor_on_mesh(
                    torch_input_b,
                    device,
                    input_b_dtype,
                    input_b_layout,
                    input_b_memory_config,
                    input_b_tensor_placement,
                )
            else:
                input_tensor_b = ttnn.from_torch(
                    torch_input_b,
                    dtype=input_b_dtype,
                    layout=input_b_layout,
                    device=device,
                    memory_config=input_b_memory_config,
                )

            if is_mesh_device and input_c_tensor_placement:
                input_tensor_c = create_tensor_on_mesh(
                    torch_input_c,
                    device,
                    input_c_dtype,
                    input_c_layout,
                    input_c_memory_config,
                    input_c_tensor_placement,
                )
            else:
                input_tensor_c = ttnn.from_torch(
                    torch_input_c,
                    dtype=input_c_dtype,
                    layout=input_c_layout,
                    device=device,
                    memory_config=input_c_memory_config,
                )
        else:
            condition_tensor = ttnn.from_torch(torch_condition, dtype=input_a_dtype, layout=input_a_layout)
            input_tensor_b = ttnn.from_torch(torch_input_b, dtype=input_b_dtype, layout=input_b_layout)
            input_tensor_c = ttnn.from_torch(torch_input_c, dtype=input_c_dtype, layout=input_c_layout)

        # Op call
        start_time = start_measuring_time()
        output_tensor = ttnn.where(condition_tensor, input_tensor_b, input_tensor_c, **op_kwargs)
        output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
        e2e_perf = stop_measuring_time(start_time)
    else:
        # Tensor creation
        try:
            scalar_true = float(scalar_if_true) if scalar_if_true is not None else 1.0
        except (ValueError, TypeError):
            scalar_true = 1.0
        try:
            scalar_false = float(scalar_if_false) if scalar_if_false is not None else 0.0
        except (ValueError, TypeError):
            scalar_false = 0.0
        torch_condition = torch.randint(0, 2, shape_a, dtype=torch.float32)
        torch_output = torch.where(torch_condition > 0, scalar_true, scalar_false)

        if not is_host:
            if is_mesh_device and input_a_tensor_placement:
                condition_tensor = create_tensor_on_mesh(
                    torch_condition,
                    device,
                    input_a_dtype,
                    input_a_layout,
                    input_a_memory_config,
                    input_a_tensor_placement,
                )
            else:
                condition_tensor = ttnn.from_torch(
                    torch_condition,
                    dtype=input_a_dtype,
                    layout=input_a_layout,
                    device=device,
                    memory_config=input_a_memory_config,
                )
        else:
            condition_tensor = ttnn.from_torch(torch_condition, dtype=input_a_dtype, layout=input_a_layout)

        # Op call
        start_time = start_measuring_time()
        output_tensor = ttnn.where(condition_tensor, scalar_true, scalar_false, **op_kwargs)
        output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
        e2e_perf = stop_measuring_time(start_time)

    # Comparison
    return [check_with_pcc(torch_output, output_tensor, 0.999), e2e_perf]
