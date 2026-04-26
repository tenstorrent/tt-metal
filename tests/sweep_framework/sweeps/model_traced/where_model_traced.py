# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_model_traced_mesh_shape,
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
    input_b_shape=None,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    input_c_shape=None,
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
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config,
    )

    is_ternary_tensor = input_b_dtype is not None and input_c_dtype is not None
    # Mixed mode: tensor for arg1 (true branch) + scalar for arg2 (false branch)
    is_mixed_tensor_scalar = input_b_dtype is not None and input_c_dtype is None
    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (tuple, list)) else input_a_shape

    # Extract scalar values from kwargs (arg1/arg2 may carry scalars)
    arg1_val = kwargs.get("arg1", None)
    arg2_val = kwargs.get("arg2", None)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    if is_ternary_tensor:
        # Tensor creation — use per-input shapes when available (broadcast support)
        shape_b = tuple(input_b_shape) if input_b_shape is not None else shape_a
        shape_c = tuple(input_c_shape) if input_c_shape is not None else shape_a
        torch_condition = torch.randint(0, 2, shape_a, dtype=torch.float32)
        torch_input_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(shape_b)
        torch_input_c = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_c_dtype
        )(shape_c)
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
    elif is_mixed_tensor_scalar:
        # Mixed mode: tensor for true branch, scalar for false branch
        # (or vice versa — determined by master trace arg positions)
        try:
            scalar_value = float(arg2_val) if arg2_val is not None else (float(scalar_if_false) if scalar_if_false is not None else 0.0)
        except (ValueError, TypeError):
            scalar_value = 0.0

        torch_condition = torch.randint(0, 2, shape_a, dtype=torch.float32)
        b_layout = input_b_layout if input_b_layout is not None else input_a_layout
        b_mem_config = input_b_memory_config if input_b_memory_config is not None else input_a_memory_config
        shape_b = tuple(input_b_shape) if input_b_shape is not None else shape_a
        torch_input_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(shape_b)
        torch_output = torch.where(torch_condition > 0, torch_input_b, scalar_value)

        if not is_host:
            if is_mesh_device and input_a_tensor_placement:
                condition_tensor = create_tensor_on_mesh(
                    torch_condition, device, input_a_dtype, input_a_layout,
                    input_a_memory_config, input_a_tensor_placement,
                )
            else:
                condition_tensor = ttnn.from_torch(
                    torch_condition, dtype=input_a_dtype, layout=input_a_layout,
                    device=device, memory_config=input_a_memory_config,
                )
            if is_mesh_device and input_b_tensor_placement:
                input_tensor_b = create_tensor_on_mesh(
                    torch_input_b, device, input_b_dtype, b_layout,
                    b_mem_config, input_b_tensor_placement,
                )
            else:
                input_tensor_b = ttnn.from_torch(
                    torch_input_b, dtype=input_b_dtype, layout=b_layout,
                    device=device, memory_config=b_mem_config,
                )
        else:
            condition_tensor = ttnn.from_torch(torch_condition, dtype=input_a_dtype, layout=input_a_layout)
            input_tensor_b = ttnn.from_torch(torch_input_b, dtype=input_b_dtype, layout=b_layout)

        start_time = start_measuring_time()
        output_tensor = ttnn.where(condition_tensor, input_tensor_b, scalar_value, **op_kwargs)
        output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
        e2e_perf = stop_measuring_time(start_time)
    else:
        # Scalar-only mode: both true and false branches are scalars
        try:
            scalar_true = float(scalar_if_true) if scalar_if_true is not None else (float(arg1_val) if arg1_val is not None else 1.0)
        except (ValueError, TypeError):
            scalar_true = 1.0
        try:
            scalar_false = float(scalar_if_false) if scalar_if_false is not None else (float(arg2_val) if arg2_val is not None else 0.0)
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
