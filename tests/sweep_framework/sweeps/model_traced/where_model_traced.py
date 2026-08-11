# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import torch

import ttnn
from models.common.utility_functions import torch_random
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    create_mesh_device,
    create_tensor_on_mesh,
    get_mesh_composer,
    get_model_traced_mesh_shape,
    mesh_tensor_to_torch,
)
from tests.sweep_framework.sweep_utils.op_kwargs_utils import (
    build_op_kwargs,
    extract_named_tensor_kwargs,
    parse_dict_value,
)
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time

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

    # Forward memory_config and dtype when master had them.
    if "memory_config" not in op_kwargs:
        traced_mc = kwargs.get("memory_config")
        if traced_mc is not None and traced_mc != "__ABSENT__":
            parsed_mc = parse_dict_value("memory_config", traced_mc) if isinstance(traced_mc, dict) else traced_mc
            if parsed_mc is not None:
                op_kwargs["memory_config"] = parsed_mc
    if "dtype" not in op_kwargs:
        traced_dt = kwargs.get("dtype")
        if traced_dt is not None and traced_dt != "__ABSENT__":
            parsed_dt = parse_dict_value("dtype", traced_dt) if isinstance(traced_dt, dict) else traced_dt
            if parsed_dt is not None:
                op_kwargs["dtype"] = parsed_dt

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
        import torch as _torch_ot

        torch_out_alloc = _torch_ot.zeros(ot_shape, dtype=_torch_ot.float32)
        is_host_check = storage_type and "HOST" in str(storage_type)
        if is_mesh_device and ot_placement:
            op_kwargs["output_tensor"] = create_tensor_on_mesh(
                torch_out_alloc, device, ot_dtype, ot_layout, ot_mem_cfg, ot_placement
            )
        elif not is_host_check:
            op_kwargs["output_tensor"] = ttnn.from_torch(
                torch_out_alloc, dtype=ot_dtype, layout=ot_layout, device=device, memory_config=ot_mem_cfg
            )

    if input_b_dtype == "__ABSENT__":
        input_b_dtype = None
    if input_c_dtype == "__ABSENT__":
        input_c_dtype = None
    if input_b_layout == "__ABSENT__":
        input_b_layout = None
    if input_b_memory_config == "__ABSENT__":
        input_b_memory_config = None
    if input_c_layout == "__ABSENT__":
        input_c_layout = None
    if input_c_memory_config == "__ABSENT__":
        input_c_memory_config = None
    is_ternary_tensor = input_b_dtype is not None and input_c_dtype is not None
    is_tensor_scalar = input_b_dtype is not None and input_c_dtype is None
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
    elif is_tensor_scalar:
        # Mixed: tensor true_value + scalar false_value
        shape_b = (
            tuple(kwargs.get("input_b_shape", shape_a))
            if kwargs.get("input_b_shape") and isinstance(kwargs.get("input_b_shape"), (tuple, list))
            else shape_a
        )
        if isinstance(shape_b, str):
            import json as _json_wb

            if len(shape_b) < 200:
                shape_b = tuple(_json_wb.loads(shape_b.replace("(", "[").replace(")", "]")))
        torch_condition = torch.randint(0, 2, shape_a, dtype=torch.float32)
        torch_input_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(shape_b)
        scalar_false = float(kwargs.get("arg2", 0.0)) if kwargs.get("arg2") is not None else 0.0
        torch_output = torch.where(torch_condition > 0, torch_input_b, scalar_false)

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
        else:
            condition_tensor = ttnn.from_torch(torch_condition, dtype=input_a_dtype, layout=input_a_layout)
            input_tensor_b = ttnn.from_torch(torch_input_b, dtype=input_b_dtype, layout=input_b_layout)

        # Apply topology to match master
        if is_mesh_device and input_a_tensor_placement:
            from tests.sweep_framework.sweep_utils.mesh_tensor_utils import apply_tensor_placement_topology

            try:
                apply_tensor_placement_topology(condition_tensor, input_a_tensor_placement, (1, 2))
            except Exception:
                pass  # Topology application is best-effort; mismatch is non-fatal
        if is_mesh_device and input_b_tensor_placement:
            from tests.sweep_framework.sweep_utils.mesh_tensor_utils import apply_tensor_placement_topology

            try:
                apply_tensor_placement_topology(input_tensor_b, input_b_tensor_placement, (1, 2))
            except Exception:
                pass  # Topology application is best-effort; mismatch is non-fatal

        start_time = start_measuring_time()
        output_tensor = ttnn.where(condition_tensor, input_tensor_b, scalar_false, **op_kwargs)
        output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
        e2e_perf = stop_measuring_time(start_time)

        from tests.sweep_framework.sweep_utils.mesh_tensor_utils import reconcile_golden_to_actual

        if is_mesh_device:
            torch_output = reconcile_golden_to_actual(torch_output, output_tensor, input_a_tensor_placement)

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
    if is_mesh_device:
        from tests.sweep_framework.sweep_utils.mesh_tensor_utils import reconcile_golden_to_actual

        torch_output = reconcile_golden_to_actual(torch_output, output_tensor, input_a_tensor_placement)
    return [check_with_pcc(torch_output, output_tensor, 0.999), e2e_perf]
