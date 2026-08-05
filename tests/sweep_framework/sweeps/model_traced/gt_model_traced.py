# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import torch

import ttnn
from models.common.utility_functions import torch_random

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    reconcile_golden_to_actual,
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

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("gt")

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
        "storage_type": ["StorageType::DEVICE"],
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
    memory_config=None,  # Alternative memory_config parameter from V2 traced configs
    dtype=None,  # Output dtype from V2 traced configs
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    scalar = kwargs.get("scalar", None)
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"scalar"}, output_memory_config=output_memory_config, device=device)

    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
    shape_b = (
        tuple(input_b_shape)
        if input_b_shape is not None and isinstance(input_b_shape, (list, tuple))
        else input_b_shape
    )

    # Determine if this is a binary (tensor-tensor) or tensor-scalar operation
    is_binary = shape_b is not None or input_b_dtype is not None

    # Create tensor A
    torch_input_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor_a = create_tensor_on_mesh(
                torch_input_a,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            input_tensor_a = ttnn.from_torch(
                torch_input_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        input_tensor_a = ttnn.from_torch(torch_input_a, dtype=input_a_dtype, layout=input_a_layout)

    if is_binary:
        if shape_b is None:
            shape_b = shape_a
        torch_input_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(shape_b)
        torch_output = ttnn.get_golden_function(ttnn.gt)(torch_input_a, torch_input_b)

        if not is_host:
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
            input_tensor_b = ttnn.from_torch(torch_input_b, dtype=input_b_dtype, layout=input_b_layout)

        # Pre-allocate output tensor if the master config recorded one
        output_tensor_info = extract_named_tensor_kwargs(kwargs, "output_tensor")
        if output_tensor_info and output_tensor_info.get("shape"):
            ot_shape_raw = output_tensor_info["shape"]
            if isinstance(ot_shape_raw, str):
                import ast

                ot_shape = tuple(ast.literal_eval(ot_shape_raw))
            else:
                ot_shape = tuple(ot_shape_raw)
            ot_dtype = output_tensor_info.get("dtype") or input_a_dtype
            if isinstance(ot_dtype, dict):
                ot_dtype = parse_dict_value("dtype", ot_dtype) or input_a_dtype
            elif isinstance(ot_dtype, str):
                ot_dtype = parse_dict_value("dtype", {"type": "DataType", "repr": ot_dtype}) or input_a_dtype
            ot_layout = output_tensor_info.get("layout") or input_a_layout
            if isinstance(ot_layout, dict):
                ot_layout = parse_dict_value("layout", ot_layout) or input_a_layout
            elif isinstance(ot_layout, str):
                ot_layout = parse_dict_value("layout", {"type": "Layout", "repr": ot_layout}) or input_a_layout
            ot_mem_cfg_raw = output_tensor_info.get("memory_config")
            if isinstance(ot_mem_cfg_raw, dict):
                from tests.sweep_framework.master_config_loader_v2 import dict_to_memory_config

                ot_mem_cfg = (
                    dict_to_memory_config(ot_mem_cfg_raw)
                    or parse_dict_value("memory_config", ot_mem_cfg_raw)
                    or input_a_memory_config
                )
            else:
                ot_mem_cfg = ot_mem_cfg_raw or input_a_memory_config
            ot_placement = output_tensor_info.get("tensor_placement")
            import torch as _torch_ot

            torch_out_alloc = _torch_ot.zeros(ot_shape, dtype=_torch_ot.float32)
            if is_mesh_device and ot_placement:
                op_kwargs["output_tensor"] = create_tensor_on_mesh(
                    torch_out_alloc, device, ot_dtype, ot_layout, ot_mem_cfg, ot_placement
                )
            elif not is_host:
                op_kwargs["output_tensor"] = ttnn.from_torch(
                    torch_out_alloc, dtype=ot_dtype, layout=ot_layout, device=device, memory_config=ot_mem_cfg
                )

        start_time = start_measuring_time()
        output_tensor = ttnn.gt(input_tensor_a, input_tensor_b, **op_kwargs)
        output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
        e2e_perf = stop_measuring_time(start_time)
    else:
        scalar_value = scalar if scalar is not None else 0
        torch_output = ttnn.get_golden_function(ttnn.gt)(torch_input_a, scalar_value)

        # Pre-allocate output tensor for scalar path too
        output_tensor_info = extract_named_tensor_kwargs(kwargs, "output_tensor")
        if output_tensor_info and output_tensor_info.get("shape"):
            import ast as _ast_gt

            ot_shape_raw = output_tensor_info["shape"]
            if isinstance(ot_shape_raw, str):
                ot_shape = tuple(_ast_gt.literal_eval(ot_shape_raw))
            else:
                ot_shape = tuple(ot_shape_raw)
            ot_dtype = output_tensor_info.get("dtype") or input_a_dtype
            if isinstance(ot_dtype, str):
                ot_dtype = parse_dict_value("dtype", {"type": "DataType", "repr": ot_dtype}) or input_a_dtype
            ot_layout = output_tensor_info.get("layout") or input_a_layout
            if isinstance(ot_layout, str):
                ot_layout = parse_dict_value("layout", {"type": "Layout", "repr": ot_layout}) or input_a_layout
            ot_mem_raw = output_tensor_info.get("memory_config")
            if isinstance(ot_mem_raw, dict):
                from tests.sweep_framework.master_config_loader_v2 import dict_to_memory_config

                ot_mem = dict_to_memory_config(ot_mem_raw) or input_a_memory_config
            else:
                ot_mem = ot_mem_raw or input_a_memory_config
            ot_placement = output_tensor_info.get("tensor_placement")
            import torch as _torch_gt

            torch_out_alloc = _torch_gt.zeros(ot_shape, dtype=_torch_gt.float32)
            if is_mesh_device and ot_placement:
                op_kwargs["output_tensor"] = create_tensor_on_mesh(
                    torch_out_alloc, device, ot_dtype, ot_layout, ot_mem, ot_placement
                )
            elif not is_host:
                op_kwargs["output_tensor"] = ttnn.from_torch(
                    torch_out_alloc, dtype=ot_dtype, layout=ot_layout, device=device, memory_config=ot_mem
                )

        start_time = start_measuring_time()
        output_tensor = ttnn.gt(input_tensor_a, scalar_value, **op_kwargs)
        output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
        e2e_perf = stop_measuring_time(start_time)

    # Comparison
    if is_mesh_device:
        torch_output = reconcile_golden_to_actual(torch_output, output_tensor, input_a_tensor_placement)
    return [check_with_pcc(torch_output.float(), output_tensor.float(), 0.999), e2e_perf]
