# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_model_traced_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    reconcile_golden_to_actual,
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, parse_dict_value

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("zeros_like")

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
    memory_config=None,
    dtype=None,
    layout=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    # Re-inject memory_config from kwargs (build_op_kwargs strips it by default)
    mc_raw = kwargs.get("memory_config")
    if mc_raw is not None and "memory_config" not in op_kwargs:
        parsed_mc = parse_dict_value("memory_config", mc_raw) if isinstance(mc_raw, dict) else mc_raw
        if parsed_mc is not None:
            op_kwargs["memory_config"] = parsed_mc
    elif memory_config is not None and "memory_config" not in op_kwargs:
        op_kwargs["memory_config"] = memory_config

    # Re-inject dtype and layout if provided
    if dtype is not None and "dtype" not in op_kwargs:
        parsed_dt = parse_dict_value("dtype", dtype) if isinstance(dtype, dict) else dtype
        if parsed_dt is not None:
            op_kwargs["dtype"] = parsed_dt
    if layout is not None and "layout" not in op_kwargs:
        parsed_lt = parse_dict_value("layout", layout) if isinstance(layout, dict) else layout
        if parsed_lt is not None:
            op_kwargs["layout"] = parsed_lt

    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    # Create input tensor
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_output = torch.zeros_like(torch_input)

    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor = create_tensor_on_mesh(
                torch_input, device, input_a_dtype, input_a_layout, input_a_memory_config, input_a_tensor_placement
            )
        else:
            input_tensor = ttnn.from_torch(
                torch_input,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        input_tensor = ttnn.from_torch(torch_input, dtype=input_a_dtype, layout=input_a_layout)

    start_time = start_measuring_time()
    output_tensor = ttnn.zeros_like(input_tensor, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    if is_mesh_device:
        torch_output = reconcile_golden_to_actual(torch_output, output_tensor, input_a_tensor_placement)
    pcc = check_with_pcc(torch_output, output_tensor, 0.999)
    return [pcc, e2e_perf]
