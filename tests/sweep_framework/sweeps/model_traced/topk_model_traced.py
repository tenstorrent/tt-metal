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

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_positional_args, extract_named_tensor_kwargs, parse_dict_value

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("topk")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "k": [5],
        "dim": [-1],
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
    k=None,
    dim=None,
    output_memory_config=None,
    memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"arg1", "arg2"}, output_memory_config=output_memory_config)

    pos_args = extract_positional_args(kwargs)
    if k is None:
        k = pos_args.get(1, 5)
    k_val = k
    if dim is None:
        dim = pos_args.get(2, -1)
    dim_val = dim
    # Read largest and sorted from op_kwargs (from traced config)
    largest = op_kwargs.get("largest", True)
    if largest is None:
        largest = True
    sorted_flag = op_kwargs.get("sorted", True)
    if sorted_flag is None:
        sorted_flag = True
    if output_memory_config is None and memory_config is not None:
        output_memory_config = memory_config

    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    torch_values, torch_indices = torch.topk(
        torch_input_tensor_a, k_val, dim=dim_val, largest=largest, sorted=sorted_flag
    )
    torch_output_tensor = torch_values

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

    # Check for indices_tensor named tensor kwarg (pre-allocated output tensor for indices)
    indices_tensor_info = extract_named_tensor_kwargs(kwargs, "indices_tensor")
    if indices_tensor_info is not None and not is_host:
        # Build the indices output shape: same as input but last dim = k
        it_shape = tuple(indices_tensor_info["shape"]) if indices_tensor_info["shape"] else list(shape)
        it_dtype = indices_tensor_info.get("dtype") or input_a_dtype
        it_layout = indices_tensor_info.get("layout") or input_a_layout
        it_mem_cfg = indices_tensor_info.get("memory_config") or input_a_memory_config
        it_dtype = parse_dict_value("indices_tensor_dtype", it_dtype) if isinstance(it_dtype, dict) else it_dtype
        it_layout = parse_dict_value("indices_tensor_layout", it_layout) if isinstance(it_layout, dict) else it_layout
        it_mem_cfg = parse_dict_value("indices_tensor_memory_config", it_mem_cfg) if isinstance(it_mem_cfg, dict) else it_mem_cfg
        it_placement = indices_tensor_info.get("tensor_placement")

        torch_preallocated_indices = torch.zeros(it_shape, dtype=torch.float32)
        if is_mesh_device and it_placement:
            preallocated_indices = create_tensor_on_mesh(
                torch_preallocated_indices, device, it_dtype, it_layout, it_mem_cfg, it_placement,
            )
        else:
            preallocated_indices = ttnn.from_torch(
                torch_preallocated_indices, dtype=it_dtype, layout=it_layout, device=device, memory_config=it_mem_cfg,
            )
        op_kwargs["indices_tensor"] = preallocated_indices

    start_time = start_measuring_time()
    topk_result = ttnn.topk(input_tensor_a, k=k_val, dim=dim_val, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(topk_result[0], device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    return [pcc, e2e_perf]
