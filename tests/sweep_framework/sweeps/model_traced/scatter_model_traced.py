# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader

TIMEOUT = 30

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("scatter")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
        "dim": [3],
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
    memory_config=None,
    dim=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")

    dim = dim or kwargs.get("arg1", 0)
    if output_memory_config is None and memory_config is not None:
        output_memory_config = memory_config

    # Handle scatter's special case with index/src shapes
    if isinstance(input_a_shape, dict):
        shape = input_a_shape.get("self", (1, 1, 32, 32))
        index_shape = input_a_shape.get("index", shape)
        src_shape = input_a_shape.get("src", shape)
    else:
        shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
        index_shape = shape
        src_shape = shape

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)
    torch_index_tensor = torch.randint(0, shape[dim], index_shape, dtype=torch.int64)
    torch_src_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(src_shape)

    torch_output_tensor = torch.scatter(torch_input_tensor, dim, torch_index_tensor, torch_src_tensor)

    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor = create_tensor_on_mesh(
                torch_input_tensor,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
            index_tensor = create_tensor_on_mesh(
                torch_index_tensor,
                device,
                ttnn.int32,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
            src_tensor = create_tensor_on_mesh(
                torch_src_tensor,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            input_tensor = ttnn.from_torch(
                torch_input_tensor,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
            index_tensor = ttnn.from_torch(
                torch_index_tensor,
                dtype=ttnn.int32,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
            src_tensor = ttnn.from_torch(
                torch_src_tensor,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        input_tensor = ttnn.from_torch(torch_input_tensor, dtype=input_a_dtype, layout=input_a_layout)
        index_tensor = ttnn.from_torch(torch_index_tensor, dtype=ttnn.int32, layout=input_a_layout)
        src_tensor = ttnn.from_torch(torch_src_tensor, dtype=input_a_dtype, layout=input_a_layout)

    start_time = start_measuring_time()
    output_tensor = ttnn.scatter(input_tensor, dim, index_tensor, src_tensor)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    return [pcc, e2e_perf]
