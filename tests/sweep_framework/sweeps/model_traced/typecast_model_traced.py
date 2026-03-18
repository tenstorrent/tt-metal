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
    get_mesh_composer,
    infer_mesh_shape_from_params,
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader, parse_dtype
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("typecast")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_dtype": [ttnn.float32],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    mesh_shape = get_mesh_shape()
    if not mesh_shape:
        mesh_shape = infer_mesh_shape_from_params(model_traced_params)
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
    output_dtype=None,
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
    op_kwargs = build_op_kwargs(kwargs, exclude={"dtype", "arg1"}, output_memory_config=output_memory_config)

    output_dtype = output_dtype or kwargs.get("dtype", kwargs.get("arg1", ttnn.float32))
    if isinstance(output_dtype, dict):
        output_dtype = parse_dtype(output_dtype.get("repr", ""))
    elif isinstance(output_dtype, str):
        output_dtype = parse_dtype(output_dtype)
    if output_dtype is None:
        output_dtype = ttnn.float32
    if output_memory_config is None and memory_config is not None:
        output_memory_config = memory_config

    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    if input_a_dtype == ttnn.uint16:
        torch_input_tensor_a = torch.randint(0, 65536, shape, dtype=torch.int32).clamp(0, 65535)
    elif input_a_dtype == ttnn.uint32:
        torch_input_tensor_a = torch.randint(0, 2**32, shape, dtype=torch.int64)
    else:
        torch_input_tensor_a = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
        )(shape)

    if output_dtype == ttnn.float32:
        torch_output_tensor = torch_input_tensor_a.to(torch.float32)
    elif output_dtype == ttnn.bfloat16:
        torch_output_tensor = torch_input_tensor_a.to(torch.bfloat16).to(torch.float32)
    elif output_dtype == ttnn.bfloat8_b:
        torch_output_tensor = torch_input_tensor_a.to(torch.float32)
    elif output_dtype == ttnn.uint16:
        torch_output_tensor = torch_input_tensor_a.clamp(0, 65535).to(torch.int32)
    elif output_dtype == ttnn.uint32:
        if input_a_dtype == ttnn.uint32:
            torch_output_tensor = torch_input_tensor_a.clamp(0, 2**32 - 1)
        else:
            torch_output_tensor = torch_input_tensor_a.clamp(0, 2**32 - 1).to(torch.int64)
    elif output_dtype == ttnn.int32:
        torch_output_tensor = torch_input_tensor_a.to(torch.int32)
    else:
        torch_output_tensor = torch_input_tensor_a.to(torch.float32)

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

    # Create mesh composer for sharded tensors to properly reassemble output
    composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None

    start_time = start_measuring_time()
    output_tensor = ttnn.typecast(input_tensor_a, dtype=output_dtype, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=composer)
    e2e_perf = stop_measuring_time(start_time)

    if output_dtype == ttnn.uint32 or input_a_dtype == ttnn.uint32:
        if torch_output_tensor.dtype != torch.int64:
            torch_output_tensor_f32 = torch_output_tensor.to(torch.int64).to(torch.float32)
        else:
            torch_output_tensor_f32 = torch_output_tensor.to(torch.float32)
        if output_tensor.dtype != torch.int64:
            output_tensor_f32 = output_tensor.to(torch.int64).to(torch.float32)
        else:
            output_tensor_f32 = output_tensor.to(torch.float32)
    else:
        torch_output_tensor_f32 = torch_output_tensor.to(torch.float32)
        output_tensor_f32 = output_tensor.to(torch.float32)

    pcc = check_with_pcc(torch_output_tensor_f32, output_tensor_f32, 0.999)
    return [pcc, e2e_perf]
