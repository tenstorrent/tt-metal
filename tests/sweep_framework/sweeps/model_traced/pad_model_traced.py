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
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("pad")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "padding": [((0, 1), (0, 1), (0, 2), (0, 2))],
        "value": [0.0],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> tuple:
    storage_type = test_vector.get("storage_type")
    if storage_type and "HOST" in str(storage_type):
        return True, "HOST storage operation: CPU-side preprocessing, not a device operation to test"
    return False, None


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
            device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
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
    padding=None,
    value=0.0,
    output_padded_shape=None,
    input_tensor_start=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"arg1", "arg2", "arg3"}, output_memory_config=output_memory_config)

    # v2 tracer captures pad args positionally:
    #   Format A: arg1=padding (nested list [[left,right],...]), value=fill_value
    #   Format B: arg1=output_padded_shape (flat list), arg2=input_tensor_start, arg3=fill_value
    arg1 = kwargs.get("arg1", None)
    arg2 = kwargs.get("arg2", None)
    arg3 = kwargs.get("arg3", None)

    if padding is None and arg1 is not None:
        is_nested = isinstance(arg1, list) and arg1 and isinstance(arg1[0], (list, tuple))
        if is_nested:
            padding = arg1
        else:
            output_padded_shape = arg1
            if arg2 is not None and input_tensor_start is None:
                input_tensor_start = arg2
            if arg3 is not None and value is None:
                value = arg3

    if value is None:
        value = 0.0

    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        shape
    )

    if padding is not None:
        pass
    elif output_padded_shape is not None and input_tensor_start is not None:
        calculated_padding = []
        for i in range(len(shape)):
            start = input_tensor_start[i] if i < len(input_tensor_start) else 0
            end = output_padded_shape[i] - shape[i] - start
            calculated_padding.append([start, max(0, end)])
        padding = calculated_padding
    else:
        padding = [[0, 0]] * len(shape)

    # PyTorch reference
    torch_padding = []
    for i in range(len(padding) - 1, -1, -1):
        for p in padding[i]:
            torch_padding.append(p)
    torch_output = torch.nn.functional.pad(torch_input, torch_padding, mode="constant", value=value)

    if isinstance(padding, list):
        padding = tuple(tuple(p) if isinstance(p, (list, tuple)) else p for p in padding)

    if not is_mesh_device or not input_a_tensor_placement:
        input_tensor = ttnn.from_torch(
            torch_input,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=input_a_memory_config,
        )
    else:
        input_tensor = create_tensor_on_mesh(
            torch_input,
            device,
            input_a_dtype,
            input_a_layout,
            input_a_memory_config,
            input_a_tensor_placement,
        )

    start_time = start_measuring_time()
    if output_memory_config is not None and "memory_config" not in op_kwargs:
        op_kwargs["memory_config"] = output_memory_config
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output, output_tensor, 0.999)
    return [pcc, e2e_perf]
