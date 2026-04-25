# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
    get_mesh_composer,
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_positional_args, extract_named_tensor_kwargs, parse_dict_value

TIMEOUT = 300

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
    mesh_shape = get_model_traced_mesh_shape()
    device = create_mesh_device(mesh_shape)
    device_name = ttnn.get_arch_name()
    yield (device, device_name)
    ttnn.close_mesh_device(device)


def run(
    input_a_shape=None,
    input_a_dtype=None,
    input_a_layout=None,
    input_a_memory_config=None,
    output_memory_config=None,
    memory_config=None,
    dim=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # V2 vectors provide input_shape (not input_a_*) plus src_* and index_* named tensors
    if input_a_shape is None:
        input_a_shape = kwargs.get("input_shape", (1, 1, 32, 32))
    if input_a_dtype is None:
        input_a_dtype = kwargs.get("input_dtype", ttnn.bfloat16)
    if input_a_layout is None:
        input_a_layout = kwargs.get("input_layout", ttnn.TILE_LAYOUT)
    if input_a_memory_config is None:
        input_a_memory_config = kwargs.get("input_memory_config", ttnn.DRAM_MEMORY_CONFIG)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", kwargs.get("input_tensor_placement", None))
    is_mesh_device = hasattr(device, "get_num_devices")

    # Extract named tensor kwargs for index and src to get correct dtypes
    index_tensor_info = extract_named_tensor_kwargs(kwargs, "index")
    src_tensor_info = extract_named_tensor_kwargs(kwargs, "src")

    op_kwargs = build_op_kwargs(
        kwargs,
        exclude={
            "arg1",
            "index_shape",
            "src_shape",
            "input_shape",
            "input_dtype",
            "input_layout",
            "input_memory_config",
            "input_tensor_placement",
        },
        output_memory_config=output_memory_config,
    )

    pos_args = extract_positional_args(kwargs)
    dim = dim or pos_args.get(1, 0)
    if isinstance(dim, float):
        dim = int(dim)

    if isinstance(input_a_shape, dict):
        shape = input_a_shape.get("self", (1, 1, 32, 32))
        index_shape = input_a_shape.get("index", shape)
        src_shape = input_a_shape.get("src", shape)
    else:
        shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
        index_shape = kwargs.get("index_shape", shape)
        src_shape = kwargs.get("src_shape", shape)

    # Determine dtypes for index and src tensors from vector config
    index_dtype = ttnn.uint16  # default; master trace typically expects uint16
    if index_tensor_info is not None and index_tensor_info.get("dtype") is not None:
        idx_dt = index_tensor_info["dtype"]
        index_dtype = parse_dict_value("index_dtype", idx_dt) if isinstance(idx_dt, dict) else idx_dt
    src_dtype = input_a_dtype  # default
    if src_tensor_info is not None and src_tensor_info.get("dtype") is not None:
        src_dt = src_tensor_info["dtype"]
        src_dtype = parse_dict_value("src_dtype", src_dt) if isinstance(src_dt, dict) else src_dt

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)
    torch_index_tensor = torch.randint(0, shape[dim], index_shape, dtype=torch.int64)
    torch_src_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), src_dtype
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
                index_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
            src_tensor = create_tensor_on_mesh(
                torch_src_tensor,
                device,
                src_dtype,
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
                dtype=index_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
            src_tensor = ttnn.from_torch(
                torch_src_tensor,
                dtype=src_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        input_tensor = ttnn.from_torch(torch_input_tensor, dtype=input_a_dtype, layout=input_a_layout)
        index_tensor = ttnn.from_torch(torch_index_tensor, dtype=index_dtype, layout=input_a_layout)
        src_tensor = ttnn.from_torch(torch_src_tensor, dtype=src_dtype, layout=input_a_layout)

    start_time = start_measuring_time()
    output_tensor = ttnn.scatter(input_tensor, dim=dim, index=index_tensor, src=src_tensor, **op_kwargs)
    mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    return [pcc, e2e_perf]
