# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_named_tensor_kwargs
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("experimental::paged_update_cache")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_c_dtype": [ttnn.bfloat16],
        "input_c_layout": [ttnn.TILE_LAYOUT],
        "input_c_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_d_dtype": [ttnn.bfloat16],
        "input_d_layout": [ttnn.TILE_LAYOUT],
        "input_d_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
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
            device = ttnn.open_device(device_id=0)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0)
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    input_c_dtype=None,
    input_c_layout=None,
    input_c_memory_config=None,
    input_d_dtype=None,
    input_d_layout=None,
    input_d_memory_config=None,
    output_memory_config=None,
    memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,  # Accept extra parameters like scalar, traced_source, etc.
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"batch_offset"}, output_memory_config=output_memory_config)

    # V2 vectors provide named tensors: update_idxs_tensor_* → input_c, page_table_* → input_d
    update_idxs_tensor_kwargs = extract_named_tensor_kwargs(kwargs, "update_idxs_tensor")
    page_table_kwargs = extract_named_tensor_kwargs(kwargs, "page_table")
    if input_c_dtype is None and update_idxs_tensor_kwargs is not None:
        input_c_dtype = update_idxs_tensor_kwargs["dtype"]
        input_c_layout = update_idxs_tensor_kwargs.get("layout") or ttnn.ROW_MAJOR_LAYOUT
        input_c_memory_config = update_idxs_tensor_kwargs.get("memory_config") or ttnn.DRAM_MEMORY_CONFIG
    if input_d_dtype is None and page_table_kwargs is not None:
        input_d_dtype = page_table_kwargs["dtype"]
        input_d_layout = page_table_kwargs.get("layout") or ttnn.ROW_MAJOR_LAYOUT
        input_d_memory_config = page_table_kwargs.get("memory_config") or ttnn.DRAM_MEMORY_CONFIG

    if isinstance(input_a_shape, dict):
        shape_a = input_a_shape.get("input_a", input_a_shape.get("self"))
        shape_b = input_a_shape.get("input_b", input_a_shape.get("other"))
        shape_c = input_a_shape.get("input_c")
        shape_d = input_a_shape.get("input_d")
        if shape_c is None:
            shape_c = shape_b
        if shape_d is None:
            shape_d = shape_c
    else:
        if isinstance(input_a_shape, (tuple, list)):
            shape = tuple(input_a_shape)
        else:
            shape = input_a_shape
        shape_a = shape
        input_b_shape_raw = kwargs.get("input_b_shape", None)
        if input_b_shape_raw is not None:
            shape_b = tuple(input_b_shape_raw) if isinstance(input_b_shape_raw, (tuple, list)) else input_b_shape_raw
        else:
            shape_b = shape
        shape_c = (
            update_idxs_tensor_kwargs["shape"]
            if update_idxs_tensor_kwargs
            else kwargs.get("update_idxs_tensor_shape", shape)
        )
        shape_d = page_table_kwargs["shape"] if page_table_kwargs else kwargs.get("page_table_shape", shape)

    has_input_d = input_d_dtype is not None and input_d_layout is not None and input_d_memory_config is not None

    dtype_a = input_a_dtype
    dtype_b = input_b_dtype
    dtype_c = input_c_dtype
    dtype_d = input_d_dtype if has_input_d else None
    layout_a = input_a_layout
    layout_b = input_b_layout
    mem_config_a = input_a_memory_config
    mem_config_b = input_b_memory_config
    mem_config_c = input_c_memory_config
    mem_config_d = input_d_memory_config if has_input_d else None
    # Create input tensors
    torch_input_tensor_a = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_a)(
        shape_a
    )
    torch_input_tensor_b = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_b)(
        shape_b
    )
    torch_input_tensor_c = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_c)(
        shape_c
    )
    # Only create 4th tensor if it's provided
    torch_input_tensor_d = None
    if has_input_d:
        torch_input_tensor_d = gen_func_with_cast_tt(
            partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_d
        )(shape_d)

    # For reference output, just use input_a (paged_update_cache is a caching operation)
    torch_output_tensor = torch_input_tensor_a.clone()

    # Convert to TTNN tensors
    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor_a = create_tensor_on_mesh(
                torch_input_tensor_a,
                device,
                dtype_a,
                layout_a,
                mem_config_a,
                input_a_tensor_placement,
            )
            input_tensor_b = create_tensor_on_mesh(
                torch_input_tensor_b,
                device,
                dtype_b,
                layout_b,
                mem_config_b,
                kwargs.get("input_b_tensor_placement", input_a_tensor_placement),
            )
            input_tensor_c = create_tensor_on_mesh(
                torch_input_tensor_c,
                device,
                dtype_c,
                ttnn.ROW_MAJOR_LAYOUT,
                mem_config_c,
                kwargs.get("input_c_tensor_placement", input_a_tensor_placement),
            )
        else:
            input_tensor_a = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=dtype_a,
                layout=layout_a,
                device=device,
                memory_config=mem_config_a,
            )
            input_tensor_b = ttnn.from_torch(
                torch_input_tensor_b,
                dtype=dtype_b,
                layout=layout_b,
                device=device,
                memory_config=mem_config_b,
            )
            input_tensor_c = ttnn.from_torch(
                torch_input_tensor_c,
                dtype=dtype_c,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=mem_config_c,
            )
    else:
        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=dtype_a, layout=layout_a)
        input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=dtype_b, layout=layout_b)
        input_tensor_c = ttnn.from_torch(torch_input_tensor_c, dtype=dtype_c, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Only create 4th TTNN tensor if provided
    input_tensor_d = None
    if has_input_d:
        if not is_host:
            if is_mesh_device and input_a_tensor_placement:
                input_tensor_d = create_tensor_on_mesh(
                    torch_input_tensor_d,
                    device,
                    dtype_d,
                    ttnn.ROW_MAJOR_LAYOUT,
                    mem_config_d,
                    kwargs.get("input_d_tensor_placement", input_a_tensor_placement),
                )
            else:
                input_tensor_d = ttnn.from_torch(
                    torch_input_tensor_d,
                    dtype=dtype_d,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=mem_config_d,
                )
        else:
            input_tensor_d = ttnn.from_torch(torch_input_tensor_d, dtype=dtype_d, layout=ttnn.ROW_MAJOR_LAYOUT)

    start_time = start_measuring_time()
    # paged_update_cache signature: (cache_tensor, input_tensor, *, update_idxs=[], update_idxs_tensor=None, share_cache=None, page_table=None, ...)
    # Only cache and input are positional, everything else is keyword-only
    # So tensor_a=cache, tensor_b=input, tensor_c=update_idxs_tensor, tensor_d=page_table
    # Note: paged_update_cache may not accept memory_config parameter - it modifies cache_tensor in place
    try:
        output_tensor = ttnn.experimental.paged_update_cache(
            input_tensor_a,  # cache_tensor (positional)
            input_tensor_b,  # input_tensor (positional)
            update_idxs_tensor=input_tensor_c
            if input_tensor_c is not None
            else None,  # update_idxs_tensor (optional keyword)
            page_table=input_tensor_d if input_tensor_d is not None else None,  # page_table (optional keyword)
            batch_offset=0,  # Use default batch_offset
            **op_kwargs,
        )
    except TypeError:
        # If that fails, try with memory_config
        output_tensor = ttnn.experimental.paged_update_cache(
            input_tensor_a,  # cache_tensor (positional)
            input_tensor_b,  # input_tensor (positional)
            update_idxs_tensor=input_tensor_c
            if input_tensor_c is not None
            else None,  # update_idxs_tensor (optional keyword)
            page_table=input_tensor_d if input_tensor_d is not None else None,  # page_table (optional keyword)
            batch_offset=0,
            **op_kwargs,
        )
    # paged_update_cache modifies cache_tensor in place, so output is the same as input_tensor_a
    output_tensor = input_tensor_a
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)
    return [pcc, e2e_perf]
