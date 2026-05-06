# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    get_model_traced_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    get_mesh_composer,
    reconcile_golden_to_actual,
)

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("experimental::paged_fill_cache")

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
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
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
    input_b_shape=None,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    input_c_shape=None,
    input_c_dtype=None,
    input_c_layout=None,
    input_c_memory_config=None,
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
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    # V2 vectors provide page_table as a named tensor (page_table_*) instead of input_c_*
    page_table_kwargs = extract_named_tensor_kwargs(kwargs, "page_table")
    if input_c_dtype is None and page_table_kwargs is not None:
        input_c_dtype = page_table_kwargs["dtype"]
        input_c_layout = page_table_kwargs.get("layout") or ttnn.ROW_MAJOR_LAYOUT
        input_c_memory_config = page_table_kwargs.get("memory_config") or ttnn.DRAM_MEMORY_CONFIG

    if isinstance(input_a_shape, dict):
        shape_a = input_a_shape.get("input_a", input_a_shape.get("self"))
        shape_b = input_a_shape.get("input_b", input_a_shape.get("other"))
        shape_c = input_a_shape.get("input_c")
        if shape_c is None:
            shape_c = shape_b
    else:
        if isinstance(input_a_shape, (tuple, list)):
            shape = tuple(input_a_shape)
        else:
            shape = input_a_shape
        shape_a = shape
        shape_b = tuple(input_b_shape) if input_b_shape is not None else shape
        # Use input_c_shape (3rd positional tensor) for page table, falling back to page_table_shape
        if input_c_shape is not None:
            shape_c = tuple(input_c_shape)
        elif page_table_kwargs and page_table_kwargs.get("shape") is not None:
            shape_c = page_table_kwargs["shape"]
        else:
            pt_shape = kwargs.get("page_table_shape")
            shape_c = tuple(pt_shape) if pt_shape is not None else shape

    dtype_a = input_a_dtype
    dtype_b = input_b_dtype
    dtype_c = input_c_dtype
    layout_a = input_a_layout
    layout_b = input_b_layout
    layout_c = input_c_layout
    mem_config_a = input_a_memory_config
    mem_config_b = input_b_memory_config
    mem_config_c = input_c_memory_config
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

    # Real paged_fill_cache golden: write input_b's seq_len chunks into the
    # cache pages indexed by page_table[batch_idx]. cache layout is
    # [num_pages, num_heads, page_size, head_dim]; input layout is
    # [batch, num_heads, seq_len, head_dim]. mesh_tensor_to_torch reassembles
    # the sharded cache back to global shape, so a global torch reference
    # matches the actual reassembled output.
    torch_output_tensor = torch_input_tensor_a.clone()
    if len(shape_a) == 4 and len(shape_b) == 4 and torch_input_tensor_c.numel() > 0:
        _page_size = shape_a[2]
        _seq_len = shape_b[2]
        _bidx = int(op_kwargs.get("batch_idx", 0))
        _page_idx_row = torch_input_tensor_c.to(torch.int64).reshape(torch_input_tensor_c.shape[0], -1)
        if 0 <= _bidx < _page_idx_row.shape[0]:
            _pages = _page_idx_row[_bidx].tolist()
            for _chunk_idx, _page in enumerate(_pages):
                _start = _chunk_idx * _page_size
                _end = _start + _page_size
                if _end > _seq_len or not (0 <= _page < shape_a[0]):
                    break
                torch_output_tensor[_page, :, :_page_size, :] = torch_input_tensor_b[_bidx, :, _start:_end, :]

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
                layout_c,
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
                layout=layout_c,
                device=device,
                memory_config=mem_config_c,
            )
    else:
        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=dtype_a, layout=layout_a)
        input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=dtype_b, layout=layout_b)
        input_tensor_c = ttnn.from_torch(torch_input_tensor_c, dtype=dtype_c, layout=layout_c)

    # Ensure batch_idx has a default value in op_kwargs
    if "batch_idx" not in op_kwargs or op_kwargs["batch_idx"] is None:
        op_kwargs["batch_idx"] = 0

    start_time = start_measuring_time()
    try:
        output_tensor = ttnn.experimental.paged_fill_cache(
            input_tensor_a,  # cache_tensor
            input_tensor_b,  # input_tensor
            input_tensor_c,  # page_table
            **op_kwargs,
        )
    except TypeError:
        output_tensor = ttnn.experimental.paged_fill_cache(
            input_tensor_a,  # cache_tensor
            input_tensor_b,  # input_tensor
            input_tensor_c,  # page_table
            **op_kwargs,
        )
    # paged_fill_cache modifies cache_tensor in place, so output is the same as input_tensor_a
    output_tensor = input_tensor_a
    mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(
            torch_output_tensor, output_tensor, input_a_tensor_placement, kwargs.get("input_b_tensor_placement", None)
        )
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)
    return [pcc, e2e_perf]
