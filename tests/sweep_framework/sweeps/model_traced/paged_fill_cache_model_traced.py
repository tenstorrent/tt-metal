# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import re

import torch
import ttnn
from ttnn import ShardTensor2dMesh

from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_named_tensor_kwargs
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    infer_mesh_shape_from_params,
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
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def _parse_mesh_shape(mesh_device_shape):
    """Parse mesh_device_shape which may be a list, tuple, or string like '[4, 8]'."""
    if isinstance(mesh_device_shape, (list, tuple)):
        return tuple(int(x) for x in mesh_device_shape)
    if isinstance(mesh_device_shape, str):
        nums = re.findall(r"\d+", mesh_device_shape)
        if len(nums) >= 2:
            return tuple(int(x) for x in nums[:2])
    return None


def _parse_shard_dims_from_placement(tensor_placement):
    """Extract (dim0, dim1) for ShardTensor2dMesh from a traced tensor_placement dict.

    Returns None when the field cannot be parsed or placement is all-Replicate.
    """
    if not tensor_placement:
        return None
    placement = tensor_placement.get("placement", "")
    if isinstance(placement, list):
        placement = " ".join(str(p) for p in placement)
    if "PlacementShard" not in placement:
        return None
    dims = []
    for m in re.finditer(r"PlacementShard\((-?\d+)\)|PlacementReplicate", placement):
        if m.group(1) is not None:
            dims.append(int(m.group(1)))
        else:
            dims.append(None)
    return tuple(dims) if len(dims) == 2 else None


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
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")

    page_table_kwargs = extract_named_tensor_kwargs(kwargs, "page_table")
    if input_c_dtype is None and page_table_kwargs is not None:
        input_c_dtype = page_table_kwargs["dtype"]
        input_c_layout = page_table_kwargs.get("layout") or ttnn.ROW_MAJOR_LAYOUT
        input_c_memory_config = page_table_kwargs.get("memory_config") or ttnn.DRAM_MEMORY_CONFIG

    page_table_tensor_placement = None
    if page_table_kwargs is not None:
        page_table_tensor_placement = page_table_kwargs.get("tensor_placement")
    if page_table_tensor_placement is None:
        page_table_tensor_placement = kwargs.get("input_c_tensor_placement", input_a_tensor_placement)

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

    # Determine mesh shape from tensor placement metadata
    mesh_shape = None
    for tp in (input_a_tensor_placement, input_b_tensor_placement):
        if tp:
            mesh_shape = _parse_mesh_shape(tp.get("mesh_device_shape"))
            if mesh_shape:
                break
    is_2d_mesh = is_mesh_device and mesh_shape is not None and mesh_shape[0] > 1 and mesh_shape[1] > 1

    # Parse shard dims for arg1 (input tensor) from its traced placement.
    # The model shards arg1 with PlacementShard(2), PlacementShard(1) on a 4x8 mesh.
    shard_dims_b = _parse_shard_dims_from_placement(input_b_tensor_placement) if is_2d_mesh else None

    torch_input_tensor_a = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_a)(
        shape_a
    )

    if shard_dims_b is not None:
        # Build a global tensor for arg1 that, when sharded, yields the traced
        # per-device shape.  Each sharded dim is scaled by the mesh size on that axis.
        global_shape_b = list(shape_b)
        for axis_idx, sd in enumerate(shard_dims_b):
            if sd is not None:
                esd = sd if sd >= 0 else len(shape_b) + sd
                global_shape_b[esd] *= mesh_shape[axis_idx]
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_b
        )(global_shape_b)
    else:
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_b
        )(shape_b)

    torch_input_tensor_c = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_c)(
        shape_c
    )

    torch_output_tensor = torch_input_tensor_a.clone()

    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            # arg0 (cache tensor) — always replicated
            input_tensor_a = create_tensor_on_mesh(
                torch_input_tensor_a,
                device,
                dtype_a,
                layout_a,
                mem_config_a,
                input_a_tensor_placement,
            )

            # arg1 (input tensor) — may be sharded on a 2D mesh
            if shard_dims_b is not None:
                input_tensor_b = ttnn.from_torch(
                    torch_input_tensor_b,
                    dtype=dtype_b,
                    layout=layout_b,
                    device=device,
                    memory_config=mem_config_b,
                    mesh_mapper=ShardTensor2dMesh(device, dims=shard_dims_b, mesh_shape=mesh_shape),
                )
            else:
                input_tensor_b = create_tensor_on_mesh(
                    torch_input_tensor_b,
                    device,
                    dtype_b,
                    layout_b,
                    mem_config_b,
                    input_b_tensor_placement or input_a_tensor_placement,
                )

            # page_table tensor — typically replicated
            input_tensor_c = create_tensor_on_mesh(
                torch_input_tensor_c,
                device,
                dtype_c,
                layout_c,
                mem_config_c,
                page_table_tensor_placement,
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

    batch_idx = kwargs.get("batch_idx", 0)
    if batch_idx is None:
        batch_idx = 0

    start_time = start_measuring_time()
    # Pass page_table as a named kwarg (not positional) to match the model trace.
    # The model calls: paged_fill_cache(cache, input, page_table=pt, batch_idx=idx, ...)
    output_tensor = ttnn.experimental.paged_fill_cache(
        input_tensor_a,
        input_tensor_b,
        page_table=input_tensor_c,
        batch_idx=batch_idx,
    )
    output_tensor = input_tensor_a
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)
    return [pcc, e2e_perf]
