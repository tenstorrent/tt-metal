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
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    reconcile_golden_to_actual,
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import (
    build_op_kwargs,
    extract_positional_args,
    extract_named_tensor_kwargs,
)

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


def _scatter_input_shard_axis_and_factor(placement_dict):
    if not isinstance(placement_dict, dict):
        return None, 1
    plac_raw = placement_dict.get("placement")
    dist_raw = placement_dict.get("distribution_shape")
    if plac_raw is None or dist_raw is None:
        return None, 1
    if isinstance(plac_raw, (list, tuple)):
        plac_items = [str(x).strip().strip("'") for x in plac_raw]
    else:
        s_inner = str(plac_raw).strip()
        if s_inner.startswith("[") and s_inner.endswith("]"):
            s_inner = s_inner[1:-1]
        plac_items = [x.strip().strip("'") for x in s_inner.split(",") if x.strip()]
    if isinstance(dist_raw, (list, tuple)):
        dist_items = [int(x) for x in dist_raw]
    else:
        d_inner = str(dist_raw).strip()
        if d_inner.startswith("[") and d_inner.endswith("]"):
            d_inner = d_inner[1:-1]
        dist_items = [int(x.strip()) for x in d_inner.split(",") if x.strip()]
    axis = None
    factor = 1
    for entry, n in zip(plac_items, dist_items):
        if entry.startswith("PlacementShard("):
            try:
                d = int(entry[len("PlacementShard(") : -1])
            except ValueError:
                continue
            axis = d
            factor *= n
    return axis, factor


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

    # Extract named tensor kwargs for index and src (V2 traced configs)
    index_info = extract_named_tensor_kwargs(kwargs, "index")
    src_info = extract_named_tensor_kwargs(kwargs, "src")

    if isinstance(input_a_shape, dict):
        shape = input_a_shape.get("self", (1, 1, 32, 32))
        index_shape = input_a_shape.get("index", shape)
        src_shape = input_a_shape.get("src", shape)
    else:
        shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
        index_shape = (index_info["shape"] if index_info and index_info["shape"] else None) or kwargs.get(
            "index_shape", shape
        )
        src_shape = (src_info["shape"] if src_info and src_info["shape"] else None) or kwargs.get("src_shape", shape)

    # Use traced dtypes for index and src tensors (don't hardcode int32)
    index_dtype = (index_info["dtype"] if index_info and index_info.get("dtype") else None) or ttnn.int32
    index_layout = (index_info["layout"] if index_info and index_info.get("layout") else None) or input_a_layout
    index_mem = (
        index_info["memory_config"] if index_info and index_info.get("memory_config") else None
    ) or input_a_memory_config
    index_placement = index_info["tensor_placement"] if index_info else None

    src_dtype = (src_info["dtype"] if src_info and src_info.get("dtype") else None) or input_a_dtype
    src_layout = (src_info["layout"] if src_info and src_info.get("layout") else None) or input_a_layout
    src_mem = (
        src_info["memory_config"] if src_info and src_info.get("memory_config") else None
    ) or input_a_memory_config
    src_placement = src_info["tensor_placement"] if src_info else None

    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # Sharded-aware scatter: when dim coincides with the input's shard axis,
    # the kernel sees per-chip slices of input/index/src. Generate index in
    # [0, per_chip_dim_size), chunk each tensor along its own shard axis
    # (input/index/src all may be Sharded), scatter per-chip, then concat
    # along the input shard axis.
    # In trace-validation mode, create_tensor_on_mesh routes shard placements
    # through replicate_with_topology, so every chip receives the FULL per-chip
    # input/index/src and computes a full scatter. The gathered result is the
    # per-chip scatter tiled along the shard axis — handled by reconcile_golden_to_actual.
    torch_index_tensor = torch.randint(0, shape[dim], index_shape, dtype=torch.int64)
    torch_src_tensor = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), src_dtype)(
        src_shape
    )
    torch_src_tensor_for_golden = torch_src_tensor.to(torch_input_tensor.dtype)
    torch_output_tensor = torch.scatter(torch_input_tensor, dim, torch_index_tensor, torch_src_tensor_for_golden)

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
                index_layout,
                index_mem,
                index_placement or input_a_tensor_placement,
            )
            src_tensor = create_tensor_on_mesh(
                torch_src_tensor,
                device,
                src_dtype,
                src_layout,
                src_mem,
                src_placement or input_a_tensor_placement,
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
                layout=index_layout,
                device=device,
                memory_config=index_mem,
            )
            src_tensor = ttnn.from_torch(
                torch_src_tensor,
                dtype=src_dtype,
                layout=src_layout,
                device=device,
                memory_config=src_mem,
            )
    else:
        input_tensor = ttnn.from_torch(torch_input_tensor, dtype=input_a_dtype, layout=input_a_layout)
        index_tensor = ttnn.from_torch(torch_index_tensor, dtype=index_dtype, layout=index_layout)
        src_tensor = ttnn.from_torch(torch_src_tensor, dtype=src_dtype, layout=src_layout)

    start_time = start_measuring_time()
    output_tensor = ttnn.scatter(input_tensor, dim=dim, index=index_tensor, src=src_tensor, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(torch_output_tensor, output_tensor, input_a_tensor_placement)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    return [pcc, e2e_perf]
