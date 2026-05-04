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
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    reconcile_golden_to_actual,
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("slice")

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


def _slice_input_shard_axis_and_factor(placement_dict):
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
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config=None,
    memory_config=None,
    storage_type="StorageType::DEVICE",
    arg1=None,  # May contain starts from V2 traced configs (positional)
    arg2=None,  # May contain ends from V2 traced configs (positional)
    arg3=None,  # May contain steps from V2 traced configs (positional)
    dtype=None,  # Output dtype from V2 traced configs
    use_legacy=None,  # Legacy mode flag from V2 traced configs
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(
        kwargs,
        exclude={"starts", "ends", "steps", "slice_dim", "num_devices"},
        output_memory_config=output_memory_config,
    )

    if output_memory_config is None and memory_config is not None:
        output_memory_config = memory_config

    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # Track which call style the master vector used so the trace matches.
    starts_kw = kwargs.get("starts")
    ends_kw = kwargs.get("ends")
    steps_kw = kwargs.get("steps")
    use_named_kwargs = starts_kw is not None or ends_kw is not None
    has_explicit_step = (steps_kw is not None) or (arg3 is not None)

    slice_start = starts_kw if starts_kw is not None else (arg1 if arg1 is not None else [0] * len(shape))
    slice_end = ends_kw if ends_kw is not None else arg2
    slice_step = steps_kw if steps_kw is not None else (arg3 if arg3 is not None else [1] * len(shape))

    if not slice_end:
        slice_end = list(shape)
        slice_end[-1] = shape[-1] // 2

    slices = []
    for start, end, step in zip(slice_start, slice_end, slice_step):
        if step == 1:
            slices.append(slice(start, end))
        else:
            slices.append(slice(start, end, step))
    # Trace-validation mode: every chip receives the FULL per-chip input via
    # replicate_with_topology and slices it independently. The gathered output
    # is the per-chip slice tiled along the shard axis — handled by
    # reconcile_golden_to_actual after mesh_tensor_to_torch.
    torch_output_tensor = torch_input_tensor_a[tuple(slices)]

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

    start_time = start_measuring_time()
    if use_named_kwargs:
        if has_explicit_step:
            output_tensor = ttnn.slice(
                input_tensor_a, starts=slice_start, ends=slice_end, steps=slice_step, **op_kwargs
            )
        else:
            output_tensor = ttnn.slice(input_tensor_a, starts=slice_start, ends=slice_end, **op_kwargs)
    else:
        if has_explicit_step:
            output_tensor = ttnn.slice(input_tensor_a, slice_start, slice_end, slice_step, **op_kwargs)
        else:
            output_tensor = ttnn.slice(input_tensor_a, slice_start, slice_end, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(torch_output_tensor, output_tensor, input_a_tensor_placement)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    return [pcc, e2e_perf]
