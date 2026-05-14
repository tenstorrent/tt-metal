# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import torch

import ttnn
from models.common.utility_functions import torch_random
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    create_mesh_device,
    create_tensor_on_mesh,
    get_mesh_shape,
    mesh_tensor_to_torch,
    reconcile_golden_to_actual,
)
from tests.sweep_framework.sweep_utils.op_kwargs_utils import (
    build_op_kwargs,
    extract_named_tensor_kwargs,
    extract_positional_args,
    parse_dict_value,
)
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time

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
    # Forward slice_dim, num_devices, and output_tensor when master had them.
    absent_keys = set(kwargs.get("__absent_keys__") or [])
    if is_mesh_device:
        if "num_devices" not in absent_keys and "num_devices" not in op_kwargs:
            traced_num_devices = kwargs.get("num_devices")
            if traced_num_devices is not None and traced_num_devices != "__ABSENT__":
                op_kwargs["num_devices"] = int(traced_num_devices)
        if "slice_dim" not in absent_keys and "slice_dim" not in op_kwargs:
            traced_slice_dim = kwargs.get("slice_dim")
            if traced_slice_dim is not None and traced_slice_dim != "__ABSENT__":
                op_kwargs["slice_dim"] = int(traced_slice_dim)
    # Re-add memory_config kwarg when the master config recorded it. build_op_kwargs
    # strips memory_config by default; sweeps that need it must inject it here.
    # Validation-vector runs deliver memory_config as a serialized dict, parse
    # it back to a ttnn.MemoryConfig before forwarding to the op binding.
    if memory_config is not None and "memory_config" not in op_kwargs:
        if isinstance(memory_config, dict):
            memory_config = parse_dict_value("memory_config", memory_config)
        op_kwargs["memory_config"] = memory_config

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

    # Pre-allocate output tensor if the master config recorded one
    output_tensor_info = extract_named_tensor_kwargs(kwargs, "output_tensor")
    if output_tensor_info and output_tensor_info.get("shape"):
        ot_shape = tuple(output_tensor_info["shape"])
        ot_dtype = output_tensor_info.get("dtype") or input_a_dtype
        if isinstance(ot_dtype, dict):
            ot_dtype = parse_dict_value("dtype", ot_dtype) or input_a_dtype
        ot_layout = output_tensor_info.get("layout") or input_a_layout
        if isinstance(ot_layout, dict):
            ot_layout = parse_dict_value("layout", ot_layout) or input_a_layout
        ot_mem_cfg_raw = output_tensor_info.get("memory_config")
        ot_mem_cfg = (
            parse_dict_value("memory_config", ot_mem_cfg_raw)
            if isinstance(ot_mem_cfg_raw, dict)
            else (ot_mem_cfg_raw or input_a_memory_config)
        )
        ot_placement = output_tensor_info.get("tensor_placement")
        torch_out_alloc = torch.zeros(ot_shape, dtype=torch.float32)
        if is_mesh_device and input_a_tensor_placement:
            op_kwargs["output_tensor"] = create_tensor_on_mesh(
                torch_out_alloc, device, ot_dtype, ot_layout, ot_mem_cfg, ot_placement or input_a_tensor_placement
            )
        elif not is_host:
            op_kwargs["output_tensor"] = ttnn.from_torch(
                torch_out_alloc, dtype=ot_dtype, layout=ot_layout, device=device, memory_config=ot_mem_cfg
            )

    # When master used tensor starts/ends (num_devices/slice_dim present),
    # convert list starts/ends to device tensors to match signature 1.
    if "num_devices" in op_kwargs or "slice_dim" in op_kwargs:
        import torch as _torch_s

        pos_args_raw = extract_positional_args(kwargs)
        if isinstance(slice_start, list):
            _start_torch = _torch_s.tensor(slice_start, dtype=_torch_s.int32)
            _start_placement = (
                pos_args_raw.get(1, {}).get("tensor_placement") if isinstance(pos_args_raw.get(1), dict) else None
            )
            if is_mesh_device:
                _sp = _start_placement or {
                    "distribution_shape": "[1, 2]",
                    "mesh_device_shape": "[1, 2]",
                    "placement": "['PlacementReplicate', 'PlacementShard(0)']",
                }
                from tests.sweep_framework.sweep_utils.mesh_tensor_utils import replicate_with_topology

                slice_start = replicate_with_topology(
                    _start_torch,
                    device,
                    ttnn.int32,
                    ttnn.ROW_MAJOR_LAYOUT,
                    ttnn.DRAM_MEMORY_CONFIG,
                    _sp,
                )
            else:
                slice_start = ttnn.from_torch(
                    _start_torch,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
        if isinstance(slice_end, list):
            _end_torch = _torch_s.tensor(slice_end, dtype=_torch_s.int32)
            _end_placement = (
                pos_args_raw.get(2, {}).get("tensor_placement") if isinstance(pos_args_raw.get(2), dict) else None
            )
            if is_mesh_device:
                _ep = _end_placement or {
                    "distribution_shape": "[1, 2]",
                    "mesh_device_shape": "[1, 2]",
                    "placement": "['PlacementReplicate', 'PlacementShard(0)']",
                }
                from tests.sweep_framework.sweep_utils.mesh_tensor_utils import replicate_with_topology

                slice_end = replicate_with_topology(
                    _end_torch,
                    device,
                    ttnn.int32,
                    ttnn.ROW_MAJOR_LAYOUT,
                    ttnn.DRAM_MEMORY_CONFIG,
                    _ep,
                )
            else:
                slice_end = ttnn.from_torch(
                    _end_torch,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

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
