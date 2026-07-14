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
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_positional_args, parse_dict_value

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("transpose")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 32, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "dim0": [0],
        "dim1": [1],
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


def _reorder_l1_mc_for_dram_sharded(mc, device):
    """Reorder an L1-sharded MemoryConfig's core_ranges to match the device's
    optimal DRAM bank → worker assignment so the recorded grid matches what
    master traced (master records cores in the optimal order)."""
    try:
        if mc is None or mc.buffer_type != ttnn.BufferType.L1:
            return mc
        if mc.shard_spec is None:
            return mc
        old_grid = mc.shard_spec.grid
        master_cores = set()
        for cr in old_grid.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    master_cores.add((x, y))
        if not master_cores:
            return mc
        try:
            optimal = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        except Exception:
            return mc
        ordered = [(c.x, c.y) for c in optimal if (c.x, c.y) in master_cores]
        if len(ordered) != len(master_cores):
            return mc
        new_ranges = [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in ordered]
        new_grid = ttnn.CoreRangeSet(new_ranges)
        new_shard_spec = ttnn.ShardSpec(new_grid, mc.shard_spec.shape, mc.shard_spec.orientation)
        return ttnn.MemoryConfig(mc.memory_layout, mc.buffer_type, new_shard_spec)
    except Exception:
        return mc


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    dim0=None,
    dim1=None,
    output_memory_config=None,
    memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    if input_a_tensor_placement is None:
        input_a_tensor_placement = kwargs.get("input_tensor_a_tensor_placement") or kwargs.get(
            "input_tensor_tensor_placement"
        )
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"arg1", "arg2"}, output_memory_config=output_memory_config)
    # Re-add memory_config kwarg when the master recorded it (build_op_kwargs strips it by default).
    # Parse dict form to ttnn.MemoryConfig so the trace records master's exact
    # shard_spec rather than the kernel's auto-derived one.
    if memory_config is not None and "memory_config" not in op_kwargs:
        if isinstance(memory_config, dict):
            from tests.sweep_framework.master_config_loader_v2 import dict_to_memory_config

            parsed_mc = dict_to_memory_config(memory_config)
        else:
            parsed_mc = memory_config
        if parsed_mc is not None:
            op_kwargs["memory_config"] = parsed_mc

    # Parse input_a_memory_config dict → ttnn.MemoryConfig.
    if isinstance(input_a_memory_config, dict):
        from tests.sweep_framework.master_config_loader_v2 import dict_to_memory_config

        input_a_memory_config = dict_to_memory_config(input_a_memory_config)

    pos_args = extract_positional_args(kwargs)
    if dim0 is None:
        dim0 = pos_args.get(1, 0)
    if dim1 is None:
        dim1 = pos_args.get(2, 1)
    if output_memory_config is None and memory_config is not None:
        output_memory_config = memory_config

    # Do NOT inject memory_config from output_memory_config — the master trace only
    # records it when the model explicitly passed it.  Injecting causes extra_key diffs.

    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    torch_output_tensor = torch.transpose(torch_input_tensor_a, dim0, dim1)

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

    def _run_transpose(tensor_a, kw):
        # PRE-OP debug
        import sys as _sys_p

        _ch_p = kwargs.get("config_hash", "")[:8]
        if _ch_p in ("f17dedb1", "76292c81"):
            print(f"[T_PRE] {_ch_p} dim0={dim0} dim1={dim1}", file=_sys_p.stderr, flush=True)
            print(
                f"[T_PRE] {_ch_p} a.shape={tensor_a.shape} a.padded={tensor_a.padded_shape} a.dtype={tensor_a.dtype} a.layout={tensor_a.layout}",
                file=_sys_p.stderr,
                flush=True,
            )
            print(f"[T_PRE] {_ch_p} a.mc={tensor_a.memory_config()}", file=_sys_p.stderr, flush=True)
            print(
                f"[T_PRE] {_ch_p} a.topo dist={tensor_a.tensor_topology().distribution_shape()} plac={tensor_a.tensor_topology().placements()}",
                file=_sys_p.stderr,
                flush=True,
            )
            for _k, _v in kw.items():
                print(f"[T_PRE] {_ch_p} kw[{_k}]={repr(_v)[:300]}", file=_sys_p.stderr, flush=True)
        # Reorder L1-sharded mcs to the device's optimal DRAM-bank order so
        # the trace records the same grid order master saw.
        try:
            if "memory_config" in kw and kw["memory_config"] is not None:
                kw["memory_config"] = _reorder_l1_mc_for_dram_sharded(kw["memory_config"], device)
            if hasattr(tensor_a, "memory_config"):
                _t_mc = tensor_a.memory_config()
                _t_mc2 = _reorder_l1_mc_for_dram_sharded(_t_mc, device)
                if _t_mc2 is not _t_mc:
                    tensor_a = ttnn.to_memory_config(tensor_a, _t_mc2)
        except Exception:
            # Best-effort: tolerate this failure so the sweep can continue.
            pass
        out = ttnn.transpose(tensor_a, dim0, dim1, **kw)
        return mesh_tensor_to_torch(out, device if is_mesh_device else None)

    start_time = start_measuring_time()
    _last_exc = None
    try:
        output_tensor = _run_transpose(input_tensor_a, op_kwargs)
    except Exception as e:
        output_tensor = None
        _last_exc = e

    if (
        output_tensor is not None
        and not is_mesh_device
        and list(output_tensor.shape) != list(torch_output_tensor.shape)
    ):
        output_tensor = None

    if output_tensor is None:
        fallback_kwargs = {k: v for k, v in op_kwargs.items() if k != "memory_config"}
        # NOTE: do NOT rebuild input_tensor_a — when the original was created via
        # create_tensor_on_mesh with sharded topology, plain from_torch here would
        # produce a second trace entry with [Replicate]-only placement, which the
        # validator joins to instead of the correct first-call entry. Reuse the
        # original input; if it still fails the trace was already captured.
        try:
            output_tensor = _run_transpose(input_tensor_a, fallback_kwargs)
        except Exception as e:
            output_tensor = None
            _last_exc = e

    e2e_perf = stop_measuring_time(start_time)

    if output_tensor is None:
        msg = f"transpose execution failed (trace captured): {type(_last_exc).__name__}: {str(_last_exc)[:200]}"
        return [(False, msg), e2e_perf]
    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(torch_output_tensor, output_tensor, input_a_tensor_placement)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)
    return [pcc, e2e_perf]
