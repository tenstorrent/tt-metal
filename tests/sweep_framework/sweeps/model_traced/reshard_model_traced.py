# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_model_traced_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    get_mesh_composer,
    dispatch_axis_for_grid,
    shard_grid_bounds,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_positional_args

TIMEOUT = 300

# Device opened per-vector (see _ensure_vector_device): reshard's input AND
# output shard grids straddle both dispatch axes — some need x=7 (ROW), others
# y=9 (COL) — so the per-suite axis used by the default fixture can't place
# every vector's tensors. Configs that couldn't be placed previously failed
# silently (no trace), leaving most master configs "missing". Pick the axis
# each vector's shard grids actually need.
_CUR_DEVICE = None
_CUR_AXIS = "__uninit__"
_CUR_SHAPE = None


def _close_vector_device():
    global _CUR_DEVICE, _CUR_AXIS, _CUR_SHAPE
    if _CUR_DEVICE is not None:
        try:
            ttnn.close_mesh_device(_CUR_DEVICE)
        except Exception:
            # best-effort teardown; a close failure must not mask the test result
            pass
    _CUR_DEVICE = None
    _CUR_AXIS = "__uninit__"
    _CUR_SHAPE = None


def _ensure_vector_device(axis):
    global _CUR_DEVICE, _CUR_AXIS, _CUR_SHAPE
    shape = get_model_traced_mesh_shape()
    if _CUR_DEVICE is None or axis != _CUR_AXIS or shape != _CUR_SHAPE:
        _close_vector_device()
        _CUR_DEVICE = create_mesh_device(shape, dispatch_core_axis=axis)
        _CUR_AXIS = axis
        _CUR_SHAPE = shape
    return _CUR_DEVICE


def _combined_dispatch_axis(*mem_configs):
    """Axis covering the shard grids of all given memory configs.

    reshard places both an input shard tensor and an output shard tensor; a
    single axis must satisfy both grids. Take the max core bounds across all
    configs and pick ROW (x>=7) over COL (y>=9) when both are demanded.
    """
    max_x = max_y = -1
    for mc in mem_configs:
        if mc is None or mc == "__ABSENT__":
            continue
        bx, by = shard_grid_bounds(mc)
        if bx is not None:
            max_x = max(max_x, bx)
        if by is not None:
            max_y = max(max_y, by)
    return dispatch_axis_for_grid(max_x if max_x >= 0 else None, max_y if max_y >= 0 else None)


# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("reshard")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.L1_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> tuple:
    """
    Invalidate test vectors with non-tile-aligned shard shapes.
    ttnn.reshard cannot work with TILE-layout tensors that have non-tile-aligned
    shard dimensions in either input or output.

    The tile-alignment constraint only applies to TILE layout. ROW_MAJOR tensors
    legitimately use sub-tile shard heights (e.g. a [1, 384] height shard) — the
    real models trace exactly these, so rejecting them here drops valid master
    configs (they'd show up as "missing" in validation).
    """
    # Cross-arch grids: a config traced on a wider chip (e.g. Blackhole's ~11x10)
    # can carry a shard grid whose cores (x>7 or y>7) don't physically exist on a
    # Wormhole chip (8x8) -> "Expected number of shards N <= total L1 banks 64".
    # Such a config can't run on Wormhole at all (it belongs on its own arch), so
    # mark it unreconstructable here. Gate to Wormhole so Blackhole — where these
    # cores DO exist — keeps them valid.
    import os as _os

    if "wormhole" in _os.environ.get("ARCH_NAME", "").lower():
        for _ck in ("input_a_memory_config", "output_memory_config"):
            _mc = test_vector.get(_ck)
            _ss = _mc.get("data", {}).get("shard_spec") if isinstance(_mc, dict) else None
            for _r in (_ss or {}).get("grid", []) if isinstance(_ss, dict) else []:
                _end = _r.get("end", {}) if isinstance(_r, dict) else {}
                if int(_end.get("x", 0)) > 7 or int(_end.get("y", 0)) > 7:
                    return (
                        True,
                        f"{_ck} shard grid uses cores beyond Wormhole's 8x8 (x/y>7); cross-arch (e.g. Blackhole) config",
                    )

    layout = test_vector.get("input_a_layout")
    if layout is not None and "ROW_MAJOR" in str(layout):
        return False, None

    # Check both input and output memory configs for non-tile-aligned shards
    for config_key in ["input_a_memory_config", "output_memory_config"]:
        mem_config = test_vector.get(config_key)

        if mem_config:
            # Check if it's a dict (during generation) or object (during execution)
            if isinstance(mem_config, dict):
                shard_spec = mem_config.get("data", {}).get("shard_spec")
                if shard_spec and "shape" in shard_spec:
                    shard_shape = shard_spec["shape"]
                    if len(shard_shape) >= 2:
                        height, width = shard_shape[-2], shard_shape[-1]
                        if height % 32 != 0 or width % 32 != 0:
                            return (
                                True,
                                f"{config_key} shard shape {shard_shape} not tile-aligned (must be divisible by 32)",
                            )
            elif hasattr(mem_config, "shard_spec") and mem_config.shard_spec:
                shard_spec = mem_config.shard_spec
                if hasattr(shard_spec, "shape"):
                    shard_shape = shard_spec.shape
                    if len(shard_shape) >= 2:
                        height, width = shard_shape[-2], shard_shape[-1]
                        if height % 32 != 0 or width % 32 != 0:
                            return True, f"{config_key} shard shape ({height}, {width}) not tile-aligned"

    return False, None


def mesh_device_fixture():
    # Device opened per-vector in run() (see _ensure_vector_device).
    yield (None, "wormhole_b0")
    _close_vector_device()


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Open this vector's device with a dispatch axis that can place BOTH the
    # input shard grid and the target (output) shard grid. Read the raw config
    # forms (dict or parsed) before any parsing — shard_grid_bounds handles
    # both. The previous fixture used a single per-suite axis and silently
    # failed to place the configs needing the other axis (left untraced).
    pos_args = extract_positional_args(kwargs)
    _out_candidates = [
        output_memory_config,
        kwargs.get("memory_config"),
        kwargs.get("input_b_memory_config"),
        pos_args.get(1),
    ]
    _ax = _combined_dispatch_axis(input_a_memory_config, *_out_candidates)
    device = _ensure_vector_device(_ax)

    # Parse input_a_memory_config if it arrived as a dict (from vector data) —
    # passing a raw dict to from_torch/create_tensor_on_mesh would raise and
    # leave the config untraced.
    if isinstance(input_a_memory_config, dict):
        from tests.sweep_framework.master_config_loader_v2 import dict_to_memory_config

        input_a_memory_config = dict_to_memory_config(input_a_memory_config)

    # Extract kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    shape = input_a_shape if isinstance(input_a_shape, (tuple, list)) else (1, 1, 32, 32)

    # Tensor creation
    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        shape
    )
    torch_output = torch_input.clone()

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            # Use mesh with placement
            input_tensor = create_tensor_on_mesh(
                torch_input,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            # Regular single-device tensor
            input_tensor = ttnn.from_torch(
                torch_input,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        # Host storage
        input_tensor = ttnn.from_torch(torch_input, dtype=input_a_dtype, layout=input_a_layout)

    # Op call — reshard's TARGET memory config is the 2nd positional arg (arg1),
    # NOT output_memory_config. The vector generator mis-populates
    # output_memory_config with arg0's (input) config; the real target lives in
    # arg1 / input_b_memory_config (the master's recorded arg1). Resolve from
    # arg1 first so the recorded target shard_spec matches the master.
    from tests.sweep_framework.master_config_loader_v2 import dict_to_memory_config

    op_kwargs.pop("memory_config", None)
    op_kwargs.pop("output_memory_config", None)

    def _as_mem_config(v):
        if v is None or v == "__ABSENT__":
            return None
        if isinstance(v, dict):
            from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

            # arg1 may be a bare memory_config dict or a {memory_config: ...} wrapper.
            return (
                dict_to_memory_config(v) if "shard_spec" in v or "data" in v else parse_dict_value("memory_config", v)
            )
        return v  # already a parsed MemoryConfig

    reshard_mem_config = (
        _as_mem_config(pos_args.get(1))
        or _as_mem_config(kwargs.get("input_b_memory_config"))
        or _as_mem_config(output_memory_config)
        or _as_mem_config(kwargs.get("memory_config"))
    )
    if reshard_mem_config is None:
        return [(False, "Missing target memory_config for reshard"), 0.0]

    # ROW_MAJOR reshard call sites in the traced models pass a preallocated
    # output tensor as a 3rd positional arg (master arg2, same config as arg1);
    # TILE call sites do not. Reproduce that form so the recorded arg list
    # matches the master (else arg2 shows as a missing extra_key).
    _has_output_tensor = input_a_layout is not None and "ROW_MAJOR" in str(input_a_layout)
    preallocated_output = None
    if _has_output_tensor:
        input_b_placement = kwargs.get("input_b_tensor_placement", input_a_tensor_placement)
        torch_out = torch.zeros_like(torch_input)
        if is_mesh_device and input_b_placement:
            preallocated_output = create_tensor_on_mesh(
                torch_out, device, input_a_dtype, input_a_layout, reshard_mem_config, input_b_placement
            )
        else:
            preallocated_output = ttnn.from_torch(
                torch_out,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=reshard_mem_config,
            )

    start_time = start_measuring_time()
    if preallocated_output is not None:
        output_tensor = ttnn.reshard(input_tensor, reshard_mem_config, preallocated_output, **op_kwargs)
    else:
        output_tensor = ttnn.reshard(input_tensor, reshard_mem_config, **op_kwargs)
    mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    # Comparison
    return [check_with_pcc(torch_output, output_tensor, 0.999), e2e_perf]
