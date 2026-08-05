# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
from functools import partial

import torch

import ttnn
from models.common.utility_functions import torch_random

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    broadcast_torch_inputs_to_global,
    create_mesh_device,
    create_tensor_on_mesh,
    dispatch_axis_for_grid,
    get_model_traced_mesh_shape,
    mesh_tensor_to_torch,
    reconcile_golden_to_actual,
    shard_grid_bounds,
    was_replicated_for_validation,
)
from tests.sweep_framework.sweep_utils.op_kwargs_utils import (
    build_op_kwargs,
    extract_named_tensor_kwargs,
    parse_dict_value,
)
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("add")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_shape": [(1, 1, 32, 32)],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


# Device opened per-vector (see _ensure_vector_device) so each vector gets the
# dispatch axis its traced shard grids need. A single per-suite COL (or ROW)
# axis can't serve both: most add configs run on COL, but a few have inputs /
# outputs WIDTH_SHARDED across a full x=0..7 row, whose reshard lands a worker on
# a dispatch core under COL (TT_FATAL "not on_dispatch_core") and silently
# degrades to DRAM-interleaved — a memory_config mismatch vs the master trace.
_CUR_DEVICE = None
_CUR_AXIS = "__uninit__"
_CUR_SHAPE = None


def _close_vector_device():
    global _CUR_DEVICE, _CUR_AXIS, _CUR_SHAPE
    if _CUR_DEVICE is not None:
        try:
            ttnn.close_mesh_device(_CUR_DEVICE)
        except Exception:
            # best-effort teardown — a failed close must not mask the real result
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


def _add_dispatch_axis(*memory_configs):
    """Dispatch axis this vector's sharded tensors need. A shard grid spanning a
    full x=0..7 row needs ROW dispatch (else the reshard worker lands on a
    dispatch core); otherwise default (COL). Derived from the union of the
    input/output shard grids — add has no program_config to consult."""
    mx = my = None
    for mc in memory_configs:
        gx, gy = shard_grid_bounds(mc)
        if gx is not None:
            mx = gx if mx is None else max(mx, gx)
        if gy is not None:
            my = gy if my is None else max(my, gy)
    if mx is not None or my is not None:
        return dispatch_axis_for_grid(mx, my)
    return None


def mesh_device_fixture():
    # Device is opened per-vector in run() (see _ensure_vector_device).
    yield (None, "wormhole_b0")
    _close_vector_device()


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_shape=None,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    arg1=None,  # May contain scalar value from V2 traced configs
    memory_config=None,  # Alternative memory_config parameter from V2 traced configs
    dtype=None,  # Output dtype from V2 traced configs
    *,
    device,
    **kwargs,  # Accept placements, traced_source, traced_machine_info, etc.
) -> list:
    torch.manual_seed(0)

    # Extract kwargs - arg1 is now a named param, use it as scalar fallback
    scalar = kwargs.get("scalar", arg1)
    # The scalar may be a float (e.g. masking value -FLT_MAX ≈ -3.4e38) that the
    # tracer serialized as a huge Python int; passing that to ttnn.add tries to
    # cast to unsigned ("can't convert negative int to unsigned"). Coerce numeric
    # (non-bool) int scalars to float so the op gets a real float scalar.
    if scalar is not None and not isinstance(scalar, bool) and isinstance(scalar, int):
        scalar = float(scalar)
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)

    # Reproduce THIS vector's traced mesh shape so a single main-process
    # invocation that loads all per-mesh files still opens each vector on its
    # own [4,8]/[8,4]/[1,32]/[1,1] mesh (else get_model_traced_mesh_shape()
    # auto-detects one shape for all → mesh_device_shape placement mismatches).
    # Some configs (e.g. tensor-scalar adds) carry no sharded placement, so fall
    # back to the per-vector traced_machine_info, which every vector records.
    import re as _re_mesh

    _ti = kwargs.get("traced_machine_info") or {}
    _mesh_sources = (
        input_a_tensor_placement.get("mesh_device_shape") if isinstance(input_a_tensor_placement, dict) else None,
        input_b_tensor_placement.get("mesh_device_shape") if isinstance(input_b_tensor_placement, dict) else None,
        _ti.get("mesh_device_shape") if isinstance(_ti, dict) else None,
    )
    for _ms in _mesh_sources:
        if _ms:
            _dims = _re_mesh.findall(r"-?\d+", str(_ms))
            if len(_dims) == 2:
                os.environ["MESH_DEVICE_SHAPE"] = f"{int(_dims[0])}x{int(_dims[1])}"
                break

    # Parse memory_config dicts from validation vectors into ttnn.MemoryConfig
    # objects. Without this, sharded mem-configs (WIDTH_SHARDED + L1) silently
    # degrade to DRAM_INTERLEAVED in create_tensor_on_mesh / from_torch, which
    # causes a hash drift vs the master trace.
    if isinstance(input_a_memory_config, dict):
        input_a_memory_config = parse_dict_value("input_a_memory_config", input_a_memory_config)
    if isinstance(input_b_memory_config, dict):
        input_b_memory_config = parse_dict_value("input_b_memory_config", input_b_memory_config)
    if isinstance(output_memory_config, dict):
        output_memory_config = parse_dict_value("output_memory_config", output_memory_config)

    # Open this vector's device with the dispatch axis its sharded tensors need
    # (the fixture yielded None; we own the device here). Consider every sharded
    # input/output grid, incl. the pre-allocated output_tensor.
    _ot_info_for_axis = extract_named_tensor_kwargs(kwargs, "output_tensor") or {}
    device = _ensure_vector_device(
        _add_dispatch_axis(
            input_a_memory_config,
            input_b_memory_config,
            output_memory_config,
            _ot_info_for_axis.get("memory_config"),
        )
    )

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")  # MeshDevice has this method
    op_kwargs = build_op_kwargs(kwargs, exclude={"scalar"}, output_memory_config=output_memory_config, device=device)

    # Re-add memory_config and dtype to op_kwargs when present in master config.
    # build_op_kwargs strips memory_config by default, but the model trace may have
    # passed it explicitly. Same for dtype (output dtype).
    # Use __absent_keys__ to distinguish "master had kwarg=None" from "master never had kwarg".
    # Only pass None when absent_keys is populated (V2 loader provided info) and the key
    # is NOT in absent_keys (meaning the master trace explicitly had this kwarg).
    absent_keys = kwargs.get("__absent_keys__")
    has_absent_info = absent_keys is not None
    absent_keys = set(absent_keys or [])
    if "memory_config" not in absent_keys:
        if memory_config is not None and memory_config != "__ABSENT__":
            parsed_mc = (
                parse_dict_value("memory_config", memory_config) if isinstance(memory_config, dict) else memory_config
            )
            if parsed_mc is not None:
                op_kwargs["memory_config"] = parsed_mc
            elif has_absent_info:
                op_kwargs["memory_config"] = None
        elif memory_config is None and has_absent_info:
            op_kwargs["memory_config"] = None
    if "dtype" not in absent_keys:
        if dtype is not None and dtype != "__ABSENT__":
            parsed_dt = parse_dict_value("dtype", dtype) if isinstance(dtype, dict) else dtype
            if parsed_dt is not None:
                op_kwargs["dtype"] = parsed_dt
            elif has_absent_info:
                op_kwargs["dtype"] = None
        elif dtype is None and has_absent_info:
            op_kwargs["dtype"] = None

    # V2 format provides separate shapes for each input
    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
    shape_b = tuple(input_b_shape) if input_b_shape and isinstance(input_b_shape, (list, tuple)) else input_b_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    # Check if this is a scalar add operation (shape_b is None or scalar is provided)
    if shape_b is None or scalar is not None:
        # Tensor-scalar add: use the scalar value directly
        # If scalar is None but shape_b is None, default to scalar=1.0
        scalar_value = scalar if scalar is not None else 1.0
        torch_output_tensor = torch.add(torch_input_tensor_a, scalar_value)
        is_scalar_add = True
    else:
        # Tensor-tensor add: generate second tensor
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(shape_b)
        ref_a, ref_b = broadcast_torch_inputs_to_global(
            torch_input_tensor_a,
            input_a_tensor_placement,
            torch_input_tensor_b,
            input_b_tensor_placement,
        )
        torch_output_tensor = torch.add(ref_a, ref_b)
        is_scalar_add = False

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Create first tensor (with mesh support if device is mesh)
    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            # Use mesh with placement
            input_tensor_a = create_tensor_on_mesh(
                torch_input_tensor_a,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            # Regular single-device tensor
            input_tensor_a = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        # Host storage
        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=input_a_layout)

    # Pre-allocate output tensor if the master config recorded one (output_tensor kwarg).
    # The tracer records this when the model passes a pre-allocated output tensor to the op.
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
        if is_mesh_device and ot_placement:
            op_kwargs["output_tensor"] = create_tensor_on_mesh(
                torch_out_alloc, device, ot_dtype, ot_layout, ot_mem_cfg, ot_placement
            )
        elif not is_host:
            op_kwargs["output_tensor"] = ttnn.from_torch(
                torch_out_alloc, dtype=ot_dtype, layout=ot_layout, device=device, memory_config=ot_mem_cfg
            )

    start_time = start_measuring_time()

    if is_scalar_add:
        # Tensor-scalar add: pass scalar directly
        scalar_value = scalar if scalar is not None else 1.0
        output_tensor = ttnn.add(input_tensor_a, scalar_value, **op_kwargs)
    else:
        # Tensor-tensor add: convert second tensor and add
        if not is_host:
            if is_mesh_device and input_b_tensor_placement:
                # Use mesh with placement for second tensor
                input_tensor_b = create_tensor_on_mesh(
                    torch_input_tensor_b,
                    device,
                    input_b_dtype,
                    input_b_layout,
                    input_b_memory_config,
                    input_b_tensor_placement,
                )
            else:
                # Regular single-device tensor
                input_tensor_b = ttnn.from_torch(
                    torch_input_tensor_b,
                    dtype=input_b_dtype,
                    layout=input_b_layout,
                    device=device,
                    memory_config=input_b_memory_config,
                )
        else:
            # Host storage
            input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=input_b_dtype, layout=input_b_layout)

        output_tensor = ttnn.add(input_tensor_a, input_tensor_b, **op_kwargs)

    # On <=8-chip meshes create_tensor_on_mesh stores a shard-placement tensor as
    # a full per-chip replica, so each chip computed the full output. Read a single
    # device's copy instead of relying on the byte-identity auto-detect in
    # mesh_tensor_to_torch, which can miss (e.g. INT32 dim-0-sharded N300 configs
    # fell through to a concat -> doubled rows -> PCC ~0.49). Mirrors chunked_sdpa.
    _replicated = is_mesh_device and was_replicated_for_validation(device, input_a_tensor_placement)
    output_tensor = mesh_tensor_to_torch(
        output_tensor, device if is_mesh_device else None, force_single_device=_replicated
    )
    e2e_perf = stop_measuring_time(start_time)

    # V2 traced configs store per-chip input shapes; when both inputs share that
    # shape broadcast_torch_inputs_to_global is a no-op, so torch_output_tensor
    # stays per-chip while mesh_tensor_to_torch returns the gathered global.
    # Tile golden up to match using whichever input placement carries the shard.
    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(
            torch_output_tensor, output_tensor, input_a_tensor_placement, input_b_tensor_placement
        )

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
