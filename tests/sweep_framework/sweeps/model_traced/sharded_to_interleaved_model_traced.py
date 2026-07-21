# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0


from functools import partial

import torch

import ttnn
from models.common.utility_functions import torch_random

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    create_mesh_device,
    create_tensor_on_mesh,
    dispatch_axis_for_grid,
    get_mesh_composer,
    get_model_traced_mesh_shape,
    mesh_tensor_to_torch,
    reconcile_golden_to_actual,
    shard_grid_bounds,
)
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_positional_args

# Device opened per-vector (see _ensure_vector_device): the input shard grids
# straddle both dispatch axes — some need x=7 (ROW), others y=9 (COL) — so a
# single per-suite axis can't place every shard. Pick the axis each vector's
# input shard grid needs.
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


from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("sharded_to_interleaved")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


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

    # Open this vector's device with the dispatch axis its input shard grid
    # needs (read the raw dict before parsing). A grid touching x=7 needs ROW;
    # y=9 needs COL — a single per-suite axis can't place both.
    _ax = dispatch_axis_for_grid(*shard_grid_bounds(input_a_memory_config))
    device = _ensure_vector_device(_ax)

    # Parse input_a_memory_config if it's a dict (from vector data)
    if isinstance(input_a_memory_config, dict):
        from tests.sweep_framework.master_config_loader_v2 import dict_to_memory_config

        input_a_memory_config = dict_to_memory_config(input_a_memory_config)

    # Extract kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    pos_args = extract_positional_args(kwargs)
    traced_output_mem_config = pos_args.get(1, None)
    traced_output_dtype = pos_args.get(2, None)
    _positional_dtype = None
    if traced_output_dtype is not None:
        from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

        parsed_dt = (
            parse_dict_value("dtype", traced_output_dtype)
            if isinstance(traced_output_dtype, dict)
            else traced_output_dtype
        )
        if parsed_dt is not None:
            _positional_dtype = parsed_dt

    # Determine the output memory config: prefer traced arg1 (positional), then explicit param
    if traced_output_mem_config is not None:
        s2i_output_config = traced_output_mem_config
    elif output_memory_config is not None:
        from tests.sweep_framework.master_config_loader_v2 import dict_to_memory_config

        s2i_output_config = (
            dict_to_memory_config(output_memory_config)
            if isinstance(output_memory_config, dict)
            else output_memory_config
        )
    else:
        s2i_output_config = None

    # Only pass output config if it's interleaved (sharded_to_interleaved requires interleaved output)
    if s2i_output_config is not None and hasattr(s2i_output_config, "memory_layout"):
        if s2i_output_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            s2i_output_config = None

    # Remove output_memory_config / memory_config / output_dtype from op_kwargs since we pass positionally
    op_kwargs.pop("output_memory_config", None)
    op_kwargs.pop("memory_config", None)

    # Some configs pass the output config as the `memory_config` kwarg (arg1 is
    # None/absent) rather than positionally. Reproduce that call form so the
    # recorded memory_config kwarg matches the master trace (else it's a
    # memory_config extra_key diff).
    _mc_kwarg = kwargs.get("memory_config")
    _no_positional_arg1 = traced_output_mem_config is None or traced_output_mem_config == "__ABSENT__"
    if _no_positional_arg1 and _mc_kwarg is not None and _mc_kwarg != "__ABSENT__":
        if isinstance(_mc_kwarg, dict):
            from tests.sweep_framework.master_config_loader_v2 import dict_to_memory_config as _d2mc

            _parsed_mc = _d2mc(_mc_kwarg)
        else:
            _parsed_mc = _mc_kwarg  # already a ttnn.MemoryConfig (V2 loader parsed it)
        if _parsed_mc is not None and getattr(_parsed_mc, "memory_layout", None) == ttnn.TensorMemoryLayout.INTERLEAVED:
            op_kwargs["memory_config"] = _parsed_mc
            s2i_output_config = None  # pass via memory_config kwarg, not positionally
    op_kwargs.pop("output_dtype", None)

    # Handle input_a_shape - ensure it's always a tuple
    if input_a_shape is None:
        raise ValueError("input_a_shape cannot be None")

    # Handle list/tuple
    if isinstance(input_a_shape, (tuple, list)):
        shape = tuple(input_a_shape)
    # Handle single int/float (invalid, but provide clear error)
    elif isinstance(input_a_shape, (int, float)):
        raise ValueError(f"input_a_shape must be a list or tuple, got {type(input_a_shape).__name__}: {input_a_shape}")
    # Handle other iterables (numpy arrays, etc.)
    else:
        try:
            shape = tuple(input_a_shape)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Cannot convert input_a_shape to tuple: {type(input_a_shape).__name__} = {input_a_shape}. Error: {e}"
            )

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # Check if input is already interleaved or needs to be sharded
    is_input_sharded = (
        hasattr(input_a_memory_config, "memory_layout")
        and input_a_memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    )

    if is_input_sharded:
        # Input should be sharded - create interleaved first, then convert to sharded
        if is_mesh_device and input_a_tensor_placement:
            input_tensor_interleaved = create_tensor_on_mesh(
                torch_input_tensor_a,
                device,
                input_a_dtype,
                input_a_layout,
                ttnn.DRAM_MEMORY_CONFIG,
                input_a_tensor_placement,
            )
        else:
            input_tensor_interleaved = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Reproduce the master's sharded arg0: attempt interleaved_to_sharded
        # with the traced sharded memory_config. Only fall back to the DRAM-
        # interleaved tensor if the device genuinely can't place the shard (the
        # op still accepts an interleaved input). The previous pre-check
        # heuristic rejected valid configs and recorded a DRAM arg0 — a
        # memory_config diff vs the master trace.
        try:
            input_tensor = ttnn.interleaved_to_sharded(input_tensor_interleaved, input_a_memory_config)
        except Exception:
            input_tensor = input_tensor_interleaved
    else:
        # Input is interleaved - use the traced config directly (op supports this)
        if is_mesh_device and input_a_tensor_placement:
            input_tensor = create_tensor_on_mesh(
                torch_input_tensor_a,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            input_tensor = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )

    # Run sharded_to_interleaved (pass output config as positional arg to match master trace)
    start_time = start_measuring_time()
    if s2i_output_config is not None:
        if _positional_dtype is not None:
            output_tensor = ttnn.sharded_to_interleaved(input_tensor, s2i_output_config, _positional_dtype, **op_kwargs)
        else:
            output_tensor = ttnn.sharded_to_interleaved(input_tensor, s2i_output_config, **op_kwargs)
    else:
        if _positional_dtype is not None:
            output_tensor = ttnn.sharded_to_interleaved(input_tensor, output_dtype=_positional_dtype, **op_kwargs)
        else:
            output_tensor = ttnn.sharded_to_interleaved(input_tensor, **op_kwargs)
    e2e_perf = stop_measuring_time(start_time)

    # Verify output is interleaved
    output_mem_config = output_tensor.memory_config()
    if output_mem_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(
            f"sharded_to_interleaved should produce interleaved output, but got {output_mem_config.memory_layout}"
        )

    # Verify correctness by comparing with original torch tensor
    mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
    output_torch = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    if is_mesh_device:
        torch_input_tensor_a = reconcile_golden_to_actual(torch_input_tensor_a, output_torch, input_a_tensor_placement)
    pcc = check_with_pcc(torch_input_tensor_a, output_torch, 0.999)

    return [pcc, e2e_perf]
