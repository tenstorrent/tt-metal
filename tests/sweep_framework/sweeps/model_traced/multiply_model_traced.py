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
    broadcast_torch_inputs_to_global,
    create_mesh_device,
    create_tensor_on_mesh,
    get_model_traced_mesh_shape,
    mesh_tensor_to_torch,
    reconcile_golden_to_actual,
)
from tests.sweep_framework.sweep_utils.op_kwargs_utils import (
    build_op_kwargs,
    check_with_pcc_safe,
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
model_traced_params = loader.get_suite_parameters("multiply")

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


def mesh_device_fixture():
    import os as _os

    mesh_shape = get_model_traced_mesh_shape()
    # Prefer WORKER COL: every traced config of this op runs on COL, but the
    # auto-detect over-routes the whole module to ROW from a single x=7/8-8 master
    # config, breaking the COL-only configs in single-pass (no-env) runs. Defer to
    # TTNN_DISPATCH_AXIS when set so CI's two-pass (row+col) is unchanged.
    _axis = (
        None
        if _os.environ.get("TTNN_DISPATCH_AXIS", "").strip().lower() in ("col", "row")
        else ttnn.DispatchCoreAxis.COL
    )
    device = create_mesh_device(mesh_shape, dispatch_core_axis=_axis)
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
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    arg1=None,  # May contain scalar value or second input
    memory_config=None,  # Alternative memory_config parameter
    dtype=None,  # Output dtype
    *,
    device,
    **kwargs,  # Accept scalar, placements, traced_source, traced_machine_info, etc.
) -> list:
    torch.manual_seed(0)

    # Extract kwargs
    scalar = kwargs.get("scalar", None)
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")  # MeshDevice has this method
    op_kwargs = build_op_kwargs(kwargs, exclude={"scalar"}, output_memory_config=output_memory_config, device=device)

    # V2 format provides separate shapes for each input
    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
    shape_b = tuple(input_b_shape) if input_b_shape and isinstance(input_b_shape, (list, tuple)) else input_b_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    # Check if this is a scalar multiply operation (shape_b is None or scalar is provided)
    if shape_b is None or scalar is not None:
        # Tensor-scalar multiply: use the scalar value directly.
        # The scalar may come from 'scalar' kwarg, 'arg1' param, or default to 2.0.
        scalar_value = scalar if scalar is not None else (arg1 if arg1 is not None else 2.0)
        # The scalar is a float (e.g. masking value -FLT_MAX ≈ -3.4e38) but the
        # tracer serializes it as a huge Python int; passing that to ttnn.multiply
        # tries to cast to unsigned ("can't convert negative int to unsigned").
        # Coerce numeric scalars to float so the op gets a real float scalar.
        if isinstance(scalar_value, bool):
            pass
        elif isinstance(scalar_value, int):
            scalar_value = float(scalar_value)
        torch_output_tensor = torch.mul(torch_input_tensor_a, scalar_value)
        is_scalar_multiply = True
    else:
        # Tensor-tensor multiply: generate second tensor
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(shape_b)
        ref_a, ref_b = broadcast_torch_inputs_to_global(
            torch_input_tensor_a,
            input_a_tensor_placement,
            torch_input_tensor_b,
            input_b_tensor_placement,
        )
        # Apply input_tensor_a_activations to ref_a so the golden matches the
        # kernel's fused activation. Master traces this as e.g.
        # [{"type": "UnaryOpType", "repr": "UnaryOpType.SILU"}].
        _act_a = kwargs.get("input_tensor_a_activations") or []
        if isinstance(_act_a, list):
            for _a in _act_a:
                _act_str = ""
                if isinstance(_a, dict):
                    _act_str = str(_a.get("repr", "")).upper()
                else:
                    _act_str = str(_a).upper()
                if "SILU" in _act_str or "SWISH" in _act_str:
                    ref_a = torch.nn.functional.silu(ref_a)
                elif "GELU" in _act_str:
                    _approx = "tanh" if "APPROX" in _act_str else "none"
                    ref_a = torch.nn.functional.gelu(ref_a, approximate=_approx)
                elif "RELU" in _act_str:
                    ref_a = torch.nn.functional.relu(ref_a)
                elif "TANH" in _act_str:
                    ref_a = torch.tanh(ref_a)
        torch_output_tensor = torch.mul(ref_a, ref_b)
        is_scalar_multiply = False

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
            # Regular single-device tensor.
            # If direct creation with sharded config fails, try DRAM→sharded conversion.
            try:
                input_tensor_a = ttnn.from_torch(
                    torch_input_tensor_a,
                    dtype=input_a_dtype,
                    layout=input_a_layout,
                    device=device,
                    memory_config=input_a_memory_config,
                )
            except RuntimeError:
                input_tensor_a = ttnn.from_torch(
                    torch_input_tensor_a,
                    dtype=input_a_dtype,
                    layout=input_a_layout,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if hasattr(input_a_memory_config, "is_sharded") and input_a_memory_config.is_sharded():
                    input_tensor_a = ttnn.to_memory_config(input_tensor_a, input_a_memory_config)
    else:
        # Host storage
        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=input_a_layout)

    # Re-add memory_config and dtype to op_kwargs when present in master config.
    # NOTE: memory_config and dtype are declared as named params on run(), so
    # they live in their own bindings — kwargs.get() would never see them.
    # Use __absent_keys__ to distinguish "master had kwarg=None" from "master never had kwarg".
    absent_keys = kwargs.get("__absent_keys__")
    has_absent_info = absent_keys is not None
    absent_keys = set(absent_keys or [])
    if has_absent_info and "memory_config" not in absent_keys:
        if memory_config is not None:
            parsed_mc = (
                parse_dict_value("memory_config", memory_config) if isinstance(memory_config, dict) else memory_config
            )
            if parsed_mc is not None:
                op_kwargs["memory_config"] = parsed_mc
            else:
                op_kwargs["memory_config"] = None
        else:
            op_kwargs["memory_config"] = None
    elif memory_config is not None:
        parsed_mc = (
            parse_dict_value("memory_config", memory_config) if isinstance(memory_config, dict) else memory_config
        )
        if parsed_mc is not None:
            op_kwargs["memory_config"] = parsed_mc
    if has_absent_info and "dtype" not in absent_keys:
        if dtype is not None:
            parsed_dt = parse_dict_value("dtype", dtype) if isinstance(dtype, dict) else dtype
            if parsed_dt is not None:
                op_kwargs["dtype"] = parsed_dt
            else:
                op_kwargs["dtype"] = None
        else:
            op_kwargs["dtype"] = None
    elif dtype is not None:
        parsed_dt = parse_dict_value("dtype", dtype) if isinstance(dtype, dict) else dtype
        if parsed_dt is not None:
            op_kwargs["dtype"] = parsed_dt

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
        if is_mesh_device and ot_placement:
            op_kwargs["output_tensor"] = create_tensor_on_mesh(
                torch_out_alloc, device, ot_dtype, ot_layout, ot_mem_cfg, ot_placement
            )
        elif not is_host:
            op_kwargs["output_tensor"] = ttnn.from_torch(
                torch_out_alloc, dtype=ot_dtype, layout=ot_layout, device=device, memory_config=ot_mem_cfg
            )

    start_time = start_measuring_time()

    if is_scalar_multiply:
        # Tensor-scalar multiply: pass scalar directly
        output_tensor = ttnn.multiply(input_tensor_a, scalar_value, **op_kwargs)
    else:
        # Tensor-tensor multiply: convert second tensor and multiply
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

        output_tensor = ttnn.multiply(input_tensor_a, input_tensor_b, **op_kwargs)

    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Reconcile the per-chip torch golden to the mesh-stitched actual output:
    # for sharded inputs the golden is the full/global tensor while the device
    # output is a per-device shard, so the golden must be sliced down to match
    # (the previous code sliced the *actual* to the golden, which is backwards
    # when the golden is larger -> shape-mismatch assert). Mirrors add_model_traced.
    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(
            torch_output_tensor, output_tensor, input_a_tensor_placement, input_b_tensor_placement
        )

    # Slice output back to original shape in case tile padding expanded it
    if output_tensor.shape != torch_output_tensor.shape:
        output_tensor = output_tensor[tuple(slice(0, s) for s in torch_output_tensor.shape)]

    # Check with PCC
    pcc = check_with_pcc_safe(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
