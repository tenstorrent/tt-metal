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
    get_model_traced_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

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
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    arg1=None,  # May contain scalar value or second input
    use_legacy=None,  # Legacy mode flag
    memory_config="__ABSENT__",  # __ABSENT__ sentinel: distinguishes "not in trace" from "trace had None"
    dtype="__ABSENT__",  # __ABSENT__ sentinel: distinguishes "not in trace" from "trace had None"
    input_tensor_a_activations="__ABSENT__",  # Fused activations (e.g., SILU) for binary ops
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
    # Only include memory_config/dtype when explicitly set in the traced config.
    # V2 loader uses "__ABSENT__" for keys missing from a config, but sometimes
    # fills None for absent keys too.  For multiply, master configs that match
    # do NOT have dtype/memory_config, so we must filter both None and __ABSENT__
    # to avoid injecting extra keys into the re-trace.
    extra_kw = {}
    if memory_config is not None and memory_config != "__ABSENT__":
        extra_kw["memory_config"] = memory_config
    if dtype is not None and dtype != "__ABSENT__":
        extra_kw["dtype"] = dtype
    op_kwargs = build_op_kwargs(kwargs, exclude={"scalar", "output_tensor"}, output_memory_config=output_memory_config,
        extra_kwargs=extra_kw,
    )

    # Handle fused activations (e.g., SILU) from traced config.
    # input_tensor_a_activations is now a named param to guarantee it's received
    # from the V2 loader (not lost in **kwargs processing).
    # _activations suffix is filtered by build_op_kwargs as tensor metadata,
    # but input_tensor_a_activations is actually an op kwarg for binary ops.
    if input_tensor_a_activations != "__ABSENT__" and input_tensor_a_activations is not None:
        activations_raw = input_tensor_a_activations
        if isinstance(activations_raw, list) and len(activations_raw) > 0:
            parsed_activations = []
            for act in activations_raw:
                if isinstance(act, dict):
                    repr_str = act.get("repr", "")
                    # Parse "UnaryOpType.SILU" -> ttnn.UnaryOpType.SILU
                    if "SILU" in repr_str:
                        parsed_activations.append(ttnn.UnaryOpType.SILU)
                    elif "RELU" in repr_str:
                        parsed_activations.append(ttnn.UnaryOpType.RELU)
                    elif "GELU" in repr_str:
                        parsed_activations.append(ttnn.UnaryOpType.GELU)
                    else:
                        # Unknown activation type — skip to avoid crashes
                        parsed_activations = []
                        break
                elif hasattr(act, "name"):
                    # Already a ttnn.UnaryOpType enum
                    parsed_activations.append(act)
            if parsed_activations:
                op_kwargs["input_tensor_a_activations"] = parsed_activations

    # Handle pre-allocated output_tensor if present in traced config.
    output_tensor_raw = kwargs.get("output_tensor", "__ABSENT__")
    if output_tensor_raw != "__ABSENT__" and output_tensor_raw is not None and isinstance(output_tensor_raw, dict):
        try:
            out_shape = output_tensor_raw.get("original_shape", list(input_a_shape))
            out_dtype = input_a_dtype  # Default to input dtype
            torch_out = torch.zeros(out_shape, dtype=torch.float32)
            output_preallocated = ttnn.from_torch(
                torch_out, dtype=out_dtype, layout=input_a_layout,
                device=device, memory_config=input_a_memory_config,
            )
            op_kwargs["output_tensor"] = output_preallocated
        except Exception:
            pass  # Skip pre-allocated output on failure

    # V2 format provides separate shapes for each input
    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
    shape_b = tuple(input_b_shape) if input_b_shape and isinstance(input_b_shape, (list, tuple)) else input_b_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    # Determine if fused activation (e.g., SILU) applies to input_tensor_a
    has_silu_activation = "input_tensor_a_activations" in op_kwargs

    # Check if this is a scalar multiply operation (shape_b is None or scalar is provided)
    if shape_b is None or scalar is not None:
        # Tensor-scalar multiply: use the scalar value directly.
        # The scalar may come from 'scalar' kwarg, 'arg1' param, or default to 2.0.
        scalar_value = scalar if scalar is not None else (arg1 if arg1 is not None else 2.0)
        golden_a = torch.nn.functional.silu(torch_input_tensor_a) if has_silu_activation else torch_input_tensor_a
        torch_output_tensor = torch.mul(golden_a, scalar_value)
        is_scalar_multiply = True
    else:
        # Tensor-tensor multiply: generate second tensor
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
        )(shape_b)
        golden_a = torch.nn.functional.silu(torch_input_tensor_a) if has_silu_activation else torch_input_tensor_a
        torch_output_tensor = torch.mul(golden_a, torch_input_tensor_b)
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

    # Slice output back to original shape in case tile padding expanded it
    if output_tensor.shape != torch_output_tensor.shape:
        output_tensor = output_tensor[tuple(slice(0, s) for s in torch_output_tensor.shape)]

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
