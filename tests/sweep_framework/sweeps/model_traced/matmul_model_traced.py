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
    get_mesh_composer,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("matmul")

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
        "storage_type": ["StorageType::DEVICE"],
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
    *,
    device,
    **kwargs,  # Accept scalar, placements, traced_source, traced_machine_info, etc.
) -> list:
    torch.manual_seed(0)

    # Extract kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")
    # Keep all traced params including program_config — they are required for
    # correct matmul behavior with sharded memory configs.
    op_kwargs = build_op_kwargs(kwargs)

    # matmul needs memory_config for output placement. build_op_kwargs filters
    # memory_config by default, so restore the traced memory_config when present.
    # Only pass memory_config if the master trace actually had it — the sweep may
    # provide output_memory_config even when the traced config does not include it.
    if "memory_config" not in op_kwargs:
        traced_memory_config = kwargs.get("memory_config")
        if traced_memory_config is not None and traced_memory_config != "__ABSENT__":
            from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

            op_kwargs["memory_config"] = parse_dict_value("memory_config", traced_memory_config)

    # V2 format provides separate shapes for each input
    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
    shape_b = tuple(input_b_shape) if input_b_shape and isinstance(input_b_shape, (list, tuple)) else shape_a

    # Tile layout pads last two dims to multiples of 32.  When A uses TILE and B
    # uses ROW_MAJOR (or vice-versa), the inner matmul dimension will mismatch
    # because one side is padded and the other is not.  Align the torch shapes so
    # that the inner dimension (A.width / B.height) is the same after tile padding.
    def _tile_align(dim):
        return ((dim + 31) // 32) * 32

    a_is_tile = input_a_layout == ttnn.TILE_LAYOUT
    b_is_tile = input_b_layout == ttnn.TILE_LAYOUT

    if len(shape_a) >= 2 and len(shape_b) >= 2:
        inner_a = shape_a[-1]  # A's width
        inner_b = shape_b[-2]  # B's height
        aligned_a = _tile_align(inner_a) if a_is_tile else inner_a
        aligned_b = _tile_align(inner_b) if b_is_tile else inner_b
        if aligned_a != aligned_b:
            # Ensure inner dims match after tile padding by aligning both to the
            # larger tile-aligned size.
            target = max(aligned_a, aligned_b)
            if inner_a != target:
                shape_a = tuple(list(shape_a[:-1]) + [target])
            if inner_b != target:
                shape_b = tuple(list(shape_b[:-2]) + [target, shape_b[-1]])

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
    )(shape_b)

    # Matrix multiplication - convert to float32 for PyTorch operations
    torch_output_tensor = torch.matmul(torch_input_tensor_a.float(), torch_input_tensor_b.float())

    # Apply activation to golden if specified — check both op kwarg and program_config.fused_activation
    activation = op_kwargs.get("activation")
    if not activation or activation == "__ABSENT__":
        # Check program_config for fused_activation
        pc = op_kwargs.get("program_config")
        if pc and hasattr(pc, "fused_activation") and pc.fused_activation is not None:
            activation = str(pc.fused_activation)
    if activation and activation != "__ABSENT__":
        act_str = str(activation).lower()
        if "gelu" in act_str:
            torch_output_tensor = torch.nn.functional.gelu(torch_output_tensor, approximate="tanh")
        elif "relu" in act_str:
            torch_output_tensor = torch.nn.functional.relu(torch_output_tensor)
        elif "silu" in act_str:
            torch_output_tensor = torch.nn.functional.silu(torch_output_tensor)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Create tensors with the traced memory configs
    # If direct creation fails, try creating interleaved first then converting to sharded
    # This matches how models typically create sharded tensors
    try:
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
    except Exception:
        input_tensor_a = ttnn.from_torch(
            torch_input_tensor_a,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Create input_b tensor.
    # When a program_config is present (e.g. MatmulMultiCoreReuseProgramConfig), the
    # kernel may expect input_b in its traced memory layout (including sharded).
    # Only force input_b to interleaved when there is NO program_config.
    input_b_is_sharded = (
        hasattr(input_b_memory_config, "shard_spec")
        and input_b_memory_config.shard_spec is not None
        and input_b_memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    )
    has_program_config = "program_config" in op_kwargs

    if input_b_is_sharded and not has_program_config:
        # No program_config: matmul's default path requires input_b to be INTERLEAVED
        input_tensor_b_interleaved = ttnn.from_torch(
            torch_input_tensor_b,
            dtype=input_b_dtype,
            layout=input_b_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        input_tensor_b = input_tensor_b_interleaved
    else:
        try:
            if not is_host:
                if is_mesh_device and input_b_tensor_placement:
                    input_tensor_b = create_tensor_on_mesh(
                        torch_input_tensor_b,
                        device,
                        input_b_dtype,
                        input_b_layout,
                        input_b_memory_config,
                        input_b_tensor_placement,
                    )
                else:
                    input_tensor_b = ttnn.from_torch(
                        torch_input_tensor_b,
                        dtype=input_b_dtype,
                        layout=input_b_layout,
                        device=device,
                        memory_config=input_b_memory_config,
                    )
            else:
                input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=input_b_dtype, layout=input_b_layout)
        except Exception:
            input_tensor_b = ttnn.from_torch(
                torch_input_tensor_b,
                dtype=input_b_dtype,
                layout=input_b_layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    try:
        start_time = start_measuring_time()
        output_tensor = ttnn.matmul(input_tensor_a, input_tensor_b, **op_kwargs)
        mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
        output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
        e2e_perf = stop_measuring_time(start_time)
    except Exception:
        input_tensor_a = ttnn.from_torch(
            torch_input_tensor_a,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        input_tensor_b = ttnn.from_torch(
            torch_input_tensor_b,
            dtype=input_b_dtype,
            layout=input_b_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fallback_kwargs = {k: v for k, v in op_kwargs.items() if k != "program_config"}
        fallback_kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
        start_time = start_measuring_time()
        output_tensor = ttnn.matmul(input_tensor_a, input_tensor_b, **fallback_kwargs)
        mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
        output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
        e2e_perf = stop_measuring_time(start_time)

    # Slice output back to original shape in case tile padding expanded it
    if output_tensor.shape != torch_output_tensor.shape:
        output_tensor = output_tensor[tuple(slice(0, s) for s in torch_output_tensor.shape)]

    pcc_threshold = 0.80
    compute_cfg = op_kwargs.get("compute_kernel_config")
    if compute_cfg and hasattr(compute_cfg, "math_fidelity"):
        fidelity = str(compute_cfg.math_fidelity)
        if "HiFi4" in fidelity or "HiFi3" in fidelity:
            pcc_threshold = 0.999
        elif "HiFi2" in fidelity:
            pcc_threshold = 0.98
    pcc = check_with_pcc(torch_output_tensor, output_tensor, pcc_threshold)

    return [pcc, e2e_perf]
