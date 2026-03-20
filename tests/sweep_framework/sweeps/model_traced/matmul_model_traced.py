# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    """
    Override default device fixture.
    Creates mesh device if MESH_DEVICE_SHAPE is set, otherwise single device.
    """
    mesh_shape = get_mesh_shape()

    if mesh_shape:
        # Create mesh device based on env var
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
        # Single device (default)
        device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


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

    # Use output_memory_config as fallback for memory_config in op_kwargs
    if "memory_config" not in op_kwargs and output_memory_config is not None:
        op_kwargs["memory_config"] = output_memory_config

    # V2 format provides separate shapes for each input
    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
    shape_b = tuple(input_b_shape) if input_b_shape and isinstance(input_b_shape, (list, tuple)) else shape_a

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
    except RuntimeError:
        # If direct creation fails, try interleaved->sharded conversion
        input_tensor_a_interleaved = ttnn.from_torch(
            torch_input_tensor_a,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if hasattr(input_a_memory_config, "shard_spec") and input_a_memory_config.shard_spec is not None:
            input_tensor_a = ttnn.interleaved_to_sharded(input_tensor_a_interleaved, input_a_memory_config)
        else:
            input_tensor_a = input_tensor_a_interleaved

    # Create input_b tensor - matmul requires input_b to be INTERLEAVED
    # If traced config has input_b as sharded, convert to INTERLEAVED to match operation requirements
    input_b_is_sharded = (
        hasattr(input_b_memory_config, "shard_spec")
        and input_b_memory_config.shard_spec is not None
        and input_b_memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    )

    if input_b_is_sharded:
        # matmul requires input_b to be INTERLEAVED, so convert sharded to interleaved
        # Create as interleaved first
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
        except RuntimeError:
            # If direct creation fails, try interleaved->sharded conversion
            input_tensor_b_interleaved = ttnn.from_torch(
                torch_input_tensor_b,
                dtype=input_b_dtype,
                layout=input_b_layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if hasattr(input_b_memory_config, "shard_spec") and input_b_memory_config.shard_spec is not None:
                input_tensor_b = ttnn.interleaved_to_sharded(input_tensor_b_interleaved, input_b_memory_config)
            else:
                input_tensor_b = input_tensor_b_interleaved

    start_time = start_measuring_time()
    output_tensor = ttnn.matmul(input_tensor_a, input_tensor_b, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
