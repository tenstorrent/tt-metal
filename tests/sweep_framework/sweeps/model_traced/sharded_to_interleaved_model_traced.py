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
    infer_mesh_shape_from_params,
    detect_mesh_shape_from_hardware,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

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
    """
    Override default device fixture.
    Creates mesh device if MESH_DEVICE_SHAPE is set, otherwise single device.
    """
    mesh_shape = get_mesh_shape()

    if not mesh_shape:
        mesh_shape = infer_mesh_shape_from_params(model_traced_params)
    if not mesh_shape:
        mesh_shape = detect_mesh_shape_from_hardware()

    if mesh_shape:
        # Create mesh device based on env var
        try:
            device = create_mesh_device(mesh_shape)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_mesh_device(device)
        except Exception as e:
            print(f"Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
            device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        # Single device (default)
        device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


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

    # Extract kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

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

        # Validate shard spec fits device before calling interleaved_to_sharded (TT_FATAL can't be caught)
        shard_ok = True
        try:
            shard_spec = input_a_memory_config.shard_spec
            if shard_spec is not None:
                grid = device.compute_with_storage_grid_size()
                num_cores = grid.x * grid.y
                shard_shape = shard_spec.shape
                total_rows = 1
                for d in shape[:-1]:
                    total_rows *= d
                num_shards = (total_rows + shard_shape[0] - 1) // shard_shape[0]
                if num_shards > num_cores:
                    shard_ok = False
        except Exception:
            pass

        if shard_ok:
            # Convert to sharded using the traced config
            try:
                input_tensor = ttnn.interleaved_to_sharded(input_tensor_interleaved, input_a_memory_config)
            except (RuntimeError, ValueError) as e:
                error_msg = str(e)
                if "No core coordinate found" in error_msg or "core coordinate" in error_msg.lower():
                    raise ValueError(
                        f"Invalid core coordinates in sharding config: {error_msg}. "
                        f"This traced config uses cores that don't exist on this device."
                    )
                raise
        else:
            # Shard spec exceeds device cores — use interleaved tensor as-is
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

    # Run sharded_to_interleaved
    start_time = start_measuring_time()
    output_tensor = ttnn.sharded_to_interleaved(input_tensor, **op_kwargs)
    e2e_perf = stop_measuring_time(start_time)

    # Verify output is interleaved
    output_mem_config = output_tensor.memory_config()
    if output_mem_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(
            f"sharded_to_interleaved should produce interleaved output, but got {output_mem_config.memory_layout}"
        )

    # Verify correctness by comparing with original torch tensor
    output_torch = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    pcc = check_with_pcc(torch_input_tensor_a, output_torch, 0.999)

    return [pcc, e2e_perf]
