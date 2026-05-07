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
    reconcile_golden_to_actual,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_positional_args


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
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

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

    # Remove output_memory_config / memory_config from op_kwargs since we pass it positionally
    op_kwargs.pop("output_memory_config", None)
    op_kwargs.pop("memory_config", None)

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
                shard_grid = shard_spec.grid
                core_ranges = shard_grid.ranges() if hasattr(shard_grid, "ranges") else []
                for cr in core_ranges:
                    if cr.start.x >= grid.x or cr.start.y >= grid.y or cr.end.x >= grid.x or cr.end.y >= grid.y:
                        shard_ok = False
                        break
                if shard_ok:
                    total_shard_cores = 0
                    for cr in core_ranges:
                        total_shard_cores += (cr.end.x - cr.start.x + 1) * (cr.end.y - cr.start.y + 1)
                    if total_shard_cores > num_cores:
                        shard_ok = False
        except Exception:
            shard_ok = False

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

    # Run sharded_to_interleaved (pass output config as positional arg to match master trace)
    start_time = start_measuring_time()
    if s2i_output_config is not None:
        output_tensor = ttnn.sharded_to_interleaved(input_tensor, s2i_output_config, **op_kwargs)
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
