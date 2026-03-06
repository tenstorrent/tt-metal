# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("update_cache")

# Parameters provided to the test vector generator are defined here.
parameters = {}

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
    input_b_shape=None,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    scalar=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    """
    update_cache operation: updates a KV cache tensor with new values
    Args:
        cache (input_a): The cache tensor to update [num_users, num_heads, max_seq_len, head_dim]
        input (input_b): The new values to write [1, num_heads, 1, head_dim] (permuted to [1, num_heads, num_users, head_dim])
        cache_idx (scalar['update_index']): Index in sequence dimension to update
    """
    torch.manual_seed(0)

    # Extract kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs)

    # V2 format provides separate shapes for each input
    # Parse input_a_shape (cache shape) and input_b_shape (input shape)
    if isinstance(input_a_shape, dict):
        # Legacy dict format with 'self'/'other' keys
        import ast

        shape_a_str = input_a_shape.get("self")
        shape_b_str = input_a_shape.get("other")

        if isinstance(shape_a_str, str):
            shape_a = ast.literal_eval(shape_a_str)
        else:
            shape_a = shape_a_str

        if isinstance(shape_b_str, str):
            shape_b = ast.literal_eval(shape_b_str)
        else:
            shape_b = shape_b_str
    else:
        # V2 format: input_a_shape is the cache shape, input_b_shape is the input shape
        shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
        if input_b_shape:
            shape_b = tuple(input_b_shape) if isinstance(input_b_shape, (list, tuple)) else input_b_shape
        else:
            # Fallback if not provided
            return [1.0, 0.0]

    # Default input_b params to input_a params if not provided
    if input_b_dtype is None:
        input_b_dtype = input_a_dtype
    if input_b_layout is None:
        input_b_layout = input_a_layout
    if input_b_memory_config is None:
        input_b_memory_config = input_a_memory_config

    # Parse scalars - cache_idx and batch_offset
    if scalar and isinstance(scalar, dict):
        cache_idx = int(scalar.get("update_index", shape_a[2] // 2))
        batch_offset = int(scalar.get("batch_offset", 0))
    else:
        # Default to middle of cache sequence length
        cache_idx = shape_a[2] // 2 if len(shape_a) > 2 else 0
        batch_offset = 0

    # Generate cache tensor
    torch_cache = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        shape_a
    )

    # Generate input tensor
    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype)(
        shape_b
    )

    # Create expected output - update cache at cache_idx
    torch_output = torch_cache.clone()
    # update_cache signature: update_cache(cache_tensor, input_tensor, cache_idx, batch_offset)
    # cache: [num_users, num_heads, max_seq_len, head_dim]
    # input: [seq, num_heads, batch, head_dim] where seq=1, batch=1 (only first user)

    # For testing, we only update the cache for user at batch_offset
    user_data = torch_input[0, :, 0, :]  # [num_heads, head_dim]
    torch_output[batch_offset, :, cache_idx, :] = user_data

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Convert to TTNN tensors
    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            cache_tensor = create_tensor_on_mesh(
                torch_cache,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            cache_tensor = ttnn.from_torch(
                torch_cache,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        cache_tensor = ttnn.from_torch(torch_cache, dtype=input_a_dtype, layout=input_a_layout)

    # update_cache expects input in shape [batch=1, num_heads, seq_len=1, head_dim]
    # Our traced input is [seq=1, num_heads, num_users, head_dim]
    # Extract first user's data: [1, num_heads, 1, head_dim]
    torch_input_for_update = torch_input[:, :, 0:1, :]  # [1, num_heads, 1, head_dim]
    torch_input_for_update = torch_input_for_update.permute(2, 1, 0, 3)  # [1, num_heads, 1, head_dim]

    if not is_host:
        if is_mesh_device and input_b_tensor_placement:
            input_tensor = create_tensor_on_mesh(
                torch_input_for_update,
                device,
                input_b_dtype,
                input_b_layout,
                input_b_memory_config,
                input_b_tensor_placement,
            )
        else:
            input_tensor = ttnn.from_torch(
                torch_input_for_update,
                dtype=input_b_dtype,
                layout=input_b_layout,
                device=device,
                memory_config=input_b_memory_config,
            )
    else:
        input_tensor = ttnn.from_torch(torch_input_for_update, dtype=input_b_dtype, layout=input_b_layout)

    # Run operation
    start_time = start_measuring_time()
    output_tensor = ttnn.update_cache(
        cache_tensor,
        input_tensor,
        cache_idx,
        batch_offset=batch_offset,
        **op_kwargs,
    )
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Check PCC
    pcc_result = check_with_pcc(torch_output, output_tensor, 0.99)

    # Return result in the format expected by sweeps_runner: [(status, message), e2e_perf]
    return [pcc_result, e2e_perf]
