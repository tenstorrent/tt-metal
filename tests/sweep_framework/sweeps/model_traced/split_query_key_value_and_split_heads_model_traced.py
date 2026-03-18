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

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs


def mesh_device_fixture():
    mesh_shape = get_mesh_shape()
    if mesh_shape:
        try:
            device = create_mesh_device(mesh_shape)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_mesh_device(device)
        except Exception as e:
            print(f"Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
            device = ttnn.open_device(device_id=0, l1_small_size=32768, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=32768, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    num_heads=1,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    output_memory_config = kwargs.get("output_memory_config", None)
    op_kwargs = build_op_kwargs(
        kwargs, exclude={"num_heads", "compute_with_storage_grid_size"}, output_memory_config=output_memory_config
    )

    # Handle tuple input_a_shape for sample suite
    if isinstance(input_a_shape, (tuple, list)):
        shape = tuple(input_a_shape)
    else:
        shape = input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # Get golden function for split_query_key_value_and_split_heads
    golden_function = ttnn.get_golden_function(ttnn.transformer.split_query_key_value_and_split_heads)

    # Golden function expects 3D input [batch, seq_len, hidden_dim]
    # but traced configs have 4D [batch, 1, seq_len, hidden_dim]
    if len(shape) == 4 and shape[1] == 1:
        # Squeeze out the second dimension for golden function
        torch_input_3d = torch_input_tensor_a.squeeze(1)
        (
            torch_query_tensor,
            torch_key_tensor,
            torch_value_tensor,
        ) = golden_function(torch_input_3d, num_heads=num_heads)
        # Unsqueeze back to 4D to match ttnn output
        torch_query_tensor = torch_query_tensor.unsqueeze(1)
        torch_key_tensor = torch_key_tensor.unsqueeze(1)
        torch_value_tensor = torch_value_tensor.unsqueeze(1)
    else:
        (
            torch_query_tensor,
            torch_key_tensor,
            torch_value_tensor,
        ) = golden_function(torch_input_tensor_a, num_heads=num_heads)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # The operation expects 3D input [batch, seq_len, hidden_dim]
    # but traced configs have 4D [batch, 1, seq_len, hidden_dim]
    needs_unsqueeze = False
    if len(shape) == 4 and shape[1] == 1:
        # Squeeze out the second dimension for ttnn operation
        torch_input_tensor_a = torch_input_tensor_a.squeeze(1)
        needs_unsqueeze = True

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
            # Always create as DRAM interleaved — lines below convert sharded to DRAM anyway,
            # and traced shard specs may exceed the device's core count (TT_FATAL).
            input_tensor_a = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
    else:
        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=input_a_layout)

    start_time = start_measuring_time()
    # This operation splits QKV and heads - returns tuple of (Q, K, V)
    query_tensor, key_tensor, value_tensor = ttnn.transformer.split_query_key_value_and_split_heads(
        input_tensor_a, num_heads=num_heads, **op_kwargs
    )
    query_tensor = mesh_tensor_to_torch(query_tensor, device if is_mesh_device else None)
    key_tensor = mesh_tensor_to_torch(key_tensor, device if is_mesh_device else None)
    value_tensor = mesh_tensor_to_torch(value_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Unsqueeze back to 4D if we squeezed earlier
    if needs_unsqueeze:
        query_tensor = query_tensor.unsqueeze(1)
        key_tensor = key_tensor.unsqueeze(1)
        value_tensor = value_tensor.unsqueeze(1)

    # Check with PCC for all three outputs
    # check_with_pcc returns (bool, str) tuple
    pcc_q = check_with_pcc(torch_query_tensor, query_tensor, 0.999)
    pcc_k = check_with_pcc(torch_key_tensor, key_tensor, 0.999)
    pcc_v = check_with_pcc(torch_value_tensor, value_tensor, 0.999)

    # All three must pass for overall success
    all_pass = pcc_q[0] and pcc_k[0] and pcc_v[0]
    # Use minimum PCC value as the reported value
    min_pcc_value = min(float(pcc_q[1]), float(pcc_k[1]), float(pcc_v[1]))
    pcc_result = (all_pass, str(min_pcc_value))

    return [pcc_result, e2e_perf]
