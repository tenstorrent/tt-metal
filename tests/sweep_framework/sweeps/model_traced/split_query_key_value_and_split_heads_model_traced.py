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


# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("split_query_key_value_and_split_heads")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 96)],  # Must be divisible for QKV split
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
        "num_heads": [1],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


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
            device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
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
        kwargs, exclude={"compute_with_storage_grid_size"}, output_memory_config=output_memory_config
    )

    # num_heads comes from named param in run(), not from op_kwargs.
    # Ensure it's in op_kwargs for passing to the op.
    if "num_heads" not in op_kwargs and num_heads is not None:
        op_kwargs["num_heads"] = int(num_heads)

    # Handle tuple input_a_shape for sample suite
    if isinstance(input_a_shape, (tuple, list)):
        shape = tuple(input_a_shape)
    else:
        shape = input_a_shape

    # Traced configs may have 4D [batch, 1, seq_len, hidden_dim].
    # The experimental op handles 4D natively but requires 12+ column grid.
    # Fall back to standard op (3D) if device grid is too small.
    device_grid = device.compute_with_storage_grid_size()
    use_experimental = len(shape) == 4 and shape[1] == 1 and device_grid.x >= 12

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    if use_experimental:
        # Experimental op golden from bert_large reference test:
        # Input [batch, 1, seq, hidden] → split hidden into 3 equal parts (Q, K, V)
        # → reshape each [batch, seq, num_heads, head_dim] → transpose to [batch, num_heads, seq, head_dim]
        # K gets extra transpose to [batch, num_heads, head_dim, seq]
        batch = shape[0]
        seq_len = shape[2]
        hidden = shape[3]
        head_dim = hidden // (3 * num_heads)
        per_head = num_heads * head_dim  # size of each Q/K/V chunk

        ref_q, ref_k, ref_v = torch.split(torch_input_tensor_a.float(), per_head, dim=-1)
        torch_query_tensor = ref_q.reshape(batch, seq_len, num_heads, head_dim).transpose(-3, -2)
        torch_key_tensor = ref_k.reshape(batch, seq_len, num_heads, head_dim).transpose(-3, -2).transpose(-2, -1)
        torch_value_tensor = ref_v.reshape(batch, seq_len, num_heads, head_dim).transpose(-3, -2)
    else:
        # Standard op needs 3D input — squeeze 4D if needed
        golden_function = ttnn.get_golden_function(ttnn.transformer.split_query_key_value_and_split_heads)
        needs_squeeze = len(shape) == 4 and shape[1] == 1
        golden_input = torch_input_tensor_a.squeeze(1) if needs_squeeze else torch_input_tensor_a
        if needs_squeeze:
            torch_input_tensor_a = torch_input_tensor_a.squeeze(1)
        (
            torch_query_tensor,
            torch_key_tensor,
            torch_value_tensor,
        ) = golden_function(golden_input, num_heads=num_heads)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # When falling back to standard op (needs_squeeze), use DRAM — the traced sharded
    # config was for 4D shape and won't work with the squeezed 3D tensor.
    safe_mem = input_a_memory_config if not needs_squeeze else ttnn.DRAM_MEMORY_CONFIG

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor_a = create_tensor_on_mesh(
                torch_input_tensor_a,
                device,
                input_a_dtype,
                input_a_layout,
                safe_mem,
                input_a_tensor_placement,
            )
        else:
            input_tensor_a = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if (
                not needs_squeeze
                and hasattr(input_a_memory_config, "is_sharded")
                and input_a_memory_config.is_sharded()
            ):
                try:
                    input_tensor_a = ttnn.to_memory_config(input_tensor_a, input_a_memory_config)
                except Exception:
                    pass
    else:
        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=input_a_layout)

    start_time = start_measuring_time()
    if use_experimental:
        # Experimental op takes compute_with_storage_grid_size as positional arg
        grid = kwargs.get("compute_with_storage_grid_size")
        if isinstance(grid, dict):
            from tests.sweep_framework.master_config_loader_v2 import dict_to_core_grid

            grid = dict_to_core_grid(grid)
        if grid is None:
            grid = device.compute_with_storage_grid_size()
        # Convert CoreGrid to CoreCoord if needed
        if hasattr(grid, "x") and hasattr(grid, "y") and type(grid).__name__ == "CoreGrid":
            grid = ttnn.CoreCoord(grid.x, grid.y)
        query_tensor, key_tensor, value_tensor = ttnn.experimental.split_query_key_value_and_split_heads(
            input_tensor_a, grid, **op_kwargs
        )
    else:
        # Standard op: don't pass memory_config when falling back from experimental
        # (sharded output memory_config produces wrong results on standard op)
        std_kwargs = {k: v for k, v in op_kwargs.items() if k != "memory_config"} if needs_squeeze else op_kwargs
        query_tensor, key_tensor, value_tensor = ttnn.transformer.split_query_key_value_and_split_heads(
            input_tensor_a, **std_kwargs
        )
    query_tensor = mesh_tensor_to_torch(query_tensor, device if is_mesh_device else None)
    key_tensor = mesh_tensor_to_torch(key_tensor, device if is_mesh_device else None)
    value_tensor = mesh_tensor_to_torch(value_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # No unsqueeze needed — experimental op returns 5D [B,1,H,S,D] matching golden

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
