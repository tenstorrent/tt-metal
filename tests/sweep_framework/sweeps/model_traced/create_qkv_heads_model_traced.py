# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("create_qkv_heads", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 192)],  # Must be divisible for QKV split: 192 = 1 * 3 * 64
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
        "num_heads": [1],
        "num_kv_heads": [1],
        "transpose_k_heads": [False],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    """Custom device fixture with DispatchCoreConfig to free up more compute cores"""
    device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.device.DispatchCoreConfig())
    device_name = ttnn.get_arch_name()

    yield (device, device_name)

    ttnn.close_device(device)
    del device


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    num_heads,
    num_kv_heads,
    transpose_k_heads=False,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,  # Accept traced_source, traced_machine_info, etc.
) -> list:
    torch.manual_seed(0)

    # Handle tuple input_shape for sample suite
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    # This operation requires specific input format: QKV interleaved
    # For simplicity, we'll create random tensors and compute golden output
    # In real usage, this comes from a matmul with specific weight arrangement

    # Traced shape is [batch, 1, seq_len, hidden_dim] where hidden_dim = (num_q_heads + 2*num_kv_heads) * head_dim
    # The input contains Q, K, V interleaved
    if len(shape) == 4:
        # Shape is [batch, 1, seq_len, hidden_dim]
        batch, _, seq_len, hidden_dim = shape
    elif len(shape) == 3:
        # Shape is [batch, seq_len, hidden_dim]
        batch, seq_len, hidden_dim = shape
    else:
        raise ValueError(f"Unexpected input shape: {shape}")

    # Infer head_dim from the input dimensions
    # hidden_dim = num_heads * head_dim + 2 * num_kv_heads * head_dim
    head_dim = hidden_dim // (num_heads + 2 * num_kv_heads)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-0.1, high=0.1, dtype=torch.float32), input_a_dtype
    )(shape)

    # Create golden output: split QKV and reshape heads
    torch_input_flat = torch_input_tensor_a.reshape(batch, seq_len, -1)
    (ref_q, ref_k, ref_v) = torch.split(
        torch_input_flat, [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
    )
    ref_q = torch.reshape(ref_q, [batch, seq_len, num_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)

    # Apply transpose_k_heads if specified (like unit test does)
    if transpose_k_heads:
        ref_k = ref_k.transpose(-2, -1)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Build from_torch arguments based on storage_type
    from_torch_kwargs = {
        "dtype": input_a_dtype,
        "layout": input_a_layout,
    }

    # Only add device and memory_config if not HOST storage
    if not is_host:
        from_torch_kwargs["device"] = device
        # Check if memory config is sharded - create_qkv_heads requires tile-aligned shard shapes
        # If sharded, use interleaved instead to avoid non-tile-aligned shard validation errors
        is_input_sharded = False

        # Handle dict memory configs (from JSON deserialization in pipeline)
        if isinstance(input_a_memory_config, dict):
            # Skip dict memory configs - just use DRAM_MEMORY_CONFIG
            from_torch_kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
        else:
            # Handle proper ttnn MemoryConfig objects
            if hasattr(input_a_memory_config, "is_sharded") and callable(
                getattr(input_a_memory_config, "is_sharded", None)
            ):
                is_input_sharded = input_a_memory_config.is_sharded()
            elif hasattr(input_a_memory_config, "memory_layout"):
                # Check memory_layout attribute for sharded types
                is_input_sharded = input_a_memory_config.memory_layout in [
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ]

            if is_input_sharded:
                from_torch_kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
            else:
                from_torch_kwargs["memory_config"] = input_a_memory_config

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs)

    # If output memory config is sharded, use interleaved instead
    actual_output_memory_config = output_memory_config
    is_output_sharded = False

    # Handle dict memory configs (from JSON deserialization in pipeline)
    if isinstance(output_memory_config, dict):
        # Skip dict memory configs - just use DRAM_MEMORY_CONFIG
        actual_output_memory_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        # Handle proper ttnn MemoryConfig objects
        if hasattr(output_memory_config, "is_sharded") and callable(getattr(output_memory_config, "is_sharded", None)):
            is_output_sharded = output_memory_config.is_sharded()
        elif hasattr(output_memory_config, "memory_layout"):
            is_output_sharded = output_memory_config.memory_layout in [
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ]

        if is_output_sharded:
            actual_output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    start_time = start_measuring_time()
    # This operation creates QKV heads from input tensor
    # Note: Using nlp_create_qkv_heads which is the actual implementation
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        input_tensor_a,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=transpose_k_heads,
        memory_config=actual_output_memory_config,
    )
    q = ttnn.to_torch(q)
    k = ttnn.to_torch(k)
    v = ttnn.to_torch(v)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC for all three outputs using check_with_pcc
    # Use 0.99 as common threshold for all dtypes
    pcc_q = check_with_pcc(ref_q, q, 0.99)
    pcc_k = check_with_pcc(ref_k, k, 0.99)
    pcc_v = check_with_pcc(ref_v, v, 0.99)

    # All three must pass - combine the results
    all_passed = pcc_q[0] and pcc_k[0] and pcc_v[0]

    # Combine messages if any fail
    if not all_passed:
        messages = []
        if not pcc_q[0]:
            messages.append(f"Q: {pcc_q[1]}")
        if not pcc_k[0]:
            messages.append(f"K: {pcc_k[1]}")
        if not pcc_v[0]:
            messages.append(f"V: {pcc_v[1]}")
        combined_message = "; ".join(messages)
        pcc_result = (False, combined_message)
    else:
        # All passed, return the Q PCC result (or we could combine messages)
        pcc_result = pcc_q

    return [pcc_result, e2e_perf]
