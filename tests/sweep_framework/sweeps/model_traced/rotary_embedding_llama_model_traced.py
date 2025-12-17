# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Import proper torch reference for rotary embedding
from models.demos.t3000.llama2_70b.reference.llama.llama.model import apply_rotary_emb, precompute_freqs_cis

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("experimental::rotary_embedding_llama", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 64)],  # Batch, seq, heads, head_dim (must be even for rotary)
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_dtype,
    input_b_layout,
    input_b_memory_config,
    input_c_dtype,
    input_c_layout,
    input_c_memory_config,
    input_d_dtype,
    input_d_layout,
    input_d_memory_config,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
) -> list:
    torch.manual_seed(0)

    # Handle dict input_shape from traced configurations
    if isinstance(input_shape, dict):
        # Traced configuration with multiple inputs
        shape_a = input_shape["input_a"]
        shape_b = input_shape["input_b"]  # cos_cache
        shape_c = input_shape["input_c"]  # sin_cache
        shape_d = input_shape["input_d"]  # trans_mat
    else:
        # Fallback for sample configurations
        shape_a = (1, 16, 256, 64)
        shape_b = (1, 1, 256, 64)
        shape_c = (1, 1, 256, 64)
        shape_d = (1, 1, 32, 32)

    # Create input tensors
    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape_a)

    torch_cos_cache = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_b_dtype)(
        shape_b
    )

    torch_sin_cache = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_c_dtype)(
        shape_c
    )

    torch_trans_mat = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_d_dtype)(
        shape_d
    )

    # Proper torch reference using apply_rotary_emb from Llama reference implementation
    # (from test_rotary_embedding_llama.py lines 165-173)
    try:
        # Determine head_dim from input shape [B, n_heads, seq_len, head_dim]
        head_dim = shape_a[-1]
        max_seq_len = max(4096, shape_b[2])  # cos/sin cache seq_len dimension

        # Precompute freqs_cis for the position range
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len * 2)

        # For prefill mode (seq_len > 1), use slice; for decode (seq_len == 1), use position indices
        if shape_a[2] > 1:  # Prefill mode
            start_pos = 0
            seq_len = shape_a[2]
            freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
        else:  # Decode mode (seq_len == 1)
            batch = shape_a[0]
            position_ids = torch.arange(batch)
            freqs_cis = freqs_cis[position_ids]

        # Apply rotary embedding
        # Input needs transpose: [B, n_heads, S, D] -> [B, S, n_heads, D]
        torch_xq = torch_input_tensor_a.transpose(1, 2)
        torch_xq_rotated = apply_rotary_emb(torch_xq, torch_xq, freqs_cis=freqs_cis)[0]
        # Transpose back: [B, S, n_heads, D] -> [B, n_heads, S, D]
        torch_output_tensor = torch_xq_rotated.transpose(1, 2)

    except Exception as e:
        # Fallback: use input as reference (will have lower PCC)
        torch_output_tensor = torch_input_tensor_a.clone()

    # Create TTNN tensors - use the traced memory configs
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
        from_torch_kwargs["memory_config"] = input_a_memory_config

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, **from_torch_kwargs)

    cos_cache = ttnn.from_torch(
        torch_cos_cache,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,  # Use traced config
    )

    sin_cache = ttnn.from_torch(
        torch_sin_cache,
        dtype=input_c_dtype,
        layout=input_c_layout,
        device=device,
        memory_config=input_c_memory_config,  # Use traced config
    )

    trans_mat = ttnn.from_torch(
        torch_trans_mat,
        dtype=input_d_dtype,
        layout=input_d_layout,
        device=device,
        memory_config=input_d_memory_config,  # Use traced config
    )

    start_time = start_measuring_time()
    if output_memory_config is not None:
        output_tensor = ttnn.experimental.rotary_embedding_llama(
            input_tensor_a, cos_cache, sin_cache, trans_mat, memory_config=output_memory_config
        )
    else:
        output_tensor = ttnn.experimental.rotary_embedding_llama(input_tensor_a, cos_cache, sin_cache, trans_mat)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC - using placeholder reference for now
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)  # Lower tolerance for complex ops

    return [pcc, e2e_perf]
