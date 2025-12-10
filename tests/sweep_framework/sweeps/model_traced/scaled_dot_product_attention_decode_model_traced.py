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

# Override the default timeout in seconds for hang detection.
TIMEOUT = 60

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("transformer::scaled_dot_product_attention_decode", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_shape": [(1, 8, 1, 64)],  # Batch, heads, decode_len=1, head_dim
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


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    input_c_dtype=None,
    input_c_layout=None,
    input_c_memory_config=None,
    input_d_dtype=None,  # Ignored - may be present for 4-input detection
    input_d_layout=None,  # Ignored
    input_d_memory_config=None,  # Ignored
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
) -> list:
    torch.manual_seed(0)

    # Handle dict input_shape from traced configurations (multi-input)
    if isinstance(input_shape, dict):
        # Traced configuration with multiple inputs (Q, K, V)
        shape_q = input_shape.get("input_a", input_shape.get("self"))
        shape_k = input_shape.get("input_b", input_shape.get("other"))
        shape_v = input_shape.get("input_c")
        if shape_v is None:
            # If only 2 inputs, use K shape for V
            shape_v = shape_k
    else:
        # Fallback for sample configurations
        if isinstance(input_shape, (tuple, list)):
            shape = tuple(input_shape)
        else:
            shape = input_shape
        shape_q = shape_k = shape_v = shape

    # Convert to tuples if needed
    if isinstance(shape_q, list):
        shape_q = tuple(shape_q)
    if isinstance(shape_k, list):
        shape_k = tuple(shape_k)
    if isinstance(shape_v, list):
        shape_v = tuple(shape_v)

    # For decode, Q typically has seq_len=1
    # Adjust shapes if needed for attention compatibility
    if isinstance(shape_q, tuple) and len(shape_q) == 4:
        # Q: [B, H, 1, D] for decode
        # K, V: [B, H, S, D] where S is cache length
        if len(shape_k) == 4 and len(shape_v) == 4:
            # Ensure head dimension matches
            if shape_q[3] != shape_k[3]:
                shape_k = (shape_k[0], shape_k[1], shape_k[2], shape_q[3])
            if shape_q[3] != shape_v[3]:
                shape_v = (shape_v[0], shape_v[1], shape_v[2], shape_q[3])
            # Handle GQA: replicate heads if needed
            if shape_q[1] != shape_k[1]:
                shape_k = (shape_k[0], shape_q[1], shape_k[2], shape_k[3])
            if shape_q[1] != shape_v[1]:
                shape_v = (shape_v[0], shape_q[1], shape_v[2], shape_v[3])

    # Use provided dtypes with defaults
    dtype_q = input_a_dtype
    dtype_k = input_b_dtype if input_b_dtype is not None else input_a_dtype
    dtype_v = input_c_dtype if input_c_dtype is not None else input_a_dtype

    layout_q = input_a_layout
    layout_k = input_b_layout if input_b_layout is not None else input_a_layout
    layout_v = input_c_layout if input_c_layout is not None else input_a_layout

    mem_config_q = input_a_memory_config
    mem_config_k = input_b_memory_config if input_b_memory_config is not None else input_a_memory_config
    mem_config_v = input_c_memory_config if input_c_memory_config is not None else input_a_memory_config

    if output_memory_config is None:
        output_memory_config = input_a_memory_config

    # Generate random tensors for Q, K, V
    torch_q = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_q)(shape_q)
    torch_k = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_k)(shape_k)
    torch_v = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_v)(shape_v)

    # PyTorch reference: scaled_dot_product_attention
    # Ensure all tensors have the same dtype for PyTorch SDPA
    torch_q_ref = torch_q.to(torch.bfloat16)
    torch_k_ref = torch_k.to(torch.bfloat16)
    torch_v_ref = torch_v.to(torch.bfloat16)
    torch_output_tensor = torch.nn.functional.scaled_dot_product_attention(
        torch_q_ref, torch_k_ref, torch_v_ref, attn_mask=None, dropout_p=0.0, is_causal=False
    )

    is_host = storage_type and "HOST" in str(storage_type)

    from_torch_kwargs_q = {"dtype": dtype_q, "layout": layout_q}
    if not is_host:
        from_torch_kwargs_q["device"] = device
        from_torch_kwargs_q["memory_config"] = mem_config_q

    q_tensor = ttnn.from_torch(torch_q, **from_torch_kwargs_q)

    from_torch_kwargs_k = {"dtype": dtype_k, "layout": layout_k}
    if not is_host:
        from_torch_kwargs_k["device"] = device
        from_torch_kwargs_k["memory_config"] = mem_config_k

    k_tensor = ttnn.from_torch(torch_k, **from_torch_kwargs_k)

    from_torch_kwargs_v = {"dtype": dtype_v, "layout": layout_v}
    if not is_host:
        from_torch_kwargs_v["device"] = device
        from_torch_kwargs_v["memory_config"] = mem_config_v

    v_tensor = ttnn.from_torch(torch_v, **from_torch_kwargs_v)

    start_time = start_measuring_time()
    # Call ttnn scaled_dot_product_attention_decode
    output_tensor = ttnn.transformer.scaled_dot_product_attention_decode(
        q_tensor, k_tensor, v_tensor, is_causal=False, memory_config=output_memory_config
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)

    return [pcc, e2e_perf]
