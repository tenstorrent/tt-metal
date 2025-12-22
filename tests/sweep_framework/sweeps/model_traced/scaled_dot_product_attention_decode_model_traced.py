# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.master_config_loader import MasterConfigLoader

TIMEOUT = 60

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("transformer::scaled_dot_product_attention_decode", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 8, 1, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_c_dtype": [ttnn.bfloat16],
        "input_c_layout": [ttnn.TILE_LAYOUT],
        "input_c_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_d_dtype": [ttnn.bfloat16],
        "input_d_layout": [ttnn.TILE_LAYOUT],
        "input_d_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_e_dtype": [ttnn.bfloat16],
        "input_e_layout": [ttnn.TILE_LAYOUT],
        "input_e_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

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
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,  # Accept any extra parameters (like input_d_*)
) -> list:
    torch.manual_seed(0)

    # Handle both sample suite (tuple/list) and model_traced suite (dict with keys for multi-input ops)
    if isinstance(input_shape, dict):
        # Multi-input operation - extract individual shapes
        shape_q = tuple(input_shape.get("input_a", input_shape.get("self", (1, 8, 1, 64))))
    else:
        # Convert list to tuple if needed
        shape_q = tuple(input_shape) if isinstance(input_shape, list) else input_shape

    # Tensor creation
    torch_q = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype)(shape_q)
    torch_k = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_b_dtype or input_a_dtype
    )(shape_q)
    torch_v = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_c_dtype or input_a_dtype
    )(shape_q)
    torch_output = torch.nn.functional.scaled_dot_product_attention(
        torch_q.to(torch.bfloat16),
        torch_k.to(torch.bfloat16),
        torch_v.to(torch.bfloat16),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )

    q_tensor = ttnn.from_torch(
        torch_q, dtype=input_a_dtype, layout=input_a_layout, device=device, memory_config=input_a_memory_config
    )
    k_tensor = ttnn.from_torch(
        torch_k,
        dtype=input_b_dtype or input_a_dtype,
        layout=input_b_layout or input_a_layout,
        device=device,
        memory_config=input_b_memory_config or input_a_memory_config,
    )
    v_tensor = ttnn.from_torch(
        torch_v,
        dtype=input_c_dtype or input_a_dtype,
        layout=input_c_layout or input_a_layout,
        device=device,
        memory_config=input_c_memory_config or input_a_memory_config,
    )

    # Op call
    start_time = start_measuring_time()
    output_tensor = ttnn.transformer.scaled_dot_product_attention_decode(
        q_tensor, k_tensor, v_tensor, is_causal=False, memory_config=output_memory_config or input_a_memory_config
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Comparison
    return [check_with_pcc(torch_output, output_tensor, 0.99), e2e_perf]
