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

TIMEOUT = 30

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("transformer::chunked_scaled_dot_product_attention", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 8, 32, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
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

    shape_q = input_shape if isinstance(input_shape, (tuple, list)) else (1, 8, 32, 64)

    # Tensor creation
    torch_q = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype)(shape_q)
    torch_k = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_b_dtype or input_a_dtype
    )(shape_q)
    torch_v = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_c_dtype or input_a_dtype
    )(shape_q)
    torch_output = torch.nn.functional.scaled_dot_product_attention(
        torch_q, torch_k, torch_v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    torch_page_table = torch.arange(
        0, int(shape_q[0]) * max(1, (int(shape_q[2]) + 63) // 64) * 64, 64, dtype=torch.int32
    ).reshape(int(shape_q[0]), -1)

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
    page_table_tensor = ttnn.from_torch(
        torch_page_table,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_a_memory_config,
    )

    # Op call
    start_time = start_measuring_time()
    output_tensor = ttnn.transformer.chunked_scaled_dot_product_attention(
        q_tensor, k_tensor, v_tensor, page_table_tensor, 0, memory_config=output_memory_config or input_a_memory_config
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Comparison
    return [check_with_pcc(torch_output, output_tensor, 0.999), e2e_perf]
