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
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("experimental::nlp_concat_heads", all_cases=False)

parameters = {
    "model_traced_sample": {
        "input_shape": [(1, 12, 32, 64)],  # Batch, heads, seq, head_dim
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape)

    # nlp_concat_heads concatenates heads: [B, H, S, D] -> [B, 1, S, H*D]
    # So we need to compute the expected output shape
    if len(shape) == 4:
        batch, num_heads, seq_len, head_dim = shape
        expected_output_shape = (batch, 1, seq_len, num_heads * head_dim)
        # Reshape input to match expected output for comparison
        torch_output_tensor = (
            torch_input_tensor_a.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch, seq_len, num_heads * head_dim)
            .unsqueeze(1)
        )
    else:
        # Fallback: just clone the input
        torch_output_tensor = torch_input_tensor_a.clone()

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.experimental.nlp_concat_heads(input_tensor_a, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)
    return [pcc, e2e_perf]
