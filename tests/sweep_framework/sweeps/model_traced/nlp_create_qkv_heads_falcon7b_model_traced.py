# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("experimental::nlp_create_qkv_heads_falcon7b", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with Falcon-7B specific shape
    "model_traced_sample": {
        "input_shape": [(1, 1, 32, 4672)],  # Falcon-7B uses 4672 hidden dim
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
    output_memory_config,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Handle tuple input_shape
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    batch, _, seq_len, hidden_dim = shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # PyTorch reference: Falcon-7B splits into Q (4544), K (64), V (64)
    (ref_q, ref_k, ref_v) = torch.split(torch_input_tensor_a, [4544, 64, 64], dim=-1)
    # Additional shuffling for Q head (71 heads x 64 dims)
    ref_q = torch.reshape(ref_q, [batch, seq_len, 71, 64]).transpose(-3, -2)

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

    start_time = start_measuring_time()
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_falcon7b(input_tensor_a, memory_config=output_memory_config)
    q_torch = ttnn.to_torch(q)
    k_torch = ttnn.to_torch(k)
    v_torch = ttnn.to_torch(v)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC for all three outputs
    pcc_q = check_with_pcc(ref_q, q_torch, 0.999)
    pcc_k = check_with_pcc(ref_k, k_torch, 0.999)
    pcc_v = check_with_pcc(ref_v, v_torch, 0.999)

    # All three must pass
    pcc = pcc_q and pcc_k and pcc_v

    return [pcc, e2e_perf]
