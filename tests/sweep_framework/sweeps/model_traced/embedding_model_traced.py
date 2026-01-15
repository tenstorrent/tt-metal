# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Tuple
from functools import partial

import random
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("embedding", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "embedding_args": [(1, 32, 32, 128)],  # batch_size, seq_length, embeddings_dim, num_embeddings
        "input_dtype": [ttnn.uint32],
        "weight_dtype": [ttnn.bfloat16],
        "output_dtype": [ttnn.bfloat16],
        "input_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "weight_layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "weight_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],  # Sample uses device
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    # For embedding, check the input_a_layout (indices) and input_b_layout (weights)
    if test_vector.get("input_a_layout") == ttnn.TILE_LAYOUT:
        return True, "Input indices must be in row major layout"
    if test_vector.get("input_b_layout") == ttnn.TILE_LAYOUT:
        return True, "Weights must be in row major layout"
    if test_vector.get("output_dtype") == ttnn.bfloat8_b:
        return True, "bfloat8_b is not supported for output tensor"
    if (
        test_vector.get("input_b_layout") == ttnn.ROW_MAJOR_LAYOUT
        and test_vector.get("input_b_dtype") == ttnn.bfloat8_b
    ):
        return True, "bfloat8_b is only supported on tiled layout"
    return False, None


# Only add model_traced suite if it has valid configurations
if model_traced_params and any(len(v) > 0 for v in model_traced_params.values() if isinstance(v, list)):
    parameters["model_traced"] = model_traced_params


def run(
    embedding_args,
    input_dtype,
    weight_dtype,
    output_dtype,
    input_layout,
    weight_layout,
    input_memory_config,
    weight_memory_config,
    output_memory_config,
    *,
    device,
    **kwargs,  # Accept traced_source, traced_machine_info, etc.
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    batch_size, seq_length, embeddings_dim, num_embeddings = embedding_args

    input_shape = (batch_size, seq_length)
    weight_shape = (num_embeddings, embeddings_dim)

    torch_input_tensor = torch_random(input_shape, 0, num_embeddings, torch.int64)
    torch_weight_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), weight_dtype
    )(weight_shape)

    golden_function = ttnn.get_golden_function(ttnn.embedding)
    torch_output_tensor = golden_function(torch_input_tensor, torch_weight_tensor).squeeze()
    # torch_output_tensor = torch.nn.functional.embedding(torch_input_tensor, torch_weight_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_memory_config,
    )
    weight_tensor = ttnn.from_torch(
        torch_weight_tensor,
        dtype=weight_dtype,
        layout=weight_layout,
        device=device,
        memory_config=weight_memory_config,
    )

    start_time = start_measuring_time()
    output_tensor = ttnn.embedding(input_tensor, weight_tensor, dtype=output_dtype, memory_config=output_memory_config)
    e2e_perf = stop_measuring_time(start_time)

    output_tensor = ttnn.to_torch(output_tensor).squeeze()

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
