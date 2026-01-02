# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.sweep_framework.master_config_loader import MasterConfigLoader
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("transformer::attention_softmax_", all_cases=False)

# Parameters provided to the test vector generator are defined here.
parameters = {}

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
    output_memory_config=None,
    scalar=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Parse input_shape - single tensor input
    if isinstance(input_shape, (tuple, list)):
        shape_a = tuple(input_shape)
    else:
        shape_a = input_shape

    # Generate input tensor
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    # attention_softmax_ requires an attention_mask
    # Create a causal mask for the sequence
    # Input shape is typically [batch, num_heads, seq_len, seq_len]
    if len(shape_a) >= 3:
        seq_len = shape_a[-1]
        # Create causal mask: upper triangular matrix with -inf
        torch_attention_mask = torch.zeros(shape_a, dtype=torch.float32)
        for i in range(seq_len):
            torch_attention_mask[..., i, i + 1 :] = float("-inf")
    else:
        # Fallback: create a mask of zeros (no masking)
        torch_attention_mask = torch.zeros(shape_a, dtype=torch.float32)

    # Get golden output
    golden_function = ttnn.get_golden_function(ttnn.transformer.attention_softmax_)
    torch_output_tensor = golden_function(
        torch_input_tensor,
        head_size=None,
        attention_mask=torch_attention_mask,
    )

    # Convert to TTNN tensors
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    # Convert attention mask to TTNN tensor
    mask_dtype = input_b_dtype if input_b_dtype else input_a_dtype
    mask_layout = input_b_layout if input_b_layout else input_a_layout
    mask_memory_config = input_b_memory_config if input_b_memory_config else input_a_memory_config

    attention_mask_tensor = ttnn.from_torch(
        torch_attention_mask,
        dtype=mask_dtype,
        layout=mask_layout,
        device=device,
        memory_config=mask_memory_config,
    )

    # Run operation (in-place operation modifies input)
    start_time = start_measuring_time()
    output_tensor = ttnn.transformer.attention_softmax_(
        input_tensor,
        head_size=None,
        attention_mask=attention_mask_tensor,
        causal_mask=True,
    )
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Check PCC - check_with_pcc returns (bool, pcc_value) tuple
    pcc_result = check_with_pcc(torch_output_tensor, output_tensor, 0.996)

    # Return result in the format expected by sweeps_runner: [(status, message), e2e_perf]
    return [pcc_result, e2e_perf]
