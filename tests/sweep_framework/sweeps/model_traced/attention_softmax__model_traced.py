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
    scalar=None,
    *,
    device,
    **kwargs,
) -> list:
    """
    attention_softmax_: in-place attention softmax with mask

    Based on working transformer sweep test implementation.
    Key difference from unit tests: uses binary mask (0/1) instead of causal_mask parameter.
    """
    torch.manual_seed(0)

    # Parse input_shape - single tensor input
    if isinstance(input_shape, (tuple, list)):
        shape_a = tuple(input_shape)
    else:
        shape_a = input_shape

    # Get head_size from scalar if provided (as traced configs do)
    head_size = scalar if scalar is not None else None

    # Generate input tensor
    torch_input_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    # attention_softmax_ requires an attention_mask
    # Use binary mask (0 or 1) as in the working transformer sweep test
    # NOT -inf masks as in some unit tests
    torch_mask_tensor = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32),
        input_b_dtype if input_b_dtype else input_a_dtype,
    )(shape_a)
    # Convert to binary mask: values > 0 become 1, else 0
    torch_mask_tensor = (torch_mask_tensor > 0).to(torch.float32)

    # Get golden output using the ttnn golden function
    golden_function = ttnn.get_golden_function(ttnn.transformer.attention_softmax_)
    # Clone input as operation is in-place
    tmp_input = torch.clone(torch_input_tensor)
    torch_output_tensor = golden_function(tmp_input, head_size=head_size, attention_mask=torch_mask_tensor)

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

    mask_tensor = ttnn.from_torch(
        torch_mask_tensor,
        dtype=mask_dtype,
        layout=mask_layout,
        device=device,
        memory_config=mask_memory_config,
    )

    # Run operation (in-place operation modifies input)
    # Do NOT use causal_mask parameter - use the binary mask instead
    start_time = start_measuring_time()
    result = ttnn.transformer.attention_softmax_(input_tensor, head_size=head_size, attention_mask=mask_tensor)
    output_tensor = ttnn.to_torch(result)
    e2e_perf = stop_measuring_time(start_time)

    # Check PCC - using 0.999 as in transformer sweep test
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    # Return result in the format expected by sweeps_runner
    return [pcc, e2e_perf]
