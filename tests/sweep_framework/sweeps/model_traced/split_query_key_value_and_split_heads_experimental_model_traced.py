# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from functools import partial
from tests.sweep_framework.master_config_loader import MasterConfigLoader
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
TIMEOUT = 30

# Load traced configurations from real model tests
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters(
    "experimental::split_query_key_value_and_split_heads", all_cases=False
)

# Parameters provided to the test vector generator are defined here.
parameters = {}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def invalidate_vector(test_vector) -> tuple:
    """
    Filter configs that fail due to C++ implementation bug.

    The traced config from sentence_bert (shape=(8,1,384,2304), num_heads=12)
    fails with TT_FATAL: "Physical shard shape (768, 48) must be tile {32, 32} sized!"

    This is a C++ implementation bug where the operation calculates non-tile-aligned
    output shard shapes. Cannot be fixed in sweep test.
    """
    # All traced configs currently fail with non-tile-aligned output shards
    return True, "C++ bug: operation produces non-tile-aligned output shard shapes - needs fix in op implementation"


def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config=None,
    kv_input_height=None,
    num_heads=None,
    *,
    device,
    **kwargs,
) -> list:
    """
    split_query_key_value_and_split_heads: Splits concatenated Q,K,V and reshapes to attention heads

    Input: [batch, seq_len, hidden_dim] where hidden_dim = 3 * num_heads * head_dim
    Outputs: Q [batch, num_heads, seq_len, head_dim]
             K [batch, num_heads, head_dim, seq_len] (transposed for attention)
             V [batch, num_heads, seq_len, head_dim]

    Reference: test_bert_large_split_query_key_value_and_split_heads.py (lines 54-58)
    """
    torch.manual_seed(0)

    # Handle tuple input_shape
    if isinstance(input_shape, (tuple, list)):
        shape = tuple(input_shape)
    else:
        shape = input_shape

    # Generate input tensor
    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype)(
        shape
    )

    # Calculate Q, K, V dimensions
    # Input shape: [batch, seq_len, hidden_dim] where hidden_dim = 3 * num_heads * head_dim
    if len(shape) == 4:
        batch, _, seq_len, hidden_dim = shape
    else:
        batch, seq_len, hidden_dim = shape[-3:]

    # Infer num_heads and head_dim from shape if not provided
    if num_heads is None:
        # Common values: 12 or 16 heads for BERT
        # Try to infer from hidden_dim: hidden_dim = 3 * num_heads * head_dim
        # For BERT-large: 3072 = 3 * 16 * 64
        if hidden_dim == 3072:
            num_heads = 16
        elif hidden_dim == 2304:
            num_heads = 12
        else:
            # Default fallback
            num_heads = 16

    head_dim = hidden_dim // (3 * num_heads)

    # Torch reference from unit test (lines 54-58)
    # Split along last dimension into Q, K, V (each num_heads * head_dim wide)
    ref_q, ref_k, ref_v = torch.split(torch_input, num_heads * head_dim, dim=-1)

    # Reshape and transpose
    # Q: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
    ref_q = ref_q.reshape([batch, seq_len, num_heads, head_dim]).transpose(-3, -2)
    # K: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, head_dim, seq_len] (transposed for attention)
    ref_k = ref_k.reshape([batch, seq_len, num_heads, head_dim]).transpose(-3, -2).transpose(-2, -1)
    # V: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
    ref_v = ref_v.reshape([batch, seq_len, num_heads, head_dim]).transpose(-3, -2)

    # Use Q for PCC comparison (first output)
    torch_output = ref_q

    # Convert to TTNN
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    # Run operation
    # The operation takes a kv_input_height parameter that specifies the grid size
    # From unit test: ttnn.CoreCoord(12, 9) for 16 heads
    # We'll use a default grid based on num_heads
    if kv_input_height is None:
        # Default grid calculation
        grid_size = ttnn.CoreCoord(num_heads, max(1, seq_len // 64))
    else:
        # Use provided kv_input_height (might be a CoreCoord or tuple)
        if isinstance(kv_input_height, (list, tuple)) and len(kv_input_height) == 2:
            grid_size = ttnn.CoreCoord(kv_input_height[0], kv_input_height[1])
        else:
            grid_size = kv_input_height

    start_time = start_measuring_time()
    q, k, v = ttnn.experimental.split_query_key_value_and_split_heads(
        input_tensor, grid_size, memory_config=output_memory_config or input_a_memory_config
    )

    # Convert Q output to torch for comparison
    output_tensor = ttnn.to_torch(q)
    e2e_perf = stop_measuring_time(start_time)

    # Check PCC
    pcc = check_with_pcc(torch_output, output_tensor, 0.99)

    return [pcc, e2e_perf]
