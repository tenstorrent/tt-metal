# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp


def flush_subnormal_values(tensor):
    # f32 and bf16 numbers are subnormals if exponent == 0
    # i.e. if  their value < 2**(-126)
    SUBNORMAL_THRESHOLD = 2.0 ** (-126)
    mask = torch.abs(tensor) < SUBNORMAL_THRESHOLD
    tensor[mask] = 0.0
    return tensor


@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
    ],
)
def test_expm1_all_bitpatterns(dtype, device):
    torch_dtype = getattr(torch, dtype)
    tt_dtype = getattr(ttnn, dtype)

    # Generate all possible 16-bit patterns and
    # For dtype == bfloat16, this covers all possible inputs
    # For dtype == float32, this only covers some inputs
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor = input_tensor.to(torch_dtype)  # bfloat16 -> torch_dtype

    # If input is subnormal then we assume hardware will flush it to 0.0
    input_tensor = flush_subnormal_values(input_tensor)

    # BF16 NaN values are packed as infinity during device transfer, so they are not
    # valid inputs for checking expm1's NaN propagation. The float32 parametrisation
    # covers NaN propagation using the same generated bit patterns.
    if tt_dtype == ttnn.bfloat16:
        input_tensor = input_tensor[~torch.isnan(input_tensor)]

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=tt_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.expm1)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.expm1(tt_in)
    result = ttnn.to_torch(tt_result)

    # If expected output is subnormal then its calculated value should be 0.0 (hardware assumed to flush to 0.0)
    result = flush_subnormal_values(result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)
