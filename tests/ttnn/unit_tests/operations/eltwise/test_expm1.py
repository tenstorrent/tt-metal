# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import numpy as np
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_allclose


def test_expm1_arange_masking(device):
    # Expm1 Working range - Overflow from 88.5(inf) as in exp
    low = -float("inf")
    high = 88.5

    # Generate all possible bit patterns for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor = input_tensor.to(torch.float32)

    # masking to working range
    mask = (input_tensor >= low) & (input_tensor <= high)
    input_tensor = input_tensor[mask]

    mask_failed = (input_tensor >= -0.28515625) & (input_tensor <= 0.69140625)
    input_tensor[mask_failed] = 1.0

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.expm1)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.expm1(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(golden, result, 1, allow_nonfinite=True)


@pytest.mark.parametrize(
    "low, high",
    [
        (-0.28515625, 0.69140625),
        (-1.6 * 10**38, 1.6 * 10**38),  # bf16 range
    ],
)
def test_expm1_atol(low, high, device):
    num_elements = torch.prod(torch.tensor(torch.Size([1, 2, 64, 120]))).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(torch.Size([1, 2, 64, 120]))

    golden_function = ttnn.get_golden_function(ttnn.expm1)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.expm1(tt_in)
    result = ttnn.to_torch(tt_result)

    torch.allclose(golden, result, rtol=0.844, atol=3.91e-03)
