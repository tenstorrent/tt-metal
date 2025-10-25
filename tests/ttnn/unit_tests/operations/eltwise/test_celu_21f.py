# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import ttnn
import math
from tests.ttnn.utils_for_testing import assert_with_ulp


def test_celu_arange(device):
    # Generate all possible bit patterns for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor = input_tensor.to(torch.float32)

    # Mask NaN, special values where celu has ULP>1 (Covered in allclose test below).
    mask = (torch.isnan(input_tensor)) | ((input_tensor >= -0.28515625) & (input_tensor <= 1.1663108012064884e-38))
    input_tensor[mask] = 1.0

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.celu)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.celu(tt_in)
    result = ttnn.to_torch(tt_result)
    assert_with_ulp(golden, result, 1, allow_nonfinite=True)


@pytest.mark.parametrize(
    "low, high, expected_Atol, expected_rtol",
    [
        (-1.6 * 10**38, -0.28515625, 0.001, 0.004),
        (-0.28515625, 1.1663108012064884e-38, 0.002, 0.02),
        (1.1663108012064884e-38, 1.6 * 10**38, 1e-6, 1e-6),
    ],
)
def test_celu_allclose(low, high, expected_Atol, expected_rtol, device):
    num_elements = math.prod([1, 3, 320, 320])
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(torch.Size([1, 3, 320, 320]))

    golden_function = ttnn.get_golden_function(ttnn.celu)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.celu(tt_in)
    result = ttnn.to_torch(tt_result)
    assert torch.allclose(golden, result, atol=expected_Atol, rtol=expected_rtol)
