# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp


@pytest.mark.parametrize(
    "testing_dtype",
    ["bfloat16", "float32"],
)
def test_fmod_nan(testing_dtype, device):
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)
    if testing_dtype == "bfloat16":
        pytest.xfail("NaN is packed as inf for ttnn.bfloat16")

    torch_input_a = torch.tensor([1.0, 0.0, -1.0], dtype=torch_dtype)
    torch_input_b = torch.tensor([0.0, 0.0, 0.0], dtype=torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn.fmod)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_in_a = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_in_b = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.fmod(tt_in_a, tt_in_b)
    output_tensor = ttnn.to_torch(tt_result)

    assert torch.equal(torch.isnan(golden), torch.isnan(output_tensor))


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_fmod_binary_accuracy(device, dtype):
    """Test fmod binary operation with specific values."""
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    # Test various cases: positive/negative, small/large values
    # Use divisors that are powers of 2 or simple fractions for better reciprocal precision
    torch_input_a = torch.tensor([[5.0, 7.0, -5.0, -7.0, 3.5, 10.0, 1.5, -1.5, 9.0, 15.0]], dtype=torch_dtype)
    torch_input_b = torch.tensor([[2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 0.5, 0.5, -2.0, -4.0]], dtype=torch_dtype)

    golden_fn = ttnn.get_golden_function(ttnn.fmod)
    golden = golden_fn(torch_input_a, torch_input_b, device=device)

    input_tensor_a = ttnn.from_torch(torch_input_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.fmod(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    if dtype == "bfloat16":
        assert_with_ulp(golden, output, 2)
    else:
        assert_with_ulp(golden, output, 8)
