# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import (
    assert_with_pcc,
    assert_with_ulp,
    assert_allclose,
    generate_all_bfloat16_bitpatterns,
    flush_subnormal_values_to_zero,
)

pytestmark = pytest.mark.use_module_device


def test_hardswish_exhaustive_bfloat16(device):
    """Test hardswish over all representable bfloat16 values using PCC comparison."""
    input_tensor = generate_all_bfloat16_bitpatterns(torch.bfloat16).flatten()
    input_tensor = flush_subnormal_values_to_zero(input_tensor)
    input_f32 = input_tensor.to(torch.float32)

    # Filter to finite values only
    mask = torch.isfinite(input_f32)
    input_tensor = input_tensor[mask]

    # Pre-compute golden and keep only positions where output is finite
    golden_check = torch.nn.functional.hardswish(input_tensor.to(torch.float32)).to(torch.bfloat16)
    finite_mask = torch.isfinite(golden_check)
    input_tensor = input_tensor[finite_mask]

    # Pad to tile-aligned length (multiple of 32)
    numel = input_tensor.numel()
    pad = (32 - numel % 32) % 32
    if pad > 0:
        input_tensor = torch.cat([input_tensor, torch.zeros(pad, dtype=torch.bfloat16)])

    golden = torch.nn.functional.hardswish(input_tensor.to(torch.float32)).to(torch.bfloat16)

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.hardswish(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_pcc(golden, result, 0.999)


def test_hardswish_ulp_bfloat16(device):
    """Test hardswish with ULP comparison in the active region [-3, 3]."""
    # Use linspace in [-10, 10] to cover saturation and transition regions
    torch_input = torch.linspace(-10, 10, 32 * 256, dtype=torch.bfloat16)

    golden = torch.nn.functional.hardswish(torch_input.to(torch.float32)).to(torch.bfloat16)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.hardswish(tt_in)
    result = ttnn.to_torch(tt_result).to(torch.bfloat16)

    assert_with_ulp(golden, result, ulp_threshold=2)


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 2, 64, 128]),
        torch.Size([1, 3, 320, 320]),
    ],
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-10, -3),
        (-3, 3),
        (3, 10),
        (-100, 100),
    ],
)
def test_hardswish_allclose(input_shape, low, high, device):
    """Test hardswish accuracy with allclose across different input regions."""
    num_elements = torch.prod(torch.tensor(input_shape)).item()
    torch_input = torch.linspace(low, high, num_elements, dtype=torch.bfloat16).reshape(input_shape)

    golden = torch.nn.functional.hardswish(torch_input.to(torch.float32)).to(torch.bfloat16)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.hardswish(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_allclose(result, golden, rtol=1.6e-2, atol=1e-2)


def test_hardswish_pcc(device):
    """Test hardswish with PCC correlation check."""
    torch.manual_seed(0)
    torch_input = torch.randn((64, 128), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.hardswish)
    golden = golden_function(torch_input)

    tt_in = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    tt_result = ttnn.hardswish(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_pcc(golden, result, 0.999)
