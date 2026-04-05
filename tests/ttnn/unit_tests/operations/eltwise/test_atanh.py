# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_atanh_bfloat16(device, h, w):
    """Test atanh with bfloat16 inputs in the valid range (-1, 1)."""
    torch.manual_seed(42)
    # Generate random values in (-0.9, 0.9) to stay within valid domain
    torch_input = torch.rand((h, w), dtype=torch.bfloat16) * 1.8 - 0.9
    torch_output = torch.atanh(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.atanh(tt_input)
    tt_result = ttnn.to_torch(tt_output)

    assert torch.allclose(torch_output.float(), tt_result.float(), rtol=1.6e-2, atol=1e-2, equal_nan=True), (
        f"atanh bfloat16 test failed: "
        f"max abs diff = {(torch_output.float() - tt_result.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_atanh_fp32(device, h, w):
    """Test atanh with fp32 inputs in the valid range (-1, 1)."""
    torch.manual_seed(42)
    # Generate random values in (-0.9, 0.9) to stay within valid domain
    torch_input = torch.rand((h, w), dtype=torch.float32) * 1.8 - 0.9
    torch_output = torch.atanh(torch_input)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.atanh(tt_input)
    tt_result = ttnn.to_torch(tt_output)

    assert torch.allclose(torch_output, tt_result, rtol=1.6e-2, atol=1e-2, equal_nan=True), (
        f"atanh fp32 test failed: " f"max abs diff = {(torch_output - tt_result).abs().max().item()}"
    )


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_atanh_zero(device, h, w):
    """Test atanh(0) = 0."""
    torch_input = torch.zeros((h, w), dtype=torch.bfloat16)
    torch_output = torch.atanh(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.atanh(tt_input)
    tt_result = ttnn.to_torch(tt_output)

    assert torch.allclose(torch_output.float(), tt_result.float(), rtol=1e-3, atol=1e-3, equal_nan=True), (
        f"atanh zero test failed: " f"max abs diff = {(torch_output.float() - tt_result.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_atanh_small_values(device, h, w):
    """Test atanh with small values where atanh(x) ~ x."""
    torch.manual_seed(123)
    torch_input = torch.rand((h, w), dtype=torch.bfloat16) * 0.2 - 0.1  # range [-0.1, 0.1]
    torch_output = torch.atanh(torch_input.float()).to(torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.atanh(tt_input)
    tt_result = ttnn.to_torch(tt_output)

    assert torch.allclose(torch_output.float(), tt_result.float(), rtol=1.6e-2, atol=1e-2, equal_nan=True), (
        f"atanh small values test failed: "
        f"max abs diff = {(torch_output.float() - tt_result.float()).abs().max().item()}"
    )
