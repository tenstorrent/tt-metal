# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Regression test for ttnn.remainder with large dividend/divisor ratios.

The old kernel produced wrong results (off by whole multiples of the divisor)
when |dividend/divisor| exceeded ~2^23, because:
1. The float multiply-by-reciprocal quotient estimation rounds imprecisely
2. The subsequent quotient * divisor subtraction introduced a second rounding
3. The cleanup loop only corrected in one direction

Resolves #54048.
"""

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "dividend,divisor",
    [
        # Specific cases from the issue
        (1.42606e7, 3.0),
        (5.70425e7, 3.0),
        # Ratios at and beyond 2^23
        (2**22 * 3.0 + 2.0, 3.0),
        (2**23 * 3.0 + 1.0, 3.0),
        (2**24 * 3.0 + 2.0, 3.0),
        (2**25 * 7.0 + 3.0, 7.0),
        (2**26 * 3.0 + 1.0, 3.0),
        # Negative dividends
        (-5.70425e7, 3.0),
        (-2**24 * 3.0 - 1.0, 3.0),
        # Negative divisors
        (5.70425e7, -3.0),
    ],
)
def test_remainder_large_ratio(device, dividend, divisor):
    """Verify remainder is correct for large dividend/divisor ratios."""
    torch_input = torch.tensor([[dividend]], dtype=torch.float32)
    torch_expected = torch.remainder(torch_input, divisor)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.to_torch(ttnn.remainder(input_tensor, divisor))

    assert torch.allclose(output_tensor, torch_expected, atol=1e-5, rtol=1e-5), (
        f"remainder({dividend}, {divisor}): got {output_tensor.item()}, "
        f"expected {torch_expected.item()}"
    )


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_remainder_sweep_large_ratios(device, h, w):
    """Sweep test with dividends producing ratios from 2^20 to 2^26."""
    torch.manual_seed(42)

    divisor = 3.0
    ratios = torch.empty((h, w), dtype=torch.float32).uniform_(2**20, 2**26)
    torch_input = ratios * divisor + torch.empty((h, w)).uniform_(0, divisor)
    torch_expected = torch.remainder(torch_input, divisor)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.to_torch(ttnn.remainder(input_tensor, divisor))

    assert torch.allclose(output_tensor, torch_expected, atol=1e-4, rtol=1e-4), (
        f"Max error: {(output_tensor - torch_expected).abs().max().item()}"
    )


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_remainder_normal_range_no_regression(device, h, w):
    """Verify normal-range remainder still works (no regression)."""
    torch.manual_seed(0)

    divisor = 7.0
    torch_input = torch.empty((h, w), dtype=torch.float32).uniform_(-100, 100)
    torch_expected = torch.remainder(torch_input, divisor)

    input_tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.to_torch(ttnn.remainder(input_tensor, divisor))

    assert torch.allclose(output_tensor, torch_expected, atol=1e-5, rtol=1e-5), (
        f"Max error: {(output_tensor - torch_expected).abs().max().item()}"
    )
