# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device

# Regression coverage for #54048: the SFPU remainder path estimates the
# quotient via multiply-by-reciprocal, whose few-ulp relative error exceeds a
# full divisor multiple once |quotient| approaches 2^23 and beyond. The
# normalization loop must walk the residual in BOTH directions so the final
# value lands in [0, s) for every sign combination.

SIGN_MATRIX = [(1.0, 3.0), (-1.0, 3.0), (1.0, -3.0), (-1.0, -3.0)]


@pytest.mark.parametrize(
    "dividend, divisor",
    [
        # Reported failing magnitudes from the issue (positive/positive regime)
        (5.70425e7, 3.0),
        (1.42606e7, 3.0),
    ],
)
def test_remainder_large_ratio_reported(device, dividend, divisor):
    torch_input = torch.tensor([[dividend]], dtype=torch.float32)
    expected = torch.remainder(torch_input, divisor)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.remainder(input_tensor, divisor))

    assert torch.equal(actual, expected), (
        f"remainder({dividend}, {divisor}) produced {actual.item()}, expected {expected.item()}"
    )


@pytest.mark.parametrize("sign_d, sign_s", SIGN_MATRIX)
@pytest.mark.parametrize(
    "mag, divisor",
    [
        (5.70425e7, 3.0),
        (1.42606e7, 3.0),
    ],
)
def test_remainder_large_ratio_sign_matrix(device, mag, divisor, sign_d, sign_s):
    dividend = mag * sign_d
    div = divisor * sign_s
    torch_input = torch.tensor([[dividend]], dtype=torch.float32)
    expected = torch.remainder(torch_input, div)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.remainder(input_tensor, div))

    assert torch.equal(actual, expected), (
        f"remainder({dividend}, {div}) produced {actual.item()}, expected {expected.item()}"
    )


@pytest.mark.parametrize("divisor", [3.0, 0.75, -7.25])
def test_remainder_large_ratio_boundary_sweep(device, divisor):
    # One ulp above exact multiples of the divisor across the quotient range
    # where the reciprocal-based estimate error crosses one, then several,
    # divisor multiples (2^22 .. 2^26).
    quotients = torch.tensor(
        [
            2**21 - 1,
            2**21,
            2**22 - 1,
            2**22,
            2**23 - 1,
            2**23,
            2**23 + 1,
            2**24,
            2**25,
            2**26,
        ],
        dtype=torch.float32,
    )
    dividends = quotients.abs() * divisor
    dividends = torch.nextafter(dividends, torch.full_like(dividends, float("inf")))
    # Exercise both dividend signs; divisor sign comes from the parametrization.
    all_inputs = torch.cat([dividends, -dividends]).reshape(1, -1)
    expected = torch.remainder(all_inputs, divisor)

    input_tensor = ttnn.from_torch(all_inputs, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.remainder(input_tensor, divisor))

    assert torch.equal(actual, expected)
