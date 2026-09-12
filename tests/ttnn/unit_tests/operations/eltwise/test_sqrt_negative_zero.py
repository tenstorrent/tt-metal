# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""sqrt(-0.0) is -0.0 and rsqrt(-0.0) is -inf.

The domain guard compares with a sign-bit test, so negative zero landed with the
genuinely negative values and came back as NaN. IEEE-754 and torch both put -0.0
in the domain.
"""

import math

import pytest
import torch
import ttnn

SHAPE = (1, 1, 32, 32)


def _t(v, dtype, device):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    return ttnn.from_torch(
        torch.full(SHAPE, v, dtype=torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )


def test_sqrt_of_negative_zero_is_negative_zero(device):
    got = ttnn.to_torch(ttnn.sqrt(_t(-0.0, ttnn.float32, device))).flatten()[0].item()
    assert got == 0.0 and math.copysign(1.0, got) < 0, f"expected -0.0, got {got}"


def test_rsqrt_of_negative_zero_is_negative_infinity(device):
    for dtype in (ttnn.float32, ttnn.bfloat16):
        got = ttnn.to_torch(ttnn.rsqrt(_t(-0.0, dtype, device))).float().flatten()[0].item()
        assert got == float("-inf"), f"{dtype}: expected -inf, got {got}"


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("value, sqrt_want, rsqrt_want", [(4.0, 2.0, 0.5), (0.0, 0.0, float("inf"))])
def test_the_domain_is_unchanged(device, dtype, value, sqrt_want, rsqrt_want):
    assert ttnn.to_torch(ttnn.sqrt(_t(value, dtype, device))).float().flatten()[0].item() == sqrt_want
    assert ttnn.to_torch(ttnn.rsqrt(_t(value, dtype, device))).float().flatten()[0].item() == rsqrt_want


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_genuinely_negative_input_still_leaves_the_domain(device, dtype):
    got = ttnn.to_torch(ttnn.sqrt(_t(-4.0, dtype, device))).float().flatten()[0].item()
    # bfloat16 Dest cannot carry NaN and returns an infinity instead.
    assert got != got or math.isinf(got), f"expected NaN or inf, got {got}"
