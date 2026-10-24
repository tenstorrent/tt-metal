# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

# std_hw, var_hw and normalize_hw across the float32 exponent range. The squared deviation
# (x - mean)^2 underflows to zero below |x - mean| = 1.0842e-19, and the sum of squares saturates or
# overflows far below the largest representable standard deviation. A composite that squares
# x - mean directly therefore returns 0, a saturated constant or inf for std_hw of an ordinary 1e-20
# or 1e20 spread, and a non-finite or wrong result for normalize_hw.
#
# Alternating columns of +v and -v give a mean of exactly zero and a population standard deviation
# of exactly v at every magnitude, so the reference is analytic: normalize_hw must return the +/-1
# pattern, std_hw must return v and var_hw must return v^2.
DEVIATIONS = (1e-25, 1e-20, 1e-19, 1e-10, 1.0, 1e10, 1e18, 1e20, 1e25)
DEVIATION_IDS = ["1e-25", "1e-20", "1e-19", "1e-10", "1.0", "1e10", "1e18", "1e20", "1e25"]

# var_hw can only be right where v^2 is itself a normal float32.
VAR_DEVIATIONS = (1e-18, 1e-10, 1.0, 1e10, 1e18, 1e19)
VAR_DEVIATION_IDS = ["1e-18", "1e-10", "1.0", "1e10", "1e18", "1e19"]

DTYPES = ((torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16))
DTYPE_IDS = ("float32", "bfloat16")

SHAPE = torch.Size([1, 1, 32, 32])


def _alternating(deviation, torch_dtype):
    """Columns of +deviation and -deviation: mean exactly 0, population std exactly |deviation|."""
    sign = torch.ones(SHAPE)
    sign[..., 1::2] = -1.0
    return sign, (sign * deviation).to(torch_dtype)


def _first(tt_tensor):
    return float(ttnn.to_torch(tt_tensor).float().flatten()[0])


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("deviation", DEVIATIONS, ids=DEVIATION_IDS)
def test_normalize_hw_is_scale_invariant(deviation, torch_dtype, ttnn_dtype, device):
    sign, in_data = _alternating(deviation, torch_dtype)
    tt_in = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    got = ttnn.to_torch(ttnn.normalize_hw(tt_in)).float()

    n_bad = int((~torch.isfinite(got)).sum())
    assert n_bad == 0, (
        f"normalize_hw with deviation {deviation:g} [{torch_dtype}] returned {n_bad} non-finite "
        f"values of {got.numel()}; every element should be +/-1"
    )
    torch.testing.assert_close(got, sign, rtol=2e-2, atol=0.0)


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("deviation", DEVIATIONS, ids=DEVIATION_IDS)
def test_std_hw_matches_the_constructed_spread(deviation, torch_dtype, ttnn_dtype, device):
    _, in_data = _alternating(deviation, torch_dtype)
    tt_in = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    got = _first(ttnn.std_hw(tt_in))
    expected = float(torch.tensor([deviation], dtype=torch_dtype)[0])

    assert got != 0.0, f"std_hw returned an exact 0 for a spread of {expected:g} [{torch_dtype}]"
    assert got == got and abs(got) != float("inf"), f"std_hw returned {got} for {expected:g}"
    message = f"std_hw returned {got:g} where the spread is exactly {expected:g} [{torch_dtype}]"
    assert abs(got - expected) <= 2e-2 * abs(expected), message


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("deviation", VAR_DEVIATIONS, ids=VAR_DEVIATION_IDS)
def test_var_hw_matches_the_constructed_spread(deviation, torch_dtype, ttnn_dtype, device):
    _, in_data = _alternating(deviation, torch_dtype)
    tt_in = ttnn.from_torch(in_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    got = _first(ttnn.var_hw(tt_in))
    spread = float(torch.tensor([deviation], dtype=torch_dtype)[0])
    expected = spread * spread

    message = f"var_hw returned {got:g} where the variance is exactly {expected:g} [{torch_dtype}]"
    assert abs(got - expected) <= 4e-2 * expected, message


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
def test_std_hw_scales_with_its_input(torch_dtype, ttnn_dtype, device):
    """std(k*x) == k*std(x) stated directly, across the exponent range."""
    _, base = _alternating(1.0, torch_dtype)
    tt_base = ttnn.from_torch(base, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    unit = _first(ttnn.std_hw(tt_base))
    assert abs(unit - 1.0) <= 2e-2, f"std_hw of the unit spread is {unit:g}, expected 1"

    for k in (1e-25, 1e-19, 1e-6, 1e6, 1e19, 1e25):
        _, scaled = _alternating(k, torch_dtype)
        tt_scaled = ttnn.from_torch(scaled, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        got = _first(ttnn.std_hw(tt_scaled))
        want = k * unit
        message = f"std_hw(k*x) is {got:g} but k*std_hw(x) is {want:g}, for k = {k:g} [{torch_dtype}]"
        assert abs(got - want) <= 2e-2 * abs(want), message
