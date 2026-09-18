# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

# std_hw, var_hw and normalize_hw compute the variance as the mean of (x - mean)^2, forming the
# square as a value of its own. That square is the only part of the computation with a narrower
# range than the answer it feeds:
#
#   |x - mean| < 1.0842e-19   the square falls below the smallest normal and flushes to zero, so
#                             the standard deviation of an ordinary spread comes back an exact 0
#                             and normalize_hw divides by it
#   |x - mean| > 1.3043818e19 the accumulator tops out at 2^127, so the standard deviation comes
#                             back as sqrt(2^127) -- the same number for every input above that,
#                             which makes normalize_hw grow linearly where the answer is fixed
#
# The tensors below are built so the answer is exact rather than approximate: alternating columns
# of +v and -v give a mean of exactly zero and a population standard deviation of exactly v, at
# every magnitude. normalize_hw must therefore return the +/-1 pattern unchanged.
#
# That construction is the reference on purpose. torch.std in float32 forms the same square and
# overflows at the top of this range, so it cannot referee the high end; the analytic value can.
DEVIATIONS = (1e-25, 1e-20, 1e-19, 1e-10, 1.0, 1e10, 1e18, 1e20, 1e25)
DEVIATION_IDS = ["1e-25", "1e-20", "1e-19", "1e-10", "1.0", "1e10", "1e18", "1e20", "1e25"]

DTYPES = ((torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16))
DTYPE_IDS = ("float32", "bfloat16")

SHAPE = torch.Size([1, 1, 32, 32])


def _alternating(deviation, torch_dtype):
    """Columns of +deviation and -deviation: mean exactly 0, population std exactly |deviation|."""
    sign = torch.ones(SHAPE)
    sign[..., 1::2] = -1.0
    return sign, (sign * deviation).to(torch_dtype)


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

    got = float(ttnn.to_torch(ttnn.std_hw(tt_in)).float().flatten()[0])
    expected = float(torch.tensor([deviation], dtype=torch_dtype)[0])

    assert got != 0.0, f"std_hw returned an exact 0 for a spread of {expected:g} [{torch_dtype}]"
    assert got == got and abs(got) != float("inf"), f"std_hw returned {got} for {expected:g}"
    assert abs(got - expected) <= 2e-2 * abs(expected), (
        f"std_hw returned {got:g} where the spread is exactly {expected:g} [{torch_dtype}]"
    )


@pytest.mark.parametrize("torch_dtype, ttnn_dtype", DTYPES, ids=DTYPE_IDS)
def test_std_hw_scales_with_its_input(torch_dtype, ttnn_dtype, device):
    """std(k*x) == k*std(x) stated directly, across the exponent range."""
    _, base = _alternating(1.0, torch_dtype)
    tt_base = ttnn.from_torch(base, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    unit = float(ttnn.to_torch(ttnn.std_hw(tt_base)).float().flatten()[0])
    assert abs(unit - 1.0) <= 2e-2, f"std_hw of the unit spread is {unit:g}, expected 1"

    for k in (1e-25, 1e-19, 1e-6, 1e6, 1e19, 1e25):
        _, scaled = _alternating(k, torch_dtype)
        tt_scaled = ttnn.from_torch(scaled, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        got = float(ttnn.to_torch(ttnn.std_hw(tt_scaled)).float().flatten()[0])
        want = k * unit
        assert abs(got - want) <= 2e-2 * abs(want), (
            f"std_hw(k*x) is {got:g} but k*std_hw(x) is {want:g}, for k = {k:g} [{torch_dtype}]"
        )
