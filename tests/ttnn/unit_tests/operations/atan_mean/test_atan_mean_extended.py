# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Extended tests for atan_mean.

Focused additions beyond the acceptance test:
  - Saturation behavior: atan is bounded in [-π/2, π/2]; large-magnitude inputs
    must converge to the asymptote.
  - Sign symmetry: atan is odd; atan_mean(-x) == -atan_mean(x).
  - Zero input: atan_mean(0) == 0.
  - Domain regime that the acceptance test does not cover: a positive-only
    domain (no sign cancellation) so the row mean is non-trivial.
  - One mid-size shape not in the acceptance test to broaden Wt coverage.

Numerical tolerance follows the Phase-0 acceptance contract (PCC >= 0.9995,
max-abs <= 1e-2). Refinement agents will tighten these.
"""

import pytest
import torch
import ttnn

from ttnn.operations.atan_mean import atan_mean

from tests.ttnn.utils_for_testing import assert_with_pcc


PCC_THRESHOLD = 0.9995
MAX_ABS_TOL = 1e-2


def _to_device(torch_input: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 96), id="Wt3_mid_shape"),
        pytest.param((1, 4, 96, 64), id="NCxHxW_mid"),
    ],
)
def test_atan_mean_extra_shapes(device, shape):
    """Mid-size shapes filling a coverage gap in Wt and NC."""
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)
    expected = torch.atan(torch_input).mean(dim=-1)

    actual = ttnn.to_torch(atan_mean(_to_device(torch_input, device))).float()

    assert (actual - expected).abs().max().item() <= MAX_ABS_TOL
    assert_with_pcc(expected, actual, pcc=PCC_THRESHOLD)


def test_atan_mean_zero_input(device):
    """atan(0) == 0, so the row-mean is zero everywhere."""
    shape = (1, 1, 64, 64)
    torch_input = torch.zeros(shape, dtype=torch.float32)
    actual = ttnn.to_torch(atan_mean(_to_device(torch_input, device))).float()
    # atan(0) = 0 → mean = 0, so this should be exactly zero (within fp32 noise).
    assert actual.abs().max().item() < 1e-5


def test_atan_mean_sign_antisymmetry(device):
    """atan is odd: atan_mean(-x) == -atan_mean(x)."""
    torch.manual_seed(7)
    shape = (1, 2, 64, 64)
    torch_input = torch.randn(shape, dtype=torch.float32)

    out_pos = ttnn.to_torch(atan_mean(_to_device(torch_input, device))).float()
    out_neg = ttnn.to_torch(atan_mean(_to_device(-torch_input, device))).float()

    # The two should sum to zero (modulo SFPU approximation symmetry — atan is
    # implemented symmetrically in the LLK, so this is tight).
    assert (out_pos + out_neg).abs().max().item() <= 2 * MAX_ABS_TOL


def test_atan_mean_saturation_large_positive(device):
    """Large positive inputs saturate to atan asymptote π/2 ≈ 1.5708."""
    shape = (1, 1, 32, 64)
    # Inputs of magnitude 1e6 → atan ≈ π/2 to within ~1e-6.
    torch_input = torch.full(shape, 1e6, dtype=torch.float32)
    actual = ttnn.to_torch(atan_mean(_to_device(torch_input, device))).float()
    # Every row-mean should be ≈ π/2.
    half_pi = 3.141592653589793 / 2.0
    assert (actual - half_pi).abs().max().item() <= 1e-3


def test_atan_mean_positive_domain(device):
    """
    Positive-only input — no sign cancellation in the row mean. atan is
    monotonic on R+ so the mean is bounded by atan(max(x)). This catches
    bugs that pass random-sign inputs but break on shifted distributions.
    """
    torch.manual_seed(3)
    shape = (1, 1, 64, 64)
    torch_input = torch.rand(shape, dtype=torch.float32) * 4.0 + 1.0  # in [1, 5]
    expected = torch.atan(torch_input).mean(dim=-1)
    actual = ttnn.to_torch(atan_mean(_to_device(torch_input, device))).float()

    assert (actual - expected).abs().max().item() <= MAX_ABS_TOL
    assert_with_pcc(expected, actual, pcc=PCC_THRESHOLD)
