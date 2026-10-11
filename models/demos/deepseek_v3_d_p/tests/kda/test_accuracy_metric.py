# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hand-computed checks of the FP64 KDA accuracy metric. CPU only."""

import pytest
import torch

from models.demos.deepseek_v3_d_p.tests.kda.accuracy_metric import gate, position_profile, slice_nrmse


def test_whole_tensor_nrmse_is_error_norm_over_golden_norm() -> None:
    golden = torch.tensor([3.0, 4.0])  # norm 5
    actual = torch.tensor([3.0, 5.0])  # error norm 1
    assert slice_nrmse(actual, golden, keep_dims=()).item() == pytest.approx(0.2, rel=1e-15)


def test_nrmse_is_reported_per_kept_slice() -> None:
    golden = torch.tensor([[3.0, 4.0], [6.0, 8.0]])  # row norms 5, 10
    actual = golden + torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    torch.testing.assert_close(
        slice_nrmse(actual, golden, keep_dims=(0,)), torch.tensor([0.2, 0.1], dtype=torch.float64), rtol=1e-15, atol=0
    )


def test_near_zero_golden_slice_is_floored_at_the_median_slice_norm() -> None:
    golden = torch.tensor([[3.0, 4.0], [0.0, 0.0], [6.0, 8.0], [30.0, 40.0]], dtype=torch.float64)  # norms 5, 0, 10, 50
    actual = golden + torch.tensor([[0.0, 0.0], [0.0, 0.01], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    # torch.median takes the lower middle of (0, 5, 10, 50): 5, so the floor is 0.05 and 0.01 / 0.05 = 0.2.
    errors = slice_nrmse(actual, golden, keep_dims=(0,))
    assert errors[1].item() == pytest.approx(0.2, rel=1e-12)
    assert errors[0].item() == errors[2].item() == errors[3].item() == 0.0


def test_metric_arithmetic_is_float64() -> None:
    golden = torch.tensor([1.0], dtype=torch.float32)
    actual = torch.tensor([1.0 + 2**-23], dtype=torch.float32)  # one FP32 ulp above 1
    assert slice_nrmse(actual, golden, keep_dims=()).item() == 2**-23


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_actual_raises(bad: float, expect_error) -> None:
    with expect_error(ValueError, "1 non-finite"):
        slice_nrmse(torch.tensor([1.0, bad]), torch.tensor([1.0, 1.0]), keep_dims=())


def test_shape_mismatch_raises(expect_error) -> None:
    with expect_error(ValueError, "shape"):
        slice_nrmse(torch.zeros(2, 3), torch.zeros(3, 2), keep_dims=())


def test_position_profile_localizes_error_to_its_window() -> None:
    golden = torch.ones(1, 5, 2)
    actual = golden.clone()
    actual[0, 4, 0] = 2.0  # last window (one token, norm sqrt 2) has error 1
    torch.testing.assert_close(
        position_profile(actual, golden, token_dim=1, bucket=2),
        torch.tensor([0.0, 0.0, 2**-0.5], dtype=torch.float64),
        rtol=1e-15,
        atol=0,
    )


def test_gate_compares_each_slice_with_scaled_budget_plus_epsilon() -> None:
    result = gate(torch.tensor([0.1, 0.3]), torch.tensor([0.1, 0.1]), margin=1.0, epsilon=0.05)  # limit 0.25
    torch.testing.assert_close(result.limit, torch.tensor([0.25, 0.25], dtype=torch.float64))
    assert not result.passed
    assert "FAIL in 1 of 2; worst slice 1" in result.describe("state")
    assert gate(torch.tensor([0.25]), torch.tensor([0.1]), margin=1.0, epsilon=0.05).passed


def test_gate_rejects_mismatched_budget_and_negative_tolerances(expect_error) -> None:
    with expect_error(ValueError, "budget shape"):
        gate(torch.zeros(2), torch.zeros(3), margin=0.0, epsilon=0.0)
    with expect_error(ValueError, "nonnegative"):
        gate(torch.zeros(2), torch.zeros(2), margin=-0.1, epsilon=0.0)
