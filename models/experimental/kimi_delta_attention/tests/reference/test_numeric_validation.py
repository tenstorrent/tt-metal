# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Contracts for shared KDA numeric-result validation."""

import pytest
import torch

from models.experimental.kimi_delta_attention.tests.utils import assert_all_finite


def test_assert_all_finite_reports_zero_counts_and_fractions(capsys: pytest.CaptureFixture[str]) -> None:
    assert_all_finite("finite", torch.tensor([0.0, 1.0, -1.0]))

    assert capsys.readouterr().out.strip() == (
        "finite finiteness: non_finite=0/3 (0.000000e+00), nan=0/3 (0.000000e+00), "
        "+inf=0/3 (0.000000e+00), -inf=0/3 (0.000000e+00)"
    )


@pytest.mark.parametrize(
    "value,expected_field",
    [
        (float("nan"), "nan=1/2 (5.000000e-01)"),
        (float("inf"), "+inf=1/2 (5.000000e-01)"),
        (-float("inf"), "-inf=1/2 (5.000000e-01)"),
    ],
)
def test_assert_all_finite_rejects_each_non_finite_category(
    value: float,
    expected_field: str,
    capsys: pytest.CaptureFixture[str],
    expect_error,
) -> None:
    with expect_error(AssertionError, "non_finite=1/2"):
        assert_all_finite("invalid", torch.tensor([0.0, value]))

    output = capsys.readouterr().out
    assert "non_finite=1/2 (5.000000e-01)" in output
    assert expected_field in output
