# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Contracts for shared KDA numeric-result validation."""

import pytest
import torch

from models.demos.deepseek_v3_d_p.tests.kda.utils import assert_accurate, assert_bit_identical, assert_equal


def test_assert_accurate_accepts_finite_high_pcc_tensors() -> None:
    golden = torch.tensor([0.0, 1.0, 2.0, 4.0])
    actual = torch.tensor([0.0, 1.0, 2.0, 4.0])

    assert_accurate(golden, actual)


@pytest.mark.parametrize(
    "side,value,expected_field",
    [
        (side, value, expected_field)
        for side in ("golden", "actual")
        for value, expected_field in (
            (float("nan"), "nan=1/4"),
            (float("inf"), r"\+inf=1/4"),
            (-float("inf"), "-inf=1/4"),
        )
    ],
)
def test_assert_accurate_rejects_non_finite_tensor(side: str, value: float, expected_field: str, expect_error) -> None:
    golden = torch.tensor([0.0, 1.0, 2.0, 4.0])
    actual = golden.clone()
    target = golden if side == "golden" else actual
    target[-1] = value

    with expect_error(AssertionError, expected_field):
        assert_accurate(golden, actual)


def test_assert_accurate_rejects_low_pcc(expect_error) -> None:
    golden = torch.tensor([0.0, 1.0, 2.0, 4.0])
    actual = torch.tensor([4.0, 2.0, 1.0, 0.0])

    with expect_error(AssertionError, "PCC"):
        assert_accurate(golden, actual, pcc_threshold=0.999)


def test_assert_equal_accepts_finite_equal_tensors() -> None:
    expected = torch.tensor([1.0, 2.0])
    assert_equal(expected, expected.clone())


@pytest.mark.parametrize(
    "side,value,expected_field",
    [
        (side, value, expected_field)
        for side in ("expected", "actual")
        for value, expected_field in (
            (float("nan"), "nan=1/2"),
            (float("inf"), r"\+inf=1/2"),
            (-float("inf"), "-inf=1/2"),
        )
    ],
)
def test_assert_equal_rejects_non_finite_tensor(side: str, value: float, expected_field: str, expect_error) -> None:
    expected = torch.tensor([1.0, 2.0])
    actual = expected.clone()
    target = expected if side == "expected" else actual
    target[-1] = value

    with expect_error(AssertionError, expected_field):
        assert_equal(expected, actual)


def test_assert_equal_rejects_changed_value(expect_error) -> None:
    with expect_error(AssertionError, "values differ"):
        assert_equal(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))


def test_assert_bit_identical_rejects_changed_value(expect_error) -> None:
    expected = torch.tensor([1.0, 2.0])
    actual = torch.tensor([1.0, 3.0])

    with expect_error(AssertionError, "not bit-identical"):
        assert_bit_identical(expected, actual)
