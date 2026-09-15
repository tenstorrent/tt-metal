# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    assert_equal,
)


@pytest.mark.parametrize(
    "value,label",
    [(float("nan"), "nan=1"), (float("inf"), r"\+inf=1"), (-float("inf"), "-inf=1")],
)
@pytest.mark.parametrize("assertion", [assert_accurate, assert_equal, assert_bit_identical])
def test_assertions_reject_nonfinite_values(assertion, value: float, label: str, expect_error) -> None:
    tensor = torch.tensor([value])
    with expect_error(AssertionError, label):
        assertion(tensor, tensor.clone())


@pytest.mark.parametrize("assertion", [assert_accurate, assert_equal, assert_bit_identical])
def test_assertions_reject_shape_metadata_mismatch(assertion, expect_error) -> None:
    with expect_error(AssertionError, "shape"):
        assertion(torch.zeros(2), torch.zeros(1, 2))


@pytest.mark.parametrize("assertion", [assert_accurate, assert_equal, assert_bit_identical])
def test_assertions_reject_dtype_metadata_mismatch(assertion, expect_error) -> None:
    with expect_error(AssertionError, "dtype"):
        assertion(torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.bfloat16))


def test_assert_accurate_rejects_pcc_below_threshold(expect_error) -> None:
    with expect_error(AssertionError, "PCC"):
        assert_accurate(torch.arange(8.0), torch.arange(7, -1, -1.0), pcc_threshold=0.99)


def test_assert_equal_rejects_value_mismatch(expect_error) -> None:
    with expect_error(AssertionError, "values differ"):
        assert_equal(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))


def test_assert_bit_identical_rejects_bit_mismatch(expect_error) -> None:
    expected = torch.tensor([1.0], dtype=torch.float32)
    actual = torch.nextafter(expected, torch.tensor([2.0]))
    with expect_error(AssertionError, "values differ"):
        assert_bit_identical(expected, actual)


def test_assert_bit_identical_rejects_signed_zero_mismatch(expect_error) -> None:
    with expect_error(AssertionError, "bit patterns differ"):
        assert_bit_identical(torch.tensor([0.0]), torch.tensor([-0.0]))


def test_assert_accurate_returns_measured_pcc() -> None:
    expected = torch.arange(8.0)
    actual = expected + torch.linspace(0.0, 1e-4, 8)
    pcc = assert_accurate(expected, actual, pcc_threshold=0.999999)
    assert 0.999999 <= pcc <= 1.0
