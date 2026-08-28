# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The assertions in `assertions.py`, checked against inputs they must reject. No device.

An assertion that never fails is indistinguishable from one that always passes, and it is
the only thing standing between a broken read and a green suite. Every case below is a
tensor pair some gate in this directory would hand it.
"""

import pytest
import torch

from models.demos.deepseek_v3_d_p.tests.attn_res.assertions import assert_accurate, assert_bit_identical, assert_equal

ALL_ASSERTIONS = [assert_accurate, assert_equal, assert_bit_identical]


@pytest.mark.parametrize(
    "value, label", [(float("nan"), "nan=1"), (float("inf"), r"\+inf=1"), (-float("inf"), "-inf=1")]
)
@pytest.mark.parametrize("assertion", ALL_ASSERTIONS)
def test_assertions_reject_nonfinite_values(assertion, value, label, expect_error):
    """Without a finiteness gate, a NaN read would pass `assert_accurate` against a NaN oracle.

    PCC of two NaN tensors is NaN, and `NaN < threshold` is false, so nothing raises. The two
    exact assertions reject NaN unaided — `torch.equal` is false wherever a NaN appears, even
    between a buffer and its own clone — so there the gate only names the cause.
    """
    tensor = torch.tensor([value])
    with expect_error(AssertionError, label):
        assertion(tensor, tensor.clone())


@pytest.mark.parametrize("assertion", ALL_ASSERTIONS)
def test_assertions_reject_shape_mismatch(assertion, expect_error):
    with expect_error(AssertionError, "shape"):
        assertion(torch.zeros(2), torch.zeros(1, 2))


@pytest.mark.parametrize("assertion", [assert_equal, assert_bit_identical])
def test_exact_assertions_reject_dtype_mismatch(assertion, expect_error):
    with expect_error(AssertionError, "dtype"):
        assertion(torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.bfloat16))


def test_assert_accurate_admits_a_dtype_mismatch():
    """The device returns bf16 and its oracle is fp32; requiring equality would narrow the oracle."""
    expected = torch.linspace(-1.0, 1.0, 64)
    assert_accurate(expected, expected.bfloat16(), pcc_threshold=0.999)


def test_assert_accurate_rejects_pcc_below_threshold(expect_error):
    with expect_error(AssertionError, "PCC"):
        assert_accurate(torch.arange(8.0), torch.arange(7, -1, -1.0), pcc_threshold=0.99)


def test_assert_accurate_rejects_a_constant_tensor(expect_error):
    """Correlation is undefined without variance, and undefined must not read as agreement."""
    with expect_error(AssertionError, "PCC nan"):
        assert_accurate(torch.ones(8), torch.zeros(8))


def test_assert_accurate_returns_the_measured_pcc():
    expected = torch.arange(8.0)
    pcc = assert_accurate(expected, expected + torch.linspace(0.0, 1e-4, 8), pcc_threshold=0.999999)
    assert 0.999999 <= pcc <= 1.0


def test_assert_equal_rejects_a_single_differing_element(expect_error):
    with expect_error(AssertionError, "values differ"):
        assert_equal(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))


def test_assert_equal_admits_signed_zero():
    """`assert_equal` compares values, and -0.0 == 0.0. Only the bit form separates them."""
    assert_equal(torch.tensor([0.0]), torch.tensor([-0.0]))


def test_assert_bit_identical_rejects_signed_zero(expect_error):
    with expect_error(AssertionError, "bit patterns differ"):
        assert_bit_identical(torch.tensor([0.0]), torch.tensor([-0.0]))


def test_assert_bit_identical_rejects_one_ulp(expect_error):
    """The smallest disagreement a repeated device read can produce."""
    expected = torch.tensor([1.0], dtype=torch.float32)
    with expect_error(AssertionError, "values differ"):
        assert_bit_identical(expected, torch.nextafter(expected, torch.tensor([2.0])))


def test_assert_bit_identical_accepts_a_true_repeat():
    tensor = torch.randn(4, 8, dtype=torch.float32)
    assert_bit_identical(tensor, tensor.clone())
