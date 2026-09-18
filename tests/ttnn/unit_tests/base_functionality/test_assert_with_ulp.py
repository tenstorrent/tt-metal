# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
import torch

from tests.ttnn.utils_for_testing import assert_with_ulp


def test_assert_with_ulp_requires_keyword_arguments():
    parameters = inspect.signature(assert_with_ulp).parameters
    assert tuple(parameters) == ("expected_result", "actual_result", "ulp_threshold", "allow_nonfinite")
    assert all(parameter.kind == inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_assert_with_ulp_uses_expected_reference(dtype, sign, expect_error):
    actual = torch.tensor([sign], dtype=dtype)
    towards_zero = torch.zeros_like(actual)
    adjacent = torch.nextafter(actual, towards_zero)
    expected = torch.nextafter(adjacent, towards_zero)

    # The spacing at |actual| == 1 is twice the spacing just below it.
    # Two steps from the reference must fail a one-ULP tolerance.
    with expect_error(AssertionError, r"Max ULP Delta: 2\.0"):
        assert_with_ulp(expected_result=expected, actual_result=actual, ulp_threshold=1)

    assert_with_ulp(expected_result=expected, actual_result=adjacent, ulp_threshold=1)

    # Deliberately swap the operands: normalizing by ULP(actual) hides the error.
    assert_with_ulp(expected_result=actual, actual_result=expected, ulp_threshold=1)
