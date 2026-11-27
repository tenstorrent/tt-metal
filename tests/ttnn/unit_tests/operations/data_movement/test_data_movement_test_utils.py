# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of data_movement_test_utils.py."""

import math

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.data_movement.data_movement_test_utils import assert_exact, full_range_input

FLOATS = [ttnn.bfloat16, ttnn.float32]
INT_RANGES = {
    ttnn.int32: (-(2**31), 2**31 - 1),
    ttnn.uint32: (0, 2**32 - 1),
    ttnn.uint16: (0, 2**16 - 1),
    ttnn.uint8: (0, 2**8 - 1),
}
SPECIAL = [math.inf, -math.inf, math.nan, -0.0, 0.0]


@pytest.mark.parametrize("nan", [True, False])
@pytest.mark.parametrize("dtype", FLOATS)
def test_floats_start_with_the_special_values(dtype, nan):
    special = [value for value in SPECIAL if nan or not math.isnan(value)]
    values = full_range_input((64, 64), dtype, nan=nan).flatten()
    assert_exact(torch.tensor(special), values[: len(special)])
    assert nan or not values.isnan().any()


@pytest.mark.parametrize("dtype", FLOATS)
def test_floats_cover_every_normal_exponent_and_both_signs(dtype):
    values = full_range_input((128, 128), dtype)
    rest = values.flatten()[len(SPECIAL) :].to(torch.float64)
    finfo = torch.finfo(values.dtype)
    assert rest.isfinite().all()
    assert (rest.abs() >= finfo.tiny).all(), "subnormal or zero values"
    exponents = torch.frexp(rest).exponent.unique()
    assert len(exponents) == 254, f"{len(exponents)} distinct exponents"
    assert (rest < 0).any() and (rest > 0).any()


@pytest.mark.parametrize("dtype", INT_RANGES)
def test_ints_cover_their_whole_range(dtype):
    low, high = INT_RANGES[dtype]
    values = full_range_input((64, 64), dtype).flatten().to(torch.int64)
    assert values[:2].tolist() == [low, high]
    assert values.min() == low and values.max() == high
    middle = (low + high) // 2
    assert (values < middle).sum() > len(values) // 4 and (values > middle).sum() > len(values) // 4


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", FLOATS + list(INT_RANGES))
def test_from_torch_takes_the_values_unchanged(dtype, layout):
    values = full_range_input((32, 64), dtype)
    assert_exact(values, ttnn.to_torch(ttnn.from_torch(values, dtype=dtype, layout=layout)))


@pytest.mark.parametrize(
    "expected, actual",
    [([math.nan, 1.0], [math.nan, 1.0]), ([-0.0, 0.0], [-0.0, 0.0]), ([math.inf, -math.inf], [math.inf, -math.inf])],
    ids=["nan", "signed-zeros", "infinities"],
)
def test_assert_exact_passes_equal_values(expected, actual):
    assert_exact(torch.tensor(expected), torch.tensor(actual))


@pytest.mark.parametrize(
    "expected, actual",
    [([math.nan], [1.0]), ([1.0], [math.nan]), ([-0.0], [0.0]), ([0.0], [-0.0]), ([1.0], [1.0 + 2**-23])],
    ids=["nan-became-number", "number-became-nan", "minus-zero-became-plus", "plus-zero-became-minus", "last-bit"],
)
def test_assert_exact_fails_different_values(expected, actual, expect_error):
    with expect_error(AssertionError, "1 of 1 values differ"):
        assert_exact(torch.tensor(expected), torch.tensor(actual))


def test_assert_exact_can_ignore_the_sign_of_zero(expect_error):
    assert_exact(torch.tensor([-0.0, 0.0]), torch.tensor([0.0, -0.0]), signed_zeros=False)
    with expect_error(AssertionError, "values differ"):
        assert_exact(torch.tensor([math.nan]), torch.tensor([0.0]), signed_zeros=False)


def test_assert_exact_compares_large_ints_exactly(expect_error):
    with expect_error(AssertionError, "values differ"):
        assert_exact(torch.tensor([2**32 - 1]), torch.tensor([2**32 - 2]))


def test_assert_exact_compares_shapes(expect_error):
    with expect_error(AssertionError, "shape"):
        assert_exact(torch.zeros(2, 3), torch.zeros(3, 2))
