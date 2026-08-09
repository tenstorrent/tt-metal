# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_ulp


# nextafter moves its input by exactly one ULP, so a correlation check cannot referee it: a PCC
# threshold cannot separate `a + 1ulp` from `a - 1ulp`, nor either of them from `a` left alone.
# These assert equality against torch.nextafter instead.


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "a_value, b_value",
    [
        (1.0, 2.0),  # step up from the bottom of a binade
        (1.0, 0.0),  # step down across a binade boundary
        (2.0, 1.0),
        (100.0, 200.0),  # a magnitude where a fixed FLT_EPSILON rounds away entirely
        (1.0e10, 2.0e10),
        (1.0e-5, 1.0),
        (-1.0, 0.0),  # negative operands: the step must shrink the magnitude
        (-1.0, -2.0),  # and here it must grow it
        (0.0, 1.0),  # zero has no magnitude bits: the neighbour is the smallest subnormal
        (0.0, -1.0),
        (7.5, 7.5),  # nextafter(x, x) is x
    ],
)
def test_nextafter_is_exact(device, dtype, a_value, b_value):
    shape = (1, 1, 32, 32)
    torch_a = torch.full(shape, a_value, dtype=dtype)
    torch_b = torch.full(shape, b_value, dtype=dtype)
    expected = torch.nextafter(torch_a, torch_b)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.nextafter(a, b))

    # ulp_threshold=0 is exactness in the units this op is defined in.
    assert_with_ulp(expected, actual, 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nextafter_moves_toward_the_target(device, dtype):
    # The direction is the property a PCC assertion is blindest to, so it gets its own check
    # across twelve decades in both directions.
    exponents = torch.arange(-6, 7, dtype=torch.float32)
    magnitudes = torch.cat([torch.pow(10.0, exponents), -torch.pow(10.0, exponents)])

    for magnitude in magnitudes.tolist():
        for target in (magnitude * 2.0, magnitude * 0.5):
            shape = (1, 1, 32, 32)
            torch_a = torch.full(shape, magnitude, dtype=dtype)
            torch_b = torch.full(shape, target, dtype=dtype)
            if torch.equal(torch_a, torch_b):
                continue

            a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
            b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
            actual = ttnn.to_torch(ttnn.nextafter(a, b))

            assert_with_ulp(torch.nextafter(torch_a, torch_b), actual, 0)
