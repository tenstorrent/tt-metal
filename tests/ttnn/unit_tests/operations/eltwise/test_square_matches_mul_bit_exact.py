# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits_in_range,
    to_tt_tensor,
)
from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.use_module_device


def test_square_matches_mul_bit_exact(device):
    # ttnn.square(x) and ttnn.mul(x, x) both narrow the same fp32 product to bfloat16;
    # mul_binary_tile already rounds to nearest-even, so square must match it bit for bit
    # instead of truncating (was biased toward zero on every inexact lane).
    input_tensor = generate_bfloat16_bits_in_range(-1e19, 1e19)
    tt_in = to_tt_tensor(input_tensor, device)

    square_result = ttnn.to_torch(ttnn.square(tt_in))
    mul_result = ttnn.to_torch(ttnn.mul(tt_in, tt_in))

    assert_equal(mul_result, square_result)
