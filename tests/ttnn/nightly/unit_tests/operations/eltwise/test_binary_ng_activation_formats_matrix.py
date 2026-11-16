# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.eltwise.test_binary_ng_activation_mixed_dtype import (
    _DTYPE_PAIRS,
    _SHAPES,
    _check_abs,
    _input,
)

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("op", [ttnn.add, ttnn.multiply], ids=["add", "multiply"])
@pytest.mark.parametrize("a_dtype,b_dtype", _DTYPE_PAIRS)
@pytest.mark.parametrize("a_shape,b_shape", _SHAPES)
@pytest.mark.parametrize("side", ["lhs", "rhs", "both"])
def test_fused_abs_formats(device, op, a_dtype, b_dtype, a_shape, b_shape, side):
    a = _input(device, a_shape, a_dtype, 3)
    b = _input(device, b_shape, b_dtype, 13)
    _check_abs(op, a, b, side)
