# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    pairwise_inputs,
    run_binary,
)

pytestmark = pytest.mark.use_module_device

"""
Category 3: binary logical ops

 1. ttnn.logical_and   - Logical AND
 2. ttnn.logical_or    - Logical OR
 3. ttnn.logical_xor   - Logical XOR

Each op also has an inplace variant (logical_and_, logical_or_, logical_xor_)
that writes the result back into the first operand.

Accuracy criteria
─────────────────
  Every op reduces to a truth-value test per operand (nonzero => True) before
  combining, so the result is bit-exact 0/1 in the input dtype for every
  finite, zero, inf, and NaN pairing (NaN is nonzero, hence True).
"""

BINARY_LOGICAL_OPS = [ttnn.logical_and, ttnn.logical_or, ttnn.logical_xor]
BINARY_LOGICAL_INPLACE_OPS = [ttnn.logical_and_, ttnn.logical_or_, ttnn.logical_xor_]


@pytest.mark.parametrize("ttnn_op", BINARY_LOGICAL_OPS + BINARY_LOGICAL_INPLACE_OPS)
def test_logical_ops(device, ttnn_op):
    """Pairwise coverage of ttnn.logical_and / logical_or / logical_xor (and their
    inplace variants) over the stratified bfloat16 grid.

    The grid includes ±0, ±inf, and one qNaN. Every nonzero operand (including
    inf and NaN) is truthy, matching torch.logical_*; the result is exact 0/1.
    """
    input_a, input_b = pairwise_inputs(include_spl_values=True)
    golden, result = run_binary(device, ttnn_op, input_a, input_b)
    assert_equal(golden.float(), result.float())
