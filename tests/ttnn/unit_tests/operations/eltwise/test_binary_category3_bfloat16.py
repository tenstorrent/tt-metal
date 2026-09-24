# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    pairwise_inputs,
    run_binary,
    to_tt_tensor,
)

pytestmark = pytest.mark.use_module_device

"""
Category 3: binary logical ops

 1. ttnn.logical_and   - Logical AND
 2. ttnn.logical_or    - Logical OR
 3. ttnn.logical_xor   - Logical XOR

Each op also has an inplace variant (logical_and_, logical_or_, logical_xor_)
that writes the result back into the first operand. All six also have a
tensor-scalar (TS) overload (TTNN_BINARY_OP_TENSOR_SCALAR_IMPL /
TTNN_BINARY_OP_INPLACE_INVOKE_IMPL in binary.cpp) alongside the
tensor-tensor (TT) one; both are exercised here.

Accuracy criteria
─────────────────
  Every op reduces to a truth-value test per operand (nonzero => True) before
  combining, so the result is bit-exact 0/1 in the input dtype for every
  finite, zero, inf, and NaN pairing (NaN is nonzero, hence True).
"""

BINARY_LOGICAL_OPS = [ttnn.logical_and, ttnn.logical_or, ttnn.logical_xor]
BINARY_LOGICAL_INPLACE_OPS = [ttnn.logical_and_, ttnn.logical_or_, ttnn.logical_xor_]

# Representative scalars spanning every truth-value / special-value class.
_SCALAR_VALUES = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    float(torch.finfo(torch.bfloat16).max),
    float(torch.finfo(torch.bfloat16).min),
    float("inf"),
    float("-inf"),
    float("nan"),
]


@pytest.mark.parametrize("ttnn_op", BINARY_LOGICAL_OPS + BINARY_LOGICAL_INPLACE_OPS)
def test_logical_ops(device, ttnn_op):
    """Pairwise (TT) coverage of ttnn.logical_and / logical_or / logical_xor
    (and their inplace variants) over the stratified bfloat16 grid.

    The grid includes ±0, ±inf, and one qNaN. Every nonzero operand (including
    inf and NaN) is truthy, matching torch.logical_*; the result is exact 0/1.
    """
    input_a, input_b = pairwise_inputs(include_spl_values=True)
    golden, result = run_binary(device, ttnn_op, input_a, input_b)
    assert_equal(golden.float(), result.float())


@pytest.mark.parametrize("ttnn_op", BINARY_LOGICAL_OPS + BINARY_LOGICAL_INPLACE_OPS)
@pytest.mark.parametrize("scalar", _SCALAR_VALUES)
def test_logical_ops_scalar(device, ttnn_op, scalar):
    """Tensor-scalar (TS) coverage of ttnn.logical_and / logical_or / logical_xor
    (and their inplace variants): the stratified bfloat16 grid against a scalar
    spanning every truth-value / special-value class.

    Reuses the 2048x2048 pairwise_inputs grid (rather than the underlying 1-D
    2048-value grid) so operand A spans many tile rows/columns -- a 1-D
    [2048] tensor in tile layout would sit in a single tile band with 31/32
    of it padding, never reaching the multi-tile part of the scalar path.

    torch.logical_* has no scalar overload, so the golden broadcasts a 0-d
    tensor of the same dtype instead -- equivalent to the TT path, just
    without uploading a second full-size tensor to device.
    """
    input_a, _ = pairwise_inputs(include_spl_values=True)
    tt_a = to_tt_tensor(input_a, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_a, torch.tensor(scalar, dtype=input_a.dtype))

    if ttnn_op.__name__.endswith("_"):
        ttnn_op(tt_a, scalar)
        result = ttnn.to_torch(tt_a)
    else:
        result = ttnn.to_torch(ttnn_op(tt_a, scalar))

    assert_equal(golden.float(), result.float())
