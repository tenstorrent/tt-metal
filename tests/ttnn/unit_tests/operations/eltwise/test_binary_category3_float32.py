# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_binary_grid,
    pairwise_from_values,
    pairwise_inputs,
    run_binary,
    to_float32_tt_tensor,
)

pytestmark = pytest.mark.use_module_device

"""
Category 3: binary logical ops

| #  | Op                | Computes              | Check (float32) |
|----|-------------------|-----------------------|-----------------|
| 1  | ttnn.logical_and  | (a != 0) and (b != 0) | exact           |
| 2  | ttnn.logical_or   | (a != 0) or (b != 0)  | exact           |
| 3  | ttnn.logical_xor  | (a != 0) xor (b != 0) | exact           |
| 4  | ttnn.logical_and_ | logical_and (inplace) | exact           |
| 5  | ttnn.logical_or_  | logical_or (inplace)  | exact           |
| 6  | ttnn.logical_xor_ | logical_xor (inplace) | exact           |

float32 operands take the SFPU path: nez on each operand, an SFPU mul / add /
sub of the two 0/1 values, then nez again. The result is 0 or 1 in float32.

nez tests the operand bits, so subnormal inputs are not flushed here: any
nonzero bit pattern is True, as in torch, including the smallest subnormal
and NaNs. ±0 is False.

The ops are covered tensor-tensor and tensor-scalar; the inplace variants are
covered tensor-tensor only.
"""


BINARY_LOGICAL_OPS = [ttnn.logical_and, ttnn.logical_or, ttnn.logical_xor]
BINARY_LOGICAL_INPLACE_OPS = [ttnn.logical_and_, ttnn.logical_or_, ttnn.logical_xor_]


def _float32_from_bits(bits):
    return torch.tensor(bits, dtype=torch.int64).to(torch.int32).view(torch.float32)


# Bit patterns the bfloat16 grid cannot hold: nonzero only in the low 16
# mantissa bits, subnormal, or a NaN whose payload sits in the low bits.
_FLOAT32_ONLY_VALUES = _float32_from_bits(
    [
        0x00000001,  # +smallest subnormal, 1.4e-45
        0x80000001,  # -smallest subnormal
        0x0000FFFF,  # subnormal, only the low 16 bits set
        0x00400000,  # subnormal, 5.9e-39
        0x007FFFFF,  # +largest subnormal
        0x807FFFFF,  # -largest subnormal
        0x3F800001,  # 1 + 2^-23
        0x7F7FFFFF,  # float32 max
        0xFF7FFFFF,  # float32 min
        0x7F800001,  # NaN, only the lowest mantissa bit set
        0xFFC00000,  # -NaN
    ]
)


@pytest.mark.parametrize("ttnn_op", BINARY_LOGICAL_OPS + BINARY_LOGICAL_INPLACE_OPS)
def test_logical_ops(device, ttnn_op):
    """Pairwise coverage of the logical ops over the stratified grid at float32.

    | Case          | Truth value | Check |
    |---------------|-------------|-------|
    | finite != 0   | True        | exact |
    | ±0            | False       | exact |
    | ±inf, NaN     | True        | exact |
    """
    input_a, input_b = pairwise_inputs(include_spl_values=True, dtype=torch.float32)
    golden, result = run_binary(device, ttnn_op, input_a, input_b, input_dtype=ttnn.float32)
    assert_equal(golden.float(), result.float())


@pytest.mark.parametrize("ttnn_op", BINARY_LOGICAL_OPS + BINARY_LOGICAL_INPLACE_OPS)
def test_logical_ops_float32_bit_patterns(device, ttnn_op):
    """Cross product of the float32-only bit patterns with ±0, ±1, ±inf and
    +NaN.

    | Pattern                     | Truth value (device and torch) |
    |-----------------------------|--------------------------------|
    | subnormal, either sign      | True (not flushed)             |
    | only the low 16 bits set    | True                           |
    | NaN with a low-bit payload  | True                           |
    | -NaN                        | True                           |
    | ±0                          | False                          |

    A nez that looked only at the upper 16 bits, or flushed subnormal inputs
    like the arithmetic ops do, would call several of these False.
    """
    special = torch.tensor([0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")])
    values = torch.cat([_FLOAT32_ONLY_VALUES, special])
    input_a, input_b = pairwise_from_values(values)
    golden, result = run_binary(device, ttnn_op, input_a, input_b, input_dtype=ttnn.float32)
    assert_equal(golden.float(), result.float())


# One scalar per truth-value class, including subnormal and NaN scalars. The
# scalar is packed at full float32 precision, so a subnormal scalar is True.
_SCALAR_VALUES = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    float(_FLOAT32_ONLY_VALUES[0]),  # +smallest subnormal
    float(_FLOAT32_ONLY_VALUES[1]),  # -smallest subnormal
    float(torch.finfo(torch.float32).tiny),
    float(torch.finfo(torch.float32).max),
    float(torch.finfo(torch.float32).min),
    float("inf"),
    float("-inf"),
    float("nan"),
]


@pytest.mark.parametrize("ttnn_op", BINARY_LOGICAL_OPS + BINARY_LOGICAL_INPLACE_OPS)
@pytest.mark.parametrize("scalar", _SCALAR_VALUES)
def test_logical_ops_scalar(device, ttnn_op, scalar):
    """Tensor-scalar coverage of the logical ops: the stratified grid,
    including ±0, ±inf and qNaN, plus the float32-only bit patterns, against
    a scalar from every truth-value class.

    Each value is repeated across one tile width, so the input spans many
    full tiles. A 1-D tensor in tile layout is one tile row with 31/32
    padding, so it never reaches the multi-tile scalar path. torch.logical_*
    has no scalar overload, so the golden uses a 0-d tensor.
    """
    values = torch.cat(
        [generate_bfloat16_binary_grid(include_spl_values=True, dtype=torch.float32), _FLOAT32_ONLY_VALUES]
    )
    tile_width = 32
    input_a = values.unsqueeze(1).expand(values.numel(), tile_width).contiguous()
    tt_a = to_float32_tt_tensor(input_a, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_a, torch.tensor(scalar, dtype=input_a.dtype))
    result = ttnn.to_torch(ttnn_op(tt_a, scalar))

    assert_equal(golden.float(), result.float())
