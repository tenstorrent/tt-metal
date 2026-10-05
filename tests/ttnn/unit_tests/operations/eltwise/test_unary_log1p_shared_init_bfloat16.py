# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""BF16 ops whose stock init calls log1p_init keep stock's accuracy next to the generated log1p kernel.

asinh and acosh (init_inverse_hyperbolic) and atanh (init_atanh) read the programmable constants
log1p_init sets. In craq-sim, stock is within 2 ULP of torch on every checked input; a log1p
kernel that changed those constants moves them by up to 71 ULP.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    SMALLEST_NORMAL_BF16,
    generate_bfloat16_bits,
    to_tt_tensor,
)
from tests.ttnn.utils_for_testing import assert_with_ulp

CALLERS = [
    (ttnn.UnaryOpType.ASINH, ttnn.asinh, torch.asinh),
    (ttnn.UnaryOpType.ACOSH, ttnn.acosh, torch.acosh),
    (ttnn.UnaryOpType.ATANH, ttnn.atanh, torch.atanh),
]
IDS = ["asinh", "acosh", "atanh"]
pytestmark = pytest.mark.skipif(not is_blackhole(), reason="compiler-generated BF16 log1p kernel ships on Blackhole")


def _checked(source, golden):
    """Normal inputs whose torch result is normal: the flushed edge is the pipeline's, not the op's."""
    mask = torch.isfinite(source) & (source.abs() > 2 * SMALLEST_NORMAL_BF16)
    return mask & torch.isfinite(golden) & (golden.abs() > 2 * SMALLEST_NORMAL_BF16)


@pytest.mark.parametrize("op_type, ttnn_op, torch_op", CALLERS, ids=IDS)
def test_log1p_init_caller_alone(device, op_type, ttnn_op, torch_op):
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)
    result = ttnn.to_torch(ttnn_op(to_tt_tensor(input_tensor, device)))
    golden = torch_op(input_tensor.float()).to(torch.bfloat16)
    checked = _checked(input_tensor, golden)
    assert checked.sum() > 1024
    assert_with_ulp(expected_result=golden[checked], actual_result=result[checked], ulp_threshold=2)


@pytest.mark.parametrize("op_type, ttnn_op, torch_op", CALLERS, ids=IDS)
def test_log1p_init_caller_after_log1p_in_one_chain(device, op_type, ttnn_op, torch_op):
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)
    tt_in = to_tt_tensor(input_tensor, device)
    log1p = ttnn.UnaryWithParam(ttnn.UnaryOpType.LOG1P)
    middle = ttnn.to_torch(ttnn.unary_chain(tt_in, [log1p]))
    result = ttnn.to_torch(ttnn.unary_chain(tt_in, [log1p, ttnn.UnaryWithParam(op_type)]))
    # The chain keeps log1p's device result in DEST, so the reference starts from it.
    golden = torch_op(middle.float()).to(torch.bfloat16)
    checked = _checked(middle, golden)
    assert checked.sum() > 1024
    assert_with_ulp(expected_result=golden[checked], actual_result=result[checked], ulp_threshold=2)
