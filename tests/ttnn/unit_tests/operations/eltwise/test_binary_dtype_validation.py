# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools

import torch
import pytest
import ttnn

from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype

pytestmark = pytest.mark.use_module_device

TENSOR_SHAPE = (1, 1, 32, 32)
UNSUPPORTED_DTYPE_ERROR = r"not supported for binary operation"
_ISCLOSE = functools.partial(ttnn.isclose, rtol=1e-5, atol=1e-8, equal_nan=False)


def _make_binary_tensors(device, dtype, shape=TENSOR_SHAPE):
    torch_dtype = tt_dtype_to_torch_dtype[dtype]
    torch_a = torch.ones(shape, dtype=torch_dtype)
    torch_b = torch.ones(shape, dtype=torch_dtype) * 2

    tensor_a = ttnn.from_torch(
        torch_a,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tensor_b = ttnn.from_torch(
        torch_b,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tensor_a, tensor_b


# High-value rejection cases
@pytest.mark.parametrize(
    "op, dtype",
    [
        pytest.param(ttnn.add, ttnn.uint8, id="add_uint8"),
        pytest.param(ttnn.remainder, ttnn.uint16, id="remainder_uint16"),
        pytest.param(ttnn.bitwise_and, ttnn.bfloat16, id="bitwise_and_bfloat16"),
        pytest.param(ttnn.logical_right_shift, ttnn.uint16, id="logical_right_shift_uint16"),
        pytest.param(ttnn.gcd, ttnn.uint8, id="gcd_uint8"),
        pytest.param(ttnn.gcd, ttnn.bfloat16, id="gcd_bfloat16"),
        pytest.param(ttnn.gcd, ttnn.uint16, id="gcd_uint16"),
        pytest.param(ttnn.logaddexp, ttnn.int32, id="logaddexp_int32"),
        pytest.param(ttnn.hypot, ttnn.int32, id="hypot_int32"),
    ],
)
def test_binary_dtype_validation_high_value_rejections(device, op, dtype):
    """Smoke negatives aligned with binary_op_dtype_policy groups."""
    tensor_a, tensor_b = _make_binary_tensors(device, dtype)
    with pytest.raises(RuntimeError, match=UNSUPPORTED_DTYPE_ERROR):
        op(tensor_a, tensor_b)


# Parametric matrix: representative ops per dtype_sets bucket in binary_op_dtype_policy.cpp.
@pytest.mark.parametrize(
    "op, dtype",
    [
        # arithmetic_fpu
        pytest.param(ttnn.add, ttnn.uint8, id="arithmetic_fpu_add_uint8"),
        pytest.param(ttnn.mul, ttnn.uint8, id="arithmetic_fpu_mul_uint8"),
        pytest.param(ttnn.subtract, ttnn.uint8, id="arithmetic_fpu_subtract_uint8"),
        # int32_only (GCD, LCM, DIV_FLOOR, DIV_TRUNC)
        pytest.param(ttnn.gcd, ttnn.uint8, id="int32_only_gcd_uint8"),
        pytest.param(ttnn.gcd, ttnn.uint16, id="int32_only_gcd_uint16"),
        pytest.param(ttnn.lcm, ttnn.bfloat16, id="int32_only_lcm_bfloat16"),
        # exp_dependent
        pytest.param(ttnn.logaddexp, ttnn.int32, id="exp_dependent_logaddexp_int32"),
        pytest.param(ttnn.logaddexp2, ttnn.uint32, id="exp_dependent_logaddexp2_uint32"),
        pytest.param(ttnn.ldexp, ttnn.int32, id="exp_dependent_ldexp_int32"),
        pytest.param(ttnn.bias_gelu, ttnn.uint16, id="exp_dependent_bias_gelu_uint16"),
        # float_only (POWER, XLOGY, ATAN2, HYPOT, QUANT)
        pytest.param(ttnn.pow, ttnn.int32, id="float_only_pow_int32"),
        pytest.param(ttnn.xlogy, ttnn.uint32, id="float_only_xlogy_uint32"),
        pytest.param(ttnn.atan2, ttnn.int32, id="float_only_atan2_int32"),
        pytest.param(ttnn.hypot, ttnn.uint16, id="float_only_hypot_uint16"),
        # div_remainder_fmod
        pytest.param(ttnn.div, ttnn.uint16, id="div_remainder_div_uint16"),
        pytest.param(ttnn.remainder, ttnn.uint16, id="div_remainder_remainder_uint16"),
        pytest.param(ttnn.fmod, ttnn.uint32, id="div_remainder_fmod_uint32"),
        # bitwise_shift
        pytest.param(ttnn.bitwise_and, ttnn.bfloat16, id="bitwise_bitwise_and_bfloat16"),
        pytest.param(ttnn.bitwise_or, ttnn.float32, id="bitwise_bitwise_or_float32"),
        pytest.param(ttnn.bitwise_left_shift, ttnn.bfloat16, id="bitwise_bitwise_left_shift_bfloat16"),
        pytest.param(ttnn.bitwise_right_shift, ttnn.bfloat16, id="bitwise_bitwise_right_shift_bfloat16"),
        # maximum_minimum
        pytest.param(ttnn.maximum, ttnn.uint16, id="max_min_maximum_uint16"),
        pytest.param(ttnn.minimum, ttnn.uint8, id="max_min_minimum_uint8"),
        # logical_right_shift
        pytest.param(ttnn.logical_right_shift, ttnn.uint16, id="logical_right_shift_uint16"),
        # isclose (float + int32 via typecast; unsigned ints not supported)
        pytest.param(_ISCLOSE, ttnn.uint16, id="isclose_uint16"),
        pytest.param(_ISCLOSE, ttnn.uint32, id="isclose_uint32"),
    ],
)
def test_binary_unsupported_input_dtype_rejected(device, op, dtype):
    tensor_a, tensor_b = _make_binary_tensors(device, dtype)
    with pytest.raises(RuntimeError, match=UNSUPPORTED_DTYPE_ERROR):
        op(tensor_a, tensor_b)
