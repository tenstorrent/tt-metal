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
MIXED_DTYPE_ERROR = r"Mixed dtype is not supported for binary operation"
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


def _make_mixed_binary_tensors(device, dtype_a, dtype_b, shape=TENSOR_SHAPE):
    torch_dtype_a = tt_dtype_to_torch_dtype[dtype_a]
    torch_dtype_b = tt_dtype_to_torch_dtype[dtype_b]
    torch_a = torch.ones(shape, dtype=torch_dtype_a)
    torch_b = torch.ones(shape, dtype=torch_dtype_b) * 2

    tensor_a = ttnn.from_torch(
        torch_a,
        dtype=dtype_a,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tensor_b = ttnn.from_torch(
        torch_b,
        dtype=dtype_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tensor_a, tensor_b


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
        pytest.param(ttnn.gcd, ttnn.bfloat16, id="int32_only_gcd_bfloat16"),
        pytest.param(ttnn.lcm, ttnn.bfloat16, id="int32_only_lcm_bfloat16"),
        # exp_dependent
        pytest.param(ttnn.logaddexp, ttnn.int32, id="exp_dependent_logaddexp_int32"),
        pytest.param(ttnn.logaddexp2, ttnn.uint32, id="exp_dependent_logaddexp2_uint32"),
        pytest.param(ttnn.ldexp, ttnn.int32, id="exp_dependent_ldexp_int32"),
        pytest.param(ttnn.bias_gelu, ttnn.uint16, id="exp_dependent_bias_gelu_uint16"),
        # float_only (POWER, XLOGY, ATAN2, HYPOT, QUANT)
        pytest.param(ttnn.pow, ttnn.int32, id="float_only_pow_int32"),
        pytest.param(ttnn.pow, ttnn.uint32, id="float_only_pow_uint32"),
        pytest.param(ttnn.xlogy, ttnn.uint32, id="float_only_xlogy_uint32"),
        pytest.param(ttnn.atan2, ttnn.int32, id="float_only_atan2_int32"),
        pytest.param(ttnn.hypot, ttnn.uint16, id="float_only_hypot_uint16"),
        pytest.param(ttnn.hypot, ttnn.int32, id="float_only_hypot_int32"),
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


# Mixed bfloat_tile dtypes (BFLOAT16 / BFLOAT8_B / BFLOAT4_B): FPU binary_ng path only.
@pytest.mark.parametrize(
    "op, dtype_a, dtype_b",
    [
        pytest.param(ttnn.add, ttnn.bfloat8_b, ttnn.bfloat16, id="add_bf8_bfloat16"),
        pytest.param(ttnn.add, ttnn.bfloat16, ttnn.bfloat8_b, id="add_bfloat16_bf8"),
        pytest.param(ttnn.subtract, ttnn.bfloat8_b, ttnn.bfloat16, id="subtract_bf8_bfloat16"),
        pytest.param(ttnn.mul, ttnn.bfloat16, ttnn.bfloat8_b, id="mul_bfloat16_bf8"),
        pytest.param(ttnn.eq, ttnn.bfloat8_b, ttnn.bfloat16, id="eq_bf8_bfloat16"),
        pytest.param(ttnn.div, ttnn.bfloat16, ttnn.bfloat8_b, id="div_bfloat16_bf8"),
    ],
)
def test_binary_mixed_bfloat_tile_allowed(device, op, dtype_a, dtype_b):
    tensor_a, tensor_b = _make_mixed_binary_tensors(device, dtype_a, dtype_b)
    op(tensor_a, tensor_b)


# Mixed-dtype pairs where each dtype is individually supported for the op.
@pytest.mark.parametrize(
    "op, dtype_a, dtype_b",
    [
        # arithmetic_fpu: int32 + uint32
        pytest.param(ttnn.add, ttnn.int32, ttnn.uint32, id="add_int32_uint32"),
        pytest.param(ttnn.mul, ttnn.int32, ttnn.uint32, id="mul_int32_uint32"),
        pytest.param(ttnn.subtract, ttnn.uint32, ttnn.int32, id="subtract_uint32_int32"),
        # arithmetic_fpu: int32 + uint16
        pytest.param(ttnn.add, ttnn.int32, ttnn.uint16, id="add_int32_uint16"),
        pytest.param(ttnn.mul, ttnn.uint16, ttnn.int32, id="mul_uint16_int32"),
        # arithmetic_fpu: uint32 + uint16
        pytest.param(ttnn.add, ttnn.uint32, ttnn.uint16, id="add_uint32_uint16"),
        # arithmetic_fpu: float + int
        pytest.param(ttnn.add, ttnn.bfloat16, ttnn.int32, id="add_bfloat16_int32"),
        pytest.param(ttnn.add, ttnn.float32, ttnn.bfloat16, id="add_float32_bfloat16"),
        pytest.param(ttnn.mul, ttnn.float32, ttnn.bfloat16, id="mul_float32_bfloat16"),
        pytest.param(ttnn.mul, ttnn.bfloat16, ttnn.float32, id="mul_bfloat16_float32"),
        pytest.param(ttnn.pow, ttnn.float32, ttnn.bfloat16, id="pow_float32_bfloat16"),
        pytest.param(ttnn.pow, ttnn.bfloat16, ttnn.float32, id="pow_bfloat16_float32"),
        pytest.param(ttnn.pow, ttnn.bfloat16, ttnn.bfloat8_b, id="pow_bfloat16_bf8"),
        pytest.param(ttnn.maximum, ttnn.bfloat16, ttnn.bfloat8_b, id="maximum_bfloat16_bf8"),
        # float_and_int32
        pytest.param(ttnn.div, ttnn.int32, ttnn.float32, id="div_int32_float32"),
        # maximum_minimum
        pytest.param(ttnn.maximum, ttnn.int32, ttnn.uint32, id="maximum_int32_uint32"),
        pytest.param(ttnn.minimum, ttnn.uint32, ttnn.int32, id="minimum_uint32_int32"),
        # bitwise_shift
        pytest.param(ttnn.bitwise_left_shift, ttnn.int32, ttnn.uint32, id="bitwise_left_shift_int32_uint32"),
        pytest.param(ttnn.bitwise_right_shift, ttnn.uint32, ttnn.uint16, id="bitwise_right_shift_uint32_uint16"),
        # logical_right_shift
        pytest.param(ttnn.logical_right_shift, ttnn.int32, ttnn.uint32, id="logical_right_shift_int32_uint32"),
    ],
)
def test_binary_mixed_dtype_rejected(device, op, dtype_a, dtype_b):
    """Operands may each use a supported dtype, but both must match."""
    tensor_a, tensor_b = _make_mixed_binary_tensors(device, dtype_a, dtype_b)
    with pytest.raises(RuntimeError, match=MIXED_DTYPE_ERROR):
        op(tensor_a, tensor_b)
