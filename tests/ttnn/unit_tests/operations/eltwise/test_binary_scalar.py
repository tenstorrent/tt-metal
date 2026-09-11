# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import random

from tests.ttnn.utils_for_testing import assert_with_ulp

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    (
        (ttnn.gt),
        (ttnn.lt),
        (ttnn.ne),
        (ttnn.ge),
        (ttnn.le),
        (ttnn.eq),
    ),
)
def test_binary_scalar_ops(input_shapes, device, ttnn_fn):
    torch.manual_seed(0)
    torch_input = torch.randn(input_shapes, dtype=torch.bfloat16) * 100
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.zeros_like(input_tensor)
    scalar = random.randint(-80, 80)
    ttnn_fn(input_tensor, scalar, output_tensor=output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    golden_tensor = golden_fn(torch_input, scalar)

    out = ttnn.to_torch(output_tensor).to(torch.bool)

    assert torch.equal(out, golden_tensor)


@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
@pytest.mark.parametrize(
    "scalar",
    [
        7,
        -13,
        0,
        -1,
        2,
        10000,
    ],
)
def test_binary_scalar_int32_arithmetic(device, op_name, scalar):
    """Verify int32 tensor + int scalar passes the scalar as int32 (not float)."""
    ttnn_fn = getattr(ttnn, op_name)
    torch_fn = getattr(torch, op_name)
    torch_input = torch.tensor(
        [
            1,
            -1,
            0,
            2147483640,
            2147483647,
            -2147483647,
            -2147483648,
            1000,
            -1000,
            42,
            123456789,
            -123456789,
            500,
            -500,
            999,
            -999,
            77,
            -77,
            2,
            -2,
            10,
            -10,
            100,
            -100,
            7,
            9,
            11,
            15,
        ],
        dtype=torch.int32,
    )
    expected = torch_fn(torch_input, scalar)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn_fn(tt_input, scalar)
    result = ttnn.to_torch(tt_output)

    assert torch.equal(expected, result)


@pytest.mark.parametrize("op_name", ["add", "sub"])
@pytest.mark.parametrize("scalar", [0, 1, 100, 65535])
def test_binary_scalar_uint32_arithmetic(device, op_name, scalar):
    """Verify uint32 tensor + int scalar near and at uint32 max boundary."""
    ttnn_fn = getattr(ttnn, op_name)
    torch_fn = getattr(torch, op_name)
    torch_input = torch.tensor(
        [
            0,
            1,
            2,
            255,
            65535,
            100000,
            2147483647,
            2147483648,
            3000000000,
            4000000000,
            4294967290,
            4294967291,
            4294967294,
            4294967295,
            16777215,
            16777216,
            16777217,
            500,
            1000,
            10000,
            1000000,
            1000000000,
            2500000000,
            3500000000,
            3999999999,
            4294000000,
            4294900000,
            4294960000,
            4294967000,
            4294967200,
        ],
    )
    expected = torch_fn(torch_input, scalar)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_output = ttnn_fn(tt_input, scalar)

    expected_tt = ttnn.from_torch(
        expected,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    comparison = ttnn.eq(tt_output, expected_tt)
    comparison_torch = ttnn.to_torch(comparison)
    assert torch.all(comparison_torch), "Mismatch in uint32 scalar arithmetic"


@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
@pytest.mark.parametrize("scalar", [1.5, -2.25, 0.0, 100.0])
def test_binary_scalar_float32_arithmetic(device, op_name, scalar):
    """Verify float32 tensor + float scalar still works correctly."""
    ttnn_fn = getattr(ttnn, op_name)
    torch_fn = getattr(torch, op_name)
    torch.manual_seed(42)
    torch_input = torch.randn([1, 1, 32, 32], dtype=torch.float32)
    expected = torch_fn(torch_input, scalar)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn_fn(tt_input, scalar)
    result = ttnn.to_torch(tt_output)

    assert torch.allclose(expected, result, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "ttnn_fn",
    [ttnn.eq, ttnn.ne, ttnn.gt, ttnn.lt, ttnn.ge, ttnn.le],
)
@pytest.mark.parametrize("scalar", [0, 1, -1, 42, -100])
def test_binary_scalar_int32_relational(device, ttnn_fn, scalar):
    """Verify relational ops with int32 tensor and int scalar."""
    torch_input = torch.tensor(
        [
            -100,
            42,
            -1,
            0,
            1,
            200,
            -200,
            -50,
            -10,
            2147483640,
            2147483647,
            -2147483647,
            -2147483648,
            300,
            -300,
            -150,
            -5,
        ],
        dtype=torch.int32,
    )

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    expected = golden_fn(torch_input, scalar)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn_fn(tt_input, scalar)
    result = ttnn.to_torch(tt_output).to(torch.bool)

    assert torch.equal(expected, result)


@pytest.mark.parametrize(
    "scalar",
    [
        16777217,
        16777366,
        2147483640,
        2147483647,
        -2147483647,
        -2147483540,
        -2147483648,
    ],
)
def test_binary_scalar_int32_large_values(scalar, device):
    """Verify that large int32 scalars are not corrupted by float conversion.

    Values > 2^24 cannot be represented exactly in float32.  With ScalarVariant
    they should be packed as int32 directly and arrive on the device unchanged.
    """

    torch_input = torch.ones([1, 1, 32, 32], dtype=torch.int32)
    expected = torch.add(torch_input, scalar)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.add(tt_input, scalar)
    result = ttnn.to_torch(tt_output)

    assert torch.equal(expected, result), (
        f"Large scalar {scalar} was likely truncated to float. "
        f"Expected {expected.flatten()[0].item()}, got {result.flatten()[0].item()}"
    )


@pytest.mark.parametrize(
    "scalar",
    [
        16777217,
        16777366,
        2147483640,
        2147483647,
        4294967200,
        4294967294,
    ],
)
def test_binary_scalar_uint32_large_values(scalar, device):
    """Verify that large uint32 scalars are not corrupted by float conversion.

    Values > 2^24 cannot be represented exactly in float32.  With ScalarVariant
    they should be packed as uint32 directly and arrive on the device unchanged.
    """

    torch_input = torch.ones([1, 1, 32, 32], dtype=torch.int64)
    expected = torch.add(torch_input, scalar)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.add(tt_input, scalar)
    result = ttnn.to_torch(tt_output, dtype=torch.int64)

    assert torch.equal(expected, result), (
        f"Large scalar {scalar} was likely truncated to float. "
        f"Expected {expected.flatten()[0].item()}, got {result.flatten()[0].item()}"
    )


@pytest.mark.parametrize("tensor_dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize("scalar", [2.5, 0.5, -1.5])
@pytest.mark.parametrize("ttnn_op", [ttnn.multiply, ttnn.div])
def test_int_tensor_float_scalar_promotes(device, ttnn_op, tensor_dtype, scalar):
    # The scalar is packed using the tensor's dtype, so without promotion 2.5 arrives as 2 and 0.5
    # as 0 -- div(int32, 0.5) used to return inf instead of 14. mul/div promote, matching both torch
    # and what the tensor-tensor path already does for a mixed int/float pair.
    torch_input = torch.tensor([[7, 6, 12, 100]], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_op(a, scalar)
    torch_golden = (torch_input.float() * scalar) if ttnn_op is ttnn.multiply else (torch_input.float() / scalar)

    assert output.dtype == ttnn.float32
    assert_with_ulp(expected_result=torch_golden, actual_result=output, ulp_threshold=1)


@pytest.mark.parametrize("tensor_dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize("ttnn_op", [ttnn.add, ttnn.subtract])
def test_int_tensor_fractional_scalar_rejected(device, ttnn_op, tensor_dtype, expect_error):
    # add/subtract reject a mixed int/float tensor pair rather than promoting, so a scalar they
    # cannot represent is rejected too instead of being silently truncated.
    torch_input = torch.tensor([[7, 6, 12, 100]], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "cannot represent the scalar"):
        ttnn_op(a, 2.5)


@pytest.mark.parametrize("tensor_dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize("ttnn_op", [ttnn.add, ttnn.subtract, ttnn.multiply])
def test_int_tensor_integral_float_scalar_stays_exact(device, ttnn_op, tensor_dtype):
    # float32 carries a 24-bit mantissa, so promoting the tensor would cap exact integers at 2^24 --
    # 16777217 * 2.0 would come back as 33554432 rather than 33554434. An integral scalar reaches the
    # kernel intact on the integer path, so it stays there and keeps these exact.
    torch_input = torch.tensor([[2**24 - 1, 2**24, 2**24 + 1, 2**24 + 3]], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_op(a, 2.0)
    torch_golden = {
        ttnn.add: torch_input + 2,
        ttnn.subtract: torch_input - 2,
        ttnn.multiply: torch_input * 2,
    }[ttnn_op]

    # compared as integers: routing these through float32 would silently round them
    assert output.dtype == tensor_dtype
    assert ttnn.to_torch(output).flatten().tolist() == torch_golden.flatten().tolist()


@pytest.mark.parametrize("tensor_dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize("ttnn_op", [ttnn.add, ttnn.subtract, ttnn.multiply])
def test_int_tensor_integer_scalar_unchanged(device, ttnn_op, tensor_dtype):
    # An integer scalar loses nothing in the pack, so it must keep the integer dtype and value.
    torch_input = torch.tensor([[7, 6, 12, 100]], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_op(a, 2)
    torch_golden = {
        ttnn.add: torch_input + 2,
        ttnn.subtract: torch_input - 2,
        ttnn.multiply: torch_input * 2,
    }[ttnn_op]

    assert output.dtype == tensor_dtype
    assert torch.equal(ttnn.to_torch(output).float(), torch_golden.float())


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
@pytest.mark.parametrize("scalar", [2.5, 0.5, -1.5, 3e9])
def test_int_tensor_float_scalar_rounded_division(device, rounding_mode, scalar):
    # The DIV_FLOOR/DIV_TRUNC kernels are int32-only and take the divisor through the int32 scalar
    # packing, so a divisor they cannot carry has to be divided in floating point and rounded after.
    # These used to be rejected outright even though div(rounding_mode=None) accepted them. 3e9 is
    # the integral-but-out-of-range case, which the fractional check alone would have missed.
    torch_input = torch.tensor([[-13, -7, 6, 100]], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.div(a, scalar, rounding_mode=rounding_mode)
    quotient = torch_input.float() / scalar
    torch_golden = torch.floor(quotient) if rounding_mode == "floor" else torch.trunc(quotient)

    # Negative numerators are in the input on purpose: the quotient has to be rounded while it is
    # still floating point, or floor(-13/2.5) comes back as -5 instead of -6.
    assert output.dtype == ttnn.float32
    assert_with_ulp(expected_result=torch_golden, actual_result=output, ulp_threshold=1)


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
@pytest.mark.parametrize("scalar, expected_dtype", [(2, ttnn.int32), (2.0, ttnn.float32)])
def test_int_tensor_rounded_division_dtype_follows_scalar_type(device, rounding_mode, scalar, expected_dtype):
    # ttnn.div decides from the divisor's type rather than its value: an integer 2 keeps exact int32
    # division, while 2.0 promotes the tensor and divides in floating point. Worth pinning because it
    # is the opposite of what multiply does -- multiply(int32, 2.0) stays int32, since an integral
    # scalar reaches the kernel intact and promoting would cap exact integers at 2**24.
    torch_input = torch.tensor([[-13, -7, 6, 100]], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.div(a, scalar, rounding_mode=rounding_mode)
    quotient = torch_input.float() / 2
    torch_golden = torch.floor(quotient) if rounding_mode == "floor" else torch.trunc(quotient)

    assert output.dtype == expected_dtype
    assert ttnn.to_torch(output).float().flatten().tolist() == torch_golden.flatten().tolist()


@pytest.mark.parametrize(
    "tensor_dtype, scalar",
    [
        # Integral, so the old fractional-only check let these through to an undefined
        # float-to-integer cast: inf arrived as INT32_MIN and 2**32 as 0.
        (ttnn.int32, 2.0**31),  # one past INT32_MAX
        (ttnn.int32, -(2.0**31) - 2048),  # one representable step below INT32_MIN
        (ttnn.int32, float("inf")),
        (ttnn.int32, float("-inf")),
        (ttnn.uint32, -3.0),  # negative against an unsigned tensor
        (ttnn.uint32, 2.0**32),  # one past UINT32_MAX
        (ttnn.uint32, float("inf")),
    ],
)
@pytest.mark.parametrize("ttnn_op", [ttnn.add, ttnn.subtract])
def test_int_tensor_unrepresentable_scalar_rejected(device, ttnn_op, tensor_dtype, scalar, expect_error):
    # Being integral is not enough: the value also has to be finite and inside the tensor dtype's
    # range, or the cast in pack_scalar_runtime_arg is undefined and silently changes it.
    torch_input = torch.tensor([[7, 6, 12, 100]], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "cannot represent the scalar"):
        ttnn_op(a, scalar)


@pytest.mark.parametrize(
    "tensor_dtype, scalar",
    [
        (ttnn.int32, -(2.0**31)),  # INT32_MIN, must survive the lower bound
        (ttnn.int32, 2147483520.0),  # largest float32 below INT32_MAX
        (ttnn.uint32, 0.0),  # lower bound for an unsigned tensor
        (ttnn.uint32, 4294965248.0),  # largest float32 below UINT32_MAX
    ],
)
def test_int_tensor_boundary_scalar_accepted(device, tensor_dtype, scalar):
    # The limits themselves have to stay on the integer path, since rejecting them would be as wrong
    # as accepting the values past them. INT32_MAX and UINT32_MAX are not float32 values, so the
    # check compares against powers of two and these are the largest floats below each limit.
    # Added to zero so the sum itself cannot overflow the dtype and confuse the result.
    torch_input = torch.zeros([1, 32], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.add(a, scalar)

    assert output.dtype == tensor_dtype
    assert ttnn.to_torch(output, dtype=torch.int64).flatten().tolist() == [int(scalar)] * torch_input.numel()
