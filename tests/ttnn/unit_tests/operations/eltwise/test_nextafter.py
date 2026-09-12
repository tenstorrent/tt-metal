# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_ulp

# nextafter moves its first operand by a single ULP, so a PCC comparison against the golden
# stays at ~1.0 even when the op returns the input untouched. Every assertion here is exact.

# The SFPU flushes subnormals to zero, so results in the subnormal range are out of scope.
SMALLEST_NORMAL = 1.1754943508222875e-38


def _all_bfloat16_values():
    """Every bfloat16 bit pattern, as float32. Pattern p widens to the float32 pattern p << 16."""
    patterns = torch.arange(1 << 16, dtype=torch.int64) << 16
    signed = torch.where(patterns >= 2**31, patterns - 2**32, patterns).to(torch.int32)
    return signed.view(torch.float32)


def _in_scope(a, expected):
    """Drop elements the SFPU cannot represent: subnormal (flushed to zero) and non-finite."""
    mask = (a.abs().float() >= SMALLEST_NORMAL) & (expected.abs().float() >= SMALLEST_NORMAL)
    return mask & torch.isfinite(a) & torch.isfinite(expected)


@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 320, 384)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_nextafter(device, shape, dtype):
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    torch_input_tensor_a = torch.rand(shape, dtype=torch_dtype) * 200 - 100
    torch_input_tensor_b = torch.rand(shape, dtype=torch_dtype) * 300 - 150

    torch_output_tensor = torch.nextafter(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.nextafter(input_tensor_a, input_tensor_b))

    in_scope = _in_scope(torch_input_tensor_a, torch_output_tensor)
    assert_with_ulp(
        expected_result=torch_output_tensor[in_scope], actual_result=output_tensor[in_scope], ulp_threshold=0
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_nextafter_direction_and_equality(device, dtype):
    """Stepping up, stepping down, and the a == b case that must return b unchanged."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    values = torch.tensor([1.0, -1.0, 3.5, -3.5, 1e30, -1e30, 1e-30, -1e-30], dtype=torch_dtype)
    a = values.repeat(3).reshape(1, 1, 1, -1).expand(1, 1, 32, 24).contiguous()
    b = torch.cat([values + 10.0, values - 10.0, values]).reshape(1, 1, 1, -1).expand(1, 1, 32, 24).contiguous()

    expected = torch.nextafter(a, b)

    ttnn_a = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.nextafter(ttnn_a, ttnn_b))

    in_scope = _in_scope(a, expected)
    assert_with_ulp(expected_result=expected[in_scope], actual_result=actual[in_scope], ulp_threshold=0)


def test_nextafter_signed_zeros(device):
    """Zeros take the sign of the target, including nextafter(+0, -0) == -0.

    The results here are zeros rather than the true subnormal neighbours, which the SFPU flushes,
    so only the sign is checked. Signs are compared through copysign because 0.0 == -0.0.

    float32 only: a bfloat16 tile does not carry a negative zero through the compute path at all,
    independently of this op. ttnn.neg on a bfloat16 +0.0 returns +0.0, where the same call on
    float32 returns -0.0, so there is no sign for this op to preserve in that format.
    """
    dtype, torch_dtype = ttnn.float32, torch.float32

    zeros = torch.tensor([0.0, -0.0, 0.0, -0.0], dtype=torch_dtype)
    targets = torch.tensor([-0.0, 0.0, -1.0, 1.0], dtype=torch_dtype)
    a = zeros.repeat(8).reshape(1, 1, 1, -1).expand(1, 1, 32, 32).contiguous()
    b = targets.repeat(8).reshape(1, 1, 1, -1).expand(1, 1, 32, 32).contiguous()

    ttnn_a = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.nextafter(ttnn_a, ttnn_b))

    assert torch.equal(actual.abs(), torch.zeros_like(actual)), f"expected zeros, got {actual.unique()}"
    expected_sign = torch.copysign(torch.ones_like(b), b)
    actual_sign = torch.copysign(torch.ones_like(actual), actual)
    assert torch.equal(actual_sign, expected_sign), "zero results must carry the target's sign"


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_nextafter_rejects_block_float(device, dtype, expect_error):
    """A bfloat16 ULP is not the next representable block-float value, so these are not accepted.

    Unrestricted, the step rounded away when the tile was packed against its shared exponent and
    the op returned its input: bfloat4_b was a no-op for every element.
    """
    a = torch.rand((1, 1, 32, 32), dtype=torch.float32) * 200 - 100
    ttnn_a = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(a + 50.0, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "is not supported for binary operation BinaryOpType::NEXTAFTER"):
        ttnn.nextafter(ttnn_a, ttnn_b)


@pytest.mark.parametrize("toward", ["up", "down"])
def test_nextafter_exhaustive_bfloat16(device, toward):
    """Walk every normal bfloat16 value one ULP up and one ULP down."""
    limit = torch.finfo(torch.bfloat16).max
    a = _all_bfloat16_values().reshape(1, 1, 256, 256).to(torch.bfloat16)
    b = torch.full_like(a, limit if toward == "up" else -limit)
    expected = torch.nextafter(a, b)

    ttnn_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.nextafter(ttnn_a, ttnn_b))

    # Of the 65536 patterns, 256 are inf/NaN and 256 are zero or subnormal; one more drops out
    # at the end of the ladder, where the last normal steps to infinity.
    in_scope = _in_scope(a, expected)
    assert in_scope.sum() == 65023, f"expected 65023 normal patterns in scope, got {in_scope.sum()}"

    assert_with_ulp(expected_result=expected[in_scope], actual_result=actual[in_scope], ulp_threshold=0)
