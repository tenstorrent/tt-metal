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
    # The target has to scale with the operand rather than sit a fixed distance away. One ULP of
    # 1e30 is about 2**76 in float32, so a fixed +/- 10.0 is absorbed entirely and those columns
    # would silently retest the equality case instead of a step -- which is the large-exponent
    # regime the fixed-epsilon bug this op had lived in.
    b = torch.cat([values * 2, values * 0.5, values]).reshape(1, 1, 1, -1).expand(1, 1, 32, 24).contiguous()

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


def test_nextafter_nan_propagates(device):
    """A NaN in either operand must come back as NaN, not as a stepped finite value.

    float32 only: a bfloat16 tile does not carry a NaN through the compute path at all, and that
    is not specific to this op -- ttnn.multiply of a bfloat16 NaN by 1.0 also returns infinity.
    The kernel classifies NaN on the integer pattern because SFPSETCC is unspecified for a NaN
    comparand: before the guard, nextafter(1.0, NaN) stepped and returned 1.0000001.
    """
    nan, inf = float("nan"), float("inf")
    a = torch.tensor([[1.0, 2.0, nan, nan, -3.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    b = torch.tensor([[nan, nan, 1.0, nan, nan, nan, inf, 2.0]], dtype=torch.float32)

    ttnn_a = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.nextafter(ttnn_a, ttnn_b))[0, : a.numel()]
    expected = torch.nextafter(a, b)[0]

    assert torch.equal(actual.isnan(), expected.isnan()), f"NaN pattern differs: {actual}"
    finite = ~expected.isnan()
    assert torch.equal(actual[finite], expected[finite]), f"finite lanes differ: {actual[finite]}"


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((1, 1, 32, 32), (1, 1, 32, 1)),
        ((1, 1, 32, 32), (1, 1, 1, 32)),
        ((1, 1, 32, 32), (1, 1, 1, 1)),
        # a on the broadcast side: get_subtile_broadcast_type only reaches the A-side and the mixed
        # row/col types when a_h or a_w is 1, so a full-shape `a` cannot select ComputeRowColBcastNg
        # at all. That leaves eltwise_binary_sfpu_row_col_bcast.cpp -- one of the four bcast kernels
        # this op is wired into -- with no coverage. It matters here because nextafter is not
        # commutative and those kernels reassign CB roles per direction.
        ((1, 1, 32, 1), (1, 1, 1, 32)),
        ((1, 1, 1, 32), (1, 1, 32, 1)),
        ((1, 1, 1, 1), (1, 1, 32, 32)),
    ],
    ids=["col_b", "row_b", "scalar_b", "row_a_col_b", "col_a_row_b", "scalar_a"],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_nextafter_broadcast(device, shape_a, shape_b, dtype):
    """The broadcast kernels are selected purely by shape, so same-shape tests never reach them.

    The broadcast operand takes a pack/unpack round trip through an L1 CB before the SFPU op, and
    only the first operand is bit-stepped, so this still has to be exact.
    """
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    a = torch.rand(shape_a, dtype=torch_dtype) * 200 - 100
    b = torch.rand(shape_b, dtype=torch_dtype) * 300 - 150

    # The ttnn operands keep their original shapes -- expanding them here would make both full and
    # select no broadcast kernel at all. Only the golden is broadcast.
    broadcast = torch.broadcast_shapes(shape_a, shape_b)
    expected = torch.nextafter(a.expand(broadcast), b.expand(broadcast))

    ttnn_a = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.nextafter(ttnn_a, ttnn_b))

    in_scope = _in_scope(a.expand(broadcast), expected)
    assert_with_ulp(expected_result=expected[in_scope], actual_result=actual[in_scope], ulp_threshold=0)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_nextafter_steps_finite_max_to_infinity(device, dtype):
    """The one input where the integer step rolls the exponent into infinity.

    For bfloat16 that is 0x7F7F0000 + 0x10000 == 0x7F800000. The exhaustive walk caps its target at
    the largest finite bfloat16 and _in_scope drops non-finite lanes, so nothing else asserts the
    overflow boundary, and the signed-zero and NaN tests are float32-only -- this is the only
    bfloat16 lane in the file that checks a guard rather than a step.
    """
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    max_finite = torch.finfo(torch_dtype).max
    inf = float("inf")

    a = torch.tensor([[max_finite, -max_finite, max_finite, -max_finite]], dtype=torch_dtype)
    b = torch.tensor([[inf, -inf, -inf, inf]], dtype=torch_dtype)
    expected = torch.nextafter(a, b)
    assert expected[0, 0].isinf() and expected[0, 1].isinf(), "first two lanes must overflow"

    ttnn_a = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.nextafter(ttnn_a, ttnn_b))[:, : a.shape[1]]

    assert_with_ulp(expected_result=expected, actual_result=actual, ulp_threshold=0, allow_nonfinite=True)


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

    # Of the 65536 patterns, 256 are inf/NaN and 256 are zero or subnormal. One more drops out at
    # each end: +/-SMALLEST_NORMAL stepping toward zero lands on the largest subnormal, which the
    # SFPU flushes. Nothing steps to infinity here, since the target is capped at the largest
    # finite bfloat16 and nextafter never overshoots it.
    in_scope = _in_scope(a, expected)
    assert in_scope.sum() == 65023, f"expected 65023 normal patterns in scope, got {in_scope.sum()}"

    assert_with_ulp(expected_result=expected[in_scope], actual_result=actual[in_scope], ulp_threshold=0)
