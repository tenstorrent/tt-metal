# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Regression coverage for ttnn.experimental.quasar.nextafter.

The quasar binding carried a copy of the composite implementation that ttnn.nextafter used before
#56050 replaced it with the SFPU kernel, and that copy had two independent defects:

  * it stepped by tt::tt_metal::hal::get_eps(), a fixed FLT_EPSILON, rather than by one ULP of the
    operand. FLT_EPSILON is one ULP only on [1, 2); for bfloat16 it is smaller than half a ULP
    everywhere, so the step was absorbed and the op returned its input for every input; and
  * the two arms were the wrong way round: a > b added epsilon and a < b subtracted it, moving the
    result away from the target instead of towards it.

Both are exercised here. A PCC comparison cannot see either one -- a single-ULP move leaves PCC at
~1.0 even when the op returns the input untouched -- so every assertion below is exact.

These run wherever the NEXTAFTER SFPU primitive exists, which today is Wormhole and Blackhole. The
Quasar LLK does not have it yet (tt_llk_quasar/common/inc/ckernel_defs.h, enum BinaryOp).

Run on Wormhole:
    pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_nextafter.py
"""

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_ulp

# The SFPU flushes subnormals to zero, so results in the subnormal range are out of scope.
SMALLEST_NORMAL = 1.1754943508222875e-38


def _in_scope(a, expected):
    """Drop elements the SFPU cannot represent: subnormal (flushed to zero) and non-finite."""
    mask = (a.abs().float() >= SMALLEST_NORMAL) & (expected.abs().float() >= SMALLEST_NORMAL)
    return mask & torch.isfinite(a) & torch.isfinite(expected)


@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (1, 1, 320, 384)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_quasar_nextafter(device, shape, dtype):
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    torch_input_tensor_a = torch.rand(shape, dtype=torch_dtype) * 200 - 100
    torch_input_tensor_b = torch.rand(shape, dtype=torch_dtype) * 300 - 150

    torch_output_tensor = torch.nextafter(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.experimental.quasar.nextafter(input_tensor_a, input_tensor_b))

    in_scope = _in_scope(torch_input_tensor_a, torch_output_tensor)
    assert_with_ulp(
        expected_result=torch_output_tensor[in_scope], actual_result=output_tensor[in_scope], ulp_threshold=0
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_quasar_nextafter_direction(device, dtype):
    """The result has to land on the target's side of the operand.

    This is the assertion the swapped arms failed: with a < b the old code returned a - eps, which
    is strictly further from b than a was.
    """
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    a = torch.tensor([1.0, 1.0, -1.0, -1.0, 3.5, 3.5, -3.5, -3.5], dtype=torch_dtype)
    b = torch.tensor([2.0, 0.5, -0.5, -2.0, 7.0, 1.75, -1.75, -7.0], dtype=torch_dtype)
    a = a.repeat(4).reshape(1, 1, 1, -1).expand(1, 1, 32, 32).contiguous()
    b = b.repeat(4).reshape(1, 1, 1, -1).expand(1, 1, 32, 32).contiguous()

    expected = torch.nextafter(a, b)

    ttnn_a = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.experimental.quasar.nextafter(ttnn_a, ttnn_b))

    in_scope = _in_scope(a, expected)
    assert_with_ulp(expected_result=expected[in_scope], actual_result=actual[in_scope], ulp_threshold=0)

    # Every element moved towards b, and none of them overshot it.
    moved = (actual.float() - a.float()) * (b.float() - a.float())
    assert torch.all(moved[in_scope] > 0), "nextafter moved away from the target"
    assert torch.all((actual.float() - b.float()).abs()[in_scope] <= (a.float() - b.float()).abs()[in_scope])


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_quasar_nextafter_step_is_one_ulp(device, dtype):
    """The step has to scale with the operand rather than being a fixed distance.

    A fixed FLT_EPSILON is one ULP only on [1, 2) in float32, and is below half a ULP everywhere in
    bfloat16. Asserting `actual != a` is what catches the absorbed step: an exact comparison alone
    would also pass if the op happened to round back onto the operand.
    """
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    values = torch.tensor([1e30, -1e30, 1e20, -1e20, 1e8, -1e8, 16777216.0, -16777216.0], dtype=torch_dtype)
    a = values.repeat(4).reshape(1, 1, 1, -1).expand(1, 1, 32, 32).contiguous()
    # The target scales with the operand: a fixed offset would be absorbed at these magnitudes and
    # would silently retest the a == b case instead of a step.
    b = (values * 2).repeat(4).reshape(1, 1, 1, -1).expand(1, 1, 32, 32).contiguous()

    expected = torch.nextafter(a, b)

    ttnn_a = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.experimental.quasar.nextafter(ttnn_a, ttnn_b))

    in_scope = _in_scope(a, expected)
    assert_with_ulp(expected_result=expected[in_scope], actual_result=actual[in_scope], ulp_threshold=0)
    assert torch.all(
        actual.float()[in_scope] != a.float()[in_scope]
    ), "the step was absorbed; the operand came back unchanged"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_quasar_nextafter_equal_operands(device, dtype):
    """nextafter(x, x) is x, for every x -- including the zeros."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    values = torch.tensor([0.0, -0.0, 1.0, -1.0, 1e30, -1e30, 1e-30, -1e-30], dtype=torch_dtype)
    a = values.repeat(4).reshape(1, 1, 1, -1).expand(1, 1, 32, 32).contiguous()

    expected = torch.nextafter(a, a)

    ttnn_a = ttnn.from_torch(a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.experimental.quasar.nextafter(ttnn_a, ttnn_a))

    assert_with_ulp(expected_result=expected, actual_result=actual, ulp_threshold=0)
