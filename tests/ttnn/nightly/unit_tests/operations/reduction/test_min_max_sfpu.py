# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.max / ttnn.min vs torch.amax / torch.amin for int32 and float32 (SFPU reduce path)."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("in_dtype", [ttnn.int32, ttnn.float32])
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 32, 32),
        (1, 1, 30, 96),
        (1, 1, 90, 32),
        (1, 3, 17, 19),
        (2, 4, 64, 60),
        (2, 1, 256, 2048),
        (2, 1, 2048, 256),
    ],
)
@pytest.mark.parametrize("dim", [0, 1, -1, -2, (-1, -2), None])
@pytest.mark.parametrize("op", ["max", "min"])
def test_max_min(device, in_dtype, input_shape, dim, op):
    torch.manual_seed(0)
    if in_dtype == ttnn.int32:
        torch_input_tensor = torch.randint(-50_000, 50_000, input_shape, dtype=torch.int32)
    else:
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    torch_op = torch.amax if op == "max" else torch.amin
    ttnn_op = ttnn.max if op == "max" else ttnn.min

    torch_output_tensor = torch_op(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=in_dtype)
    output_tensor = ttnn.to_torch(ttnn_op(input_tensor, dim=dim))

    assert output_tensor.dtype == torch_input_tensor.dtype

    if in_dtype == ttnn.int32:
        assert_equal(output_tensor, torch_output_tensor)
    else:
        assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999999)


@pytest.mark.parametrize("in_dtype", [ttnn.int32, ttnn.float32])
@pytest.mark.parametrize("scale", [2.0, 0.5, -3.0])
@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((1, 2, 64, 64), -1),
        ((1, 1, 96, 64), -2),
        ((1, 1, 64, 64), (-1, -2)),
    ],
)
@pytest.mark.parametrize("op", ["max", "min"])
def test_max_min_with_scaling(device, in_dtype, input_shape, dim, op, scale):
    torch.manual_seed(0)
    if in_dtype == ttnn.int32:
        torch_input_tensor = torch.randint(-50_000, 50_000, input_shape, dtype=torch.int32)
    else:
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    torch_op = torch.amax if op == "max" else torch.amin
    ttnn_op = ttnn.max if op == "max" else ttnn.min

    if in_dtype == ttnn.int32:
        torch_expected = torch_op(torch_input_tensor.float() * scale, dim=dim).to(torch.int32)
    else:
        torch_expected = torch_op(torch_input_tensor * scale, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=in_dtype)
    output_tensor = ttnn.to_torch(ttnn_op(input_tensor, dim=dim, scalar=scale))

    assert output_tensor.dtype == torch_input_tensor.dtype

    if in_dtype == ttnn.int32:
        assert_equal(output_tensor, torch_expected)
    else:
        assert_with_pcc(torch_expected, output_tensor, pcc=0.999999)


# ---------------------------------------------------------------------------
# FP32 precision retention: inputs all collapse to the same bf16 value but are
# distinct in fp32 — verifies the SFPU path preserves fp32 mantissa bits.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((1, 1, 32, 32), -1),  # W reduce, single tile
        ((1, 1, 32, 32), -2),  # H reduce, single tile
        ((1, 1, 64, 32), -1),  # W reduce, Ht=2
        ((1, 1, 32, 64), -2),  # H reduce, Wt=2
        ((1, 1, 32, 128), -1),
        ((1, 1, 128, 32), -2),
    ],
)
@pytest.mark.parametrize("op", ["max", "min"])
def test_fp32_precision_beyond_bf16_resolution(device, input_shape, dim, op):
    h, w = input_shape[-2], input_shape[-1]
    fp32_ulp_at_one = 2.0**-23

    n = h * w
    values = torch.tensor([1.0 + (k + 1) * fp32_ulp_at_one for k in range(n)], dtype=torch.float32)
    torch_input = values.reshape(input_shape)

    collapsed = torch_input.to(torch.bfloat16).to(torch.float32)
    assert torch.all(collapsed == 1.0).item(), "probe invariant broken: values are distinguishable in bf16"

    torch_op = torch.amax if op == "max" else torch.amin
    ttnn_op = ttnn.max if op == "max" else ttnn.min

    torch_expected = torch_op(torch_input, dim=dim)

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    output_tensor = ttnn.to_torch(ttnn_op(input_tensor, dim=dim))

    assert_equal(output_tensor, torch_expected)


# ---------------------------------------------------------------------------
# Boundary tests for INT32 input data containing INT32_MIN / INT32_MAX.
# ---------------------------------------------------------------------------
def _int32_input_with_injected_value(input_shape, base_low, base_high, injected_value, seed=0):
    """Build a deterministic int32 tensor of given shape with `injected_value`
    placed at the corners and the center so it lands in different tile rows/cols
    regardless of reduce dim."""
    torch.manual_seed(seed)
    t = torch.randint(base_low, base_high, input_shape, dtype=torch.int32)
    # Corners
    t[..., 0, 0] = injected_value
    t[..., -1, -1] = injected_value
    # Around the center
    h, w = input_shape[-2], input_shape[-1]
    if h >= 2 and w >= 2:
        t[..., h // 2, w // 2] = injected_value
    return t


@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((1, 1, 60, 30), -1),
        ((1, 1, 60, 30), -2),
    ],
)
@pytest.mark.parametrize(
    "injected_value",
    [torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max],
    ids=["INT32_MIN", "INT32_MAX"],
)
@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize(
    "base_range",
    [(10, 1000), (-1000, -10), (-1000, 1000)],
    ids=["all_positive", "all_negative", "mixed"],
)
def test_int32_reduce_with_extreme_value_in_input(device, input_shape, dim, injected_value, op, base_range):
    base_low, base_high = base_range
    _INT32_MIN = torch.iinfo(torch.int32).min
    _INT32_MAX = torch.iinfo(torch.int32).max

    # A: MIN negates input before max-finding; -INT32_MIN overflows back to INT32_MIN, losing the extreme value
    is_min_int32_edge = op == "min" and injected_value == _INT32_MIN

    # B: INT32_MIN (0x80000000) is compared as 0 via INT32_2S_COMP; 0 beats all negatives so MAX returns wrong value
    is_max_int32_min_edge = (
        op == "max"
        and injected_value == _INT32_MIN
        and (base_high < 0 or (base_low < 0 and dim == -2 and input_shape[-2] % 32 != 0))
    )

    # C: row reduce uses sign-magnitude mode; SFPSWAP picks larger magnitude, inverting negative ordering
    is_row_reduce_sign_magnitude_issue = (
        dim == -1
        and injected_value == _INT32_MAX
        and ((op == "max" and base_high < 0) or (op == "min" and base_low > 0))
    )

    if is_min_int32_edge or is_max_int32_min_edge or is_row_reduce_sign_magnitude_issue:
        pytest.xfail("Int32 SFPU reduce(min/max) returns wrong result with INT32_MIN/MAX in input (issue #44750).")

    torch_input = _int32_input_with_injected_value(input_shape, base_low, base_high, injected_value)

    torch_op = torch.amax if op == "max" else torch.amin
    ttnn_op = ttnn.max if op == "max" else ttnn.min

    torch_output = torch_op(torch_input, dim=dim)

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)
    output_tensor = ttnn.to_torch(ttnn_op(input_tensor, dim=dim))

    assert_equal(output_tensor, torch_output)
