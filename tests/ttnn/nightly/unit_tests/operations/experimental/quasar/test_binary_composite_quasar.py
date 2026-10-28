# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp


def _to_device(t, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def test_remainder_scalar_honours_activations(device):
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16) * 10
    out = ttnn.experimental.quasar.remainder(
        _to_device(x, device), -3.0, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]
    )
    # remainder by a negative divisor is <= 0, so a dropped RELU would leave negative values.
    golden = torch.relu(torch.remainder(x.float(), -3.0)).bfloat16()
    assert_with_ulp(expected_result=golden, actual_result=ttnn.to_torch(out), ulp_threshold=1)


def test_remainder_scalar_honours_dtype(device):
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16) * 10
    out = ttnn.experimental.quasar.remainder(_to_device(x, device), 3.0, dtype=ttnn.float32)
    assert out.dtype == ttnn.float32
    assert_with_ulp(expected_result=torch.remainder(x.float(), 3.0), actual_result=ttnn.to_torch(out), ulp_threshold=1)


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [((32,), (64,)), ((1, 32), (1, 64)), ((1, 32, 1), (1, 1, 64)), ((32, 1, 1, 1, 1), (1, 1, 1, 1, 64))],
)
def test_outer_low_rank(device, shape_a, shape_b):
    torch.manual_seed(0)
    a = torch.randn(shape_a, dtype=torch.bfloat16)
    b = torch.randn(shape_b, dtype=torch.bfloat16)
    out = ttnn.experimental.quasar.outer(_to_device(a, device), _to_device(b, device))
    golden = torch.outer(a.flatten().float(), b.flatten().float())
    assert_with_pcc(golden, ttnn.to_torch(out).float().reshape(golden.shape), 0.99)


def test_outer_rejects_scalar_input(device, expect_error):
    a = _to_device(torch.tensor(2.0, dtype=torch.bfloat16), device)
    b = _to_device(torch.randn(64, dtype=torch.bfloat16), device)
    with expect_error(RuntimeError, "inputs must be at least 1D"):
        ttnn.experimental.quasar.outer(a, b)


@pytest.mark.parametrize(
    "scalar", [1.5, float("inf"), 2.0**31, 2**31], ids=["fractional", "inf", "out_of_range", "uint32_out_of_range"]
)
def test_remainder_int32_rejects_inexact_scalar(device, scalar, expect_error):
    x = _to_device(torch.randint(-50, 50, (1, 1, 32, 32), dtype=torch.int32), device, dtype=ttnn.int32)
    with expect_error(RuntimeError, "INT32 input needs a finite integral scalar"):
        ttnn.experimental.quasar.remainder(x, scalar)


@pytest.mark.parametrize("op", [ttnn.add, ttnn.experimental.quasar.add], ids=["ttnn", "quasar"])
def test_output_tensor_with_extra_leading_dim_is_rejected(device, op, expect_error):
    a = _to_device(torch.randn(32, 32, dtype=torch.bfloat16), device)
    b = _to_device(torch.randn(32, 32, dtype=torch.bfloat16), device)
    out = _to_device(torch.zeros(5, 32, 32, dtype=torch.bfloat16), device)
    with expect_error(RuntimeError, "does not match the broadcasted output shape"):
        op(a, b, output_tensor=out)


def test_rank7_lhs_with_rank6_rhs(device):
    torch.manual_seed(0)
    a = torch.randn(1, 2, 1, 1, 2, 32, 32, dtype=torch.bfloat16)
    b = torch.randn(2, 1, 1, 2, 32, 32, dtype=torch.bfloat16)
    out = ttnn.experimental.quasar.add(_to_device(a, device), _to_device(b, device))
    assert_with_ulp(
        expected_result=(a.float() + b.float()).bfloat16(), actual_result=ttnn.to_torch(out), ulp_threshold=1
    )
