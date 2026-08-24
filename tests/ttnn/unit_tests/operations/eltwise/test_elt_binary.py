# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc, assert_with_ulp
from models.common.utility_functions import torch_random

pytestmark = pytest.mark.use_module_device


def run_elt_binary_test_range(device, h, w, ttnn_function, low, high, *, pcc=0.9999, exact=False):
    """Run a binary eltwise op on bf16 inputs in [low, high) and assert vs the torch golden.

    Defaults to ``assert_with_pcc(pcc)`` for composite math (ldexp/logaddexp/xlogy/bias_gelu) where
    the expected error exceeds the ULP <= 5 policy. Callers set ``exact=True`` for ops whose output
    is a bit-exact selection or boolean (maximum/minimum, logical_and/or/xor)."""
    torch.manual_seed(0)
    low = low
    high = high
    torch_input_tensor_a = torch_random((h, w), low, high, dtype=torch.bfloat16)
    torch.manual_seed(42)
    torch_input_tensor_b = torch_random((h, w), low, high, dtype=torch.bfloat16)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    if exact:
        assert_equal(torch_output_tensor.to(output_tensor.dtype), output_tensor)
    else:
        assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_ldexp(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.ldexp, -60, 60, pcc=0.9995)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logaddexp(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.logaddexp, -80, 80)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logaddexp2(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.logaddexp2, -60, 100, pcc=0.993)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logical_and(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.logical_and, -100, 100, exact=True)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logical_or(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.logical_or, -100, 100, exact=True)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logical_xor(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.logical_xor, -100, 100, exact=True)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_xlogy(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.xlogy, 1e-6, 1e6)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_bias_gelu(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.bias_gelu, -100, 100)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_maximum(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.maximum, -100, 100, exact=True)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_minimum(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.minimum, -100, 100, exact=True)


def test_arithmetic_operators(device):
    """Test basic arithmetic operators (+, -, *, /) on ttnn tensors"""

    # Create test tensors with different values
    a_torch = torch.full((32, 32), 4.0, dtype=torch.bfloat16)
    b_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)

    # Convert to ttnn tensors on device
    a = ttnn.from_torch(a_torch, device=device, layout=ttnn.TILE_LAYOUT)
    b = ttnn.from_torch(b_torch, device=device, layout=ttnn.TILE_LAYOUT)

    # Test operations
    c = a + b  # Addition: 4 + 2 = 6
    d = a - b  # Subtraction: 4 - 2 = 2
    e = a * b  # Multiplication: 4 * 2 = 8
    f = a / b  # Division: 4 / 2 = 2
    g = a / 2  # Tensor / scalar: 4 / 2 = 2
    h = 8 / a  # Scalar / tensor: 8 / 4 = 2

    # Verify results
    c_torch = ttnn.to_torch(c)
    expected_add = torch.full((32, 32), 6.0, dtype=torch.bfloat16)
    assert torch.equal(c_torch, expected_add), "Addition result incorrect"

    d_torch = ttnn.to_torch(d)
    expected_sub = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    assert torch.equal(d_torch, expected_sub), "Subtraction result incorrect"

    e_torch = ttnn.to_torch(e)
    expected_mul = torch.full((32, 32), 8.0, dtype=torch.bfloat16)
    assert torch.equal(e_torch, expected_mul), "Multiplication result incorrect"

    f_torch = ttnn.to_torch(f)
    expected_div = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    assert torch.equal(f_torch, expected_div), "Division result incorrect"

    g_torch = ttnn.to_torch(g)
    expected_tensor_div_scalar = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    assert torch.equal(g_torch, expected_tensor_div_scalar), "Tensor / scalar result incorrect"

    h_torch = ttnn.to_torch(h)
    expected_scalar_div_tensor = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    assert torch.equal(h_torch, expected_scalar_div_tensor), "Scalar / tensor result incorrect"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "broadcast_shape",
    [
        (1, 1, 1, 64),  # ROW broadcast
        (1, 1, 32, 1),  # COL broadcast
        (1, 1, 1, 1),  # SCALAR broadcast
    ],
    ids=["row_bcast", "col_bcast", "scalar_bcast"],
)
def test_fused_relu_with_broadcast(device, dtype, broadcast_shape):
    """Regression test for #44823: fused RELU silently dropped on subtile-broadcast paths.

    The PACK_RELU optimization sets ZERO_RELU once at kernel start, but subtile-broadcast
    kernels clear it via pack_reconfig_data_format mid-iteration. The fix falls through to
    the SFPU activation path for broadcast cases.
    """
    torch.manual_seed(0)
    a_shape = (1, 1, 32, 64)
    torch_a = torch.randn(a_shape).to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
    torch_b = torch.randn(broadcast_shape).to(torch_a.dtype)

    golden = torch.relu(torch_a + torch_b)

    tt_a = ttnn.from_torch(torch_a, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    tt_b = ttnn.from_torch(torch_b, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)

    tt_out = ttnn.add(tt_a, tt_b, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)])
    result = ttnn.to_torch(tt_out)

    assert_with_ulp(golden, result, 1)


# fmt: off
@pytest.mark.parametrize("ttnn_op", [ttnn.add, ttnn.subtract, ttnn.rsub])
@pytest.mark.parametrize("fast_and_approximate_mode, ulp_threshold", [(False, 0), (None, 1)])
@pytest.mark.parametrize("high, low", [(0, -1e5), (1e5, 0), (500, -500), (1e5, 1e-5), ])
# fmt: on
def test_rne_approx_modes(device, ttnn_op, fast_and_approximate_mode, ulp_threshold, high, low):
    """fast_and_approximate_mode=False routes bfloat16 add/sub/rsub through the SFPU with RNE
    rounding, which matches torch exactly. The default (unset) keeps the 1-ULP FPU kernel."""

    torch.manual_seed(0)
    assert high > low, "high must be greater than low"
    torch_input_tensor_a = torch.randn((128, 128), dtype=torch.bfloat16) * (high - low) + low
    torch_input_tensor_b = torch.randn((128, 128), dtype=torch.bfloat16) * (high - low) + low
    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    kwargs = {} if fast_and_approximate_mode is None else {"fast_and_approximate_mode": fast_and_approximate_mode}
    output = ttnn.to_torch(ttnn_op(input_tensor_a, input_tensor_b, **kwargs))

    assert_with_ulp(torch_output_tensor, output, ulp_threshold)


# fmt: off
@pytest.mark.parametrize("ttnn_op", [ttnn.add_, ttnn.subtract_, ttnn.rsub_])
@pytest.mark.parametrize("fast_and_approximate_mode, ulp_threshold", [(False, 0), (None, 1)])
# fmt: on
def test_rne_approx_modes_inplace(device, ttnn_op, fast_and_approximate_mode, ulp_threshold):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((128, 128), dtype=torch.bfloat16) * 1e5
    torch_input_tensor_b = torch.randn((128, 128), dtype=torch.bfloat16) * 1e5
    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    kwargs = {} if fast_and_approximate_mode is None else {"fast_and_approximate_mode": fast_and_approximate_mode}
    ttnn_op(input_tensor_a, input_tensor_b, **kwargs)

    assert_with_ulp(torch_output_tensor, ttnn.to_torch(input_tensor_a), ulp_threshold)


# fmt: off
@pytest.mark.parametrize("ttnn_op", [ttnn.add, ttnn.subtract, ttnn.rsub])
@pytest.mark.parametrize("dtype_a, dtype_b, output_dtype", [
    (ttnn.float32, ttnn.float32, None),
    (ttnn.int32, ttnn.int32, None),
    (ttnn.bfloat8_b, ttnn.bfloat8_b, None),
    (ttnn.bfloat4_b, ttnn.bfloat4_b, None),
    (ttnn.float32, ttnn.bfloat16, None),      # output follows lhs -> FLOAT32
    (ttnn.bfloat16, ttnn.bfloat16, ttnn.float32),
])
# fmt: on
def test_rne_accurate_mode_rejects_non_bfloat16_output(device, ttnn_op, dtype_a, dtype_b, output_dtype, expect_error):
    """The accurate path only exists to round a bfloat16 result, so asking for it on any other
    output dtype is rejected rather than silently ignored."""
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((64, 64)) * 100
    torch_input_tensor_b = torch.randn((64, 64)) * 100

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=dtype_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=dtype_b, layout=ttnn.TILE_LAYOUT, device=device)

    kwargs = {} if output_dtype is None else {"dtype": output_dtype}
    with expect_error(RuntimeError, r"fast_and_approximate_mode=false is only supported for a BFLOAT16 output"):
        ttnn_op(input_tensor_a, input_tensor_b, fast_and_approximate_mode=False, **kwargs)
