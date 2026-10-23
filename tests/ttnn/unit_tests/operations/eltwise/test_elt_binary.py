# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_ulp

pytestmark = pytest.mark.use_module_device


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

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


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
