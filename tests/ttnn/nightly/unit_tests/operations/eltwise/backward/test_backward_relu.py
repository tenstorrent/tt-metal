# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc, data_gen_with_range


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
# @pytest.mark.parametrize("threshold",[0.0])
def test_bw_relu(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.relu_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.relu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


# relu_bw used to compute gtz(input) * grad, and in float32 0 * inf and 0 * nan are NaN. torch returns
# an exact 0 wherever input <= 0, whatever grad holds.
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("grad_value", [float("inf"), float("-inf"), float("nan")], ids=["pos_inf", "neg_inf", "nan"])
@pytest.mark.parametrize("input_value", [-1.0, 1.0], ids=["inactive", "active"])
def test_bw_relu_non_finite_grad(device, dtype, grad_value, input_value):
    active = input_value > 0
    if active and grad_value != grad_value and dtype == ttnn.bfloat16:
        pytest.skip(
            "bfloat16 loses a NaN operand on the device, returning an infinity: "
            "https://github.com/tenstorrent/tt-metal/issues/31406"
        )
    shape = torch.Size([1, 1, 32, 32])
    grad = ttnn.from_torch(torch.full(shape, grad_value), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    inp = ttnn.from_torch(torch.full(shape, input_value), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.to_torch(ttnn.relu_bw(grad, inp)[0]).float()

    if not active:
        assert torch.equal(output, torch.zeros(shape)), f"expected 0, got {output[0, 0, 0, 0].item()}"
    elif grad_value != grad_value:
        assert torch.isnan(output).all(), f"expected NaN, got {output[0, 0, 0, 0].item()}"
    else:
        assert torch.equal(
            output, torch.full(shape, grad_value)
        ), f"expected {grad_value}, got {output[0, 0, 0, 0].item()}"
