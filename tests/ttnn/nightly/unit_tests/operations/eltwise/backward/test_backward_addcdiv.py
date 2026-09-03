# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [0.05, 1.0, 0.5, 5.0])
def test_bw_addcdiv(input_shapes, value, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    tensor1_data, tensor1_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    tensor2_data, tensor2_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)

    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device, False)

    tt_output_tensor_on_device = ttnn.addcdiv_bw(grad_tensor, input_tensor, tensor1_tensor, tensor2_tensor, value)

    golden_function = ttnn.get_golden_function(ttnn.addcdiv_bw)
    golden_tensor = golden_function(grad_data, in_data, tensor1_data, tensor2_data, value)

    comp_pcc = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pcc


@pytest.mark.parametrize("value", [1.0, -1.0])
def test_bw_addcdiv_zero_divisor(value, device):
    """Regression test for #55316: addcdiv_bw at tensor2 == 0 must match torch autograd sign and NaN rules."""
    input_shapes = torch.Size([1, 1, 1, 4])
    in_data = torch.zeros((1, 1, 1, 4), dtype=torch.bfloat16, requires_grad=True)
    tensor1_data = torch.tensor([[[[2.0, -2.0, 0.0, 2.0]]]], dtype=torch.bfloat16, requires_grad=True)
    tensor2_data = torch.zeros((1, 1, 1, 4), dtype=torch.bfloat16, requires_grad=True)
    grad_data = torch.tensor([[[[-1.0, 1.0, 1.0, 1.0]]]], dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(in_data, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tensor1_tensor = ttnn.from_torch(tensor1_data, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tensor2_tensor = ttnn.from_torch(tensor2_data, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    grad_tensor = ttnn.from_torch(grad_data, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    tt_output_tensor_on_device = ttnn.addcdiv_bw(grad_tensor, input_tensor, tensor1_tensor, tensor2_tensor, value)

    golden_function = ttnn.get_golden_function(ttnn.addcdiv_bw)
    golden_tensor = golden_function(grad_data, in_data, tensor1_data, tensor2_data, value)

    comp_pcc = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pcc
