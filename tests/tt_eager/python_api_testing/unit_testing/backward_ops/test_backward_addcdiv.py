# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_with_range, compare_pcc


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

    tt_output_tensor_on_device = tt_lib.tensor.addcdiv_bw(
        grad_tensor, input_tensor, tensor1_tensor, tensor2_tensor, value
    )

    in_data.retain_grad()
    tensor1_data.retain_grad()
    tensor2_data.retain_grad()

    pyt_y = torch.addcdiv(in_data, tensor1_data, tensor2_data, value=value)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad, tensor1_data.grad, tensor2_data.grad]

    comp_pcc = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [0.05, 1.0, 0.5, 0.12])
@pytest.mark.parametrize(
    "are_required_outputs",
    [
        [True, True, True],
        [True, True, False],
        [True, False, False],
        [True, False, True],
        [False, True, True],
        [False, True, False],
        [False, False, True],
    ],
)
def test_bw_addcdiv_with_optional(input_shapes, value, are_required_outputs, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 20, 30, device, True)
    tensor1_data, tensor1_tensor = data_gen_with_range(input_shapes, -30, -20, device, True)
    tensor2_data, tensor2_tensor = data_gen_with_range(input_shapes, -10, -1, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    if are_required_outputs[0]:
        _, input_grad = data_gen_with_range(input_shapes, 1, 10, device)
    else:
        input_grad = None
    if are_required_outputs[1]:
        _, tensor1_grad = data_gen_with_range(input_shapes, 10, 20, device)
    else:
        tensor1_grad = None
    if are_required_outputs[2]:
        _, tensor2_grad = data_gen_with_range(input_shapes, 20, 30, device)
    else:
        tensor2_grad = None

    tt_output_tensor_on_device = tt_lib.tensor.addcdiv_bw(
        grad_tensor,
        input_tensor,
        tensor1_tensor,
        tensor2_tensor,
        value,
        are_required_outputs=are_required_outputs,
        input_grad=input_grad,
        tensor1_grad=tensor1_grad,
        tensor2_grad=tensor2_grad,
    )

    in_data.retain_grad()
    tensor1_data.retain_grad()
    tensor2_data.retain_grad()

    pyt_y = torch.addcdiv(in_data, tensor1_data, tensor2_data, value=value)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad, tensor1_data.grad, tensor2_data.grad]

    status = True
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            status = status & compare_pcc([tt_output_tensor_on_device[i]], [golden_tensor[i]], 0.99)
    assert status
