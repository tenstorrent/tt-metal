# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_mul(input_shapes, device):
    in_data_a, input_tensor_a = data_gen_with_range(input_shapes, -1, 1, device, True)
    in_data_b, input_tensor_b = data_gen_with_range(input_shapes, -5, 5, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.mul_bw(grad_tensor, input_tensor_a, input_tensor_b)

    in_data_a.retain_grad()
    in_data_b.retain_grad()

    pyt_y = torch.mul(in_data_a, in_data_b)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data_a.grad, in_data_b.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True], [False, False]])
@pytest.mark.parametrize("pass_queue_id", [True, False])
def test_bw_mul_opt_output(input_shapes, device, are_required_outputs, pass_queue_id):
    in_data_a, input_tensor_a = data_gen_with_range(input_shapes, -90, 80, device, True)
    in_data_b, input_tensor_b = data_gen_with_range(input_shapes, -70, 90, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -60, 60, device)
    input_a_grad = None
    input_b_grad = None
    tt_output_tensor_on_device = None

    if are_required_outputs[0]:
        _, input_a_grad = data_gen_with_range(input_shapes, -1, 1, device)
    if are_required_outputs[1]:
        _, input_b_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    if pass_queue_id:
        tt_output_tensor_on_device = ttnn.mul_bw(
            grad_tensor,
            input_tensor_a,
            input_tensor_b,
            are_required_outputs=are_required_outputs,
            input_a_grad=input_a_grad,
            input_b_grad=input_b_grad,
            queue_id=cq_id,
        )
    else:
        tt_output_tensor_on_device = ttnn.mul_bw(
            grad_tensor,
            input_tensor_a,
            input_tensor_b,
            are_required_outputs=are_required_outputs,
            input_a_grad=input_a_grad,
            input_b_grad=input_b_grad,
        )

    in_data_a.retain_grad()
    in_data_b.retain_grad()

    pyt_y = torch.mul(in_data_a, in_data_b)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data_a.grad, in_data_b.grad]

    status = True
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            status = status & compare_pcc([tt_output_tensor_on_device[i]], [golden_tensor[i]])
    assert status
