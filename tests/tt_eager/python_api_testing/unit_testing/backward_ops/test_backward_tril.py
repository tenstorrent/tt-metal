# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

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
@pytest.mark.parametrize("diag", [0, 2])
def test_bw_tril(input_shapes, diag, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = tt_lib.tensor.tril_bw(grad_tensor, input_tensor, diag)

    in_data.retain_grad()

    pyt_y = torch.tril(in_data, diagonal=diag)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

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
@pytest.mark.parametrize("diag", [0, 2])
@pytest.mark.parametrize("are_required_outputs", [[True], [False]])
def test_bw_tril_with_opt_output(input_shapes, diag, device, are_required_outputs):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -70, 90, device)
    input_grad = None

    if are_required_outputs[0]:
        _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0

    tt_output_tensor_on_device = tt_lib.tensor.tril_bw(
        grad_tensor,
        input_tensor,
        diag,
        are_required_outputs=are_required_outputs,
        input_grad=input_grad,
        queue_id=cq_id,
    )

    in_data.retain_grad()

    pyt_y = torch.tril(in_data, diag)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = True
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            status = status & compare_pcc([tt_output_tensor_on_device[i]], [golden_tensor[i]])
    assert status
