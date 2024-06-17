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
def test_bw_sqrt(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -1e4, 1e4, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1e4, 1e4, device)

    pyt_y = torch.sqrt(in_data)

    tt_output_tensor_on_device = tt_lib.tensor.sqrt_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

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
def test_bw_sqrt_output(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -1e4, 1e4, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1e4, 1e4, device)
    input_grad = None

    _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_output_tensor_on_device = tt_lib.tensor.sqrt_bw(
        grad_tensor,
        input_tensor,
        are_required_outputs=[True],
        input_grad=input_grad,
        queue_id=cq_id,
    )
    pyt_y = torch.sqrt(in_data)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]
    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
