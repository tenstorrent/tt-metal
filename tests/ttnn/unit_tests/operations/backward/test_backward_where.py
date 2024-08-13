# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.unit_tests.operations.backward.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_where(input_shapes, device):
    condition_data = torch.zeros(input_shapes, dtype=torch.bool)
    condition_data.view(-1)[::2] = True

    condition_tensor = ttnn.Tensor(condition_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -1, 1, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -4, 4, device)

    tt_output_tensor_on_device = ttnn.where_bw(grad_tensor, condition_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.where_bw)
    golden_tensor = golden_function(grad_data, condition_data, in_data, other_data)

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
@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True]])
def test_bw_where_output(input_shapes, are_required_outputs, device):
    condition_data = torch.zeros(input_shapes, dtype=torch.bool)
    condition_data.view(-1)[::2] = True

    condition_tensor = ttnn.Tensor(condition_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -1, 1, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -4, 4, device)
    input_grad = None
    other_grad = None

    if are_required_outputs[0]:
        _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)
    if are_required_outputs[1]:
        _, other_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0

    ttnn.where_bw(
        grad_tensor,
        condition_tensor,
        input_tensor,
        other_tensor,
        are_required_outputs=are_required_outputs,
        input_a_grad=input_grad,
        input_b_grad=other_grad,
        queue_id=cq_id,
    )

    output_tensor = [input_grad, other_grad]

    golden_function = ttnn.get_golden_function(ttnn.where_bw)
    golden_tensor = golden_function(grad_data, condition_data, in_data, other_data)

    status = True
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            status = status & compare_pcc([output_tensor[i]], [golden_tensor[i]])
    assert status
