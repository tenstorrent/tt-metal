# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("alpha", [0.05, 2.0, 1.5, 0.12])
def test_bw_addalpha(input_shapes, alpha, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.addalpha_bw(grad_tensor, input_tensor, other_tensor, alpha)

    golden_function = ttnn.get_golden_function(ttnn.addalpha_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data, alpha)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


def test_bw_addalpha_example(device):
    x1_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True)
    x2_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16, requires_grad=True)
    grad_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    alpha = 1
    golden_function = ttnn.get_golden_function(ttnn.addalpha_bw)
    golden_tensor = golden_function(grad_tensor, x1_torch, x2_torch, alpha)
    grad_tt = ttnn.from_torch(grad_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    x1_tt = ttnn.from_torch(x1_torch, layout=ttnn.TILE_LAYOUT, device=device)
    x2_tt = ttnn.from_torch(x2_torch, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.addalpha_bw(grad_tt, x1_tt, x2_tt, alpha)
    tt_out_1 = ttnn.to_torch(y_tt[1])
    tt_out_0 = ttnn.to_torch(y_tt[0])
    comp_pass_1 = torch.allclose(tt_out_1, golden_tensor[1])
    comp_pass_0 = torch.allclose(tt_out_0, golden_tensor[0])
    assert comp_pass_1 and comp_pass_0


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_addalpha_wo_alpha(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.addalpha_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.addalpha_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

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
@pytest.mark.parametrize("alpha", [0.05, 2.0, 1.5, 0.12])
@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True]])
def test_bw_addalpha_with_opt_output(input_shapes, alpha, device, are_required_outputs):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -90, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -70, 90, device)
    input_grad = None
    other_grad = None

    if are_required_outputs[0]:
        _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)
    if are_required_outputs[1]:
        _, other_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0

    pages_before = ttnn._ttnn.reports.get_buffer_pages()
    ttnn.addalpha_bw(
        grad_tensor,
        input_tensor,
        other_tensor,
        alpha,
        are_required_outputs=are_required_outputs,
        input_a_grad=input_grad,
        input_b_grad=other_grad,
        queue_id=cq_id,
    )
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages())

    tt_output_tensor_on_device = [input_grad, other_grad]

    golden_function = ttnn.get_golden_function(ttnn.addalpha_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data, alpha)

    status = True
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            status = status & compare_pcc([tt_output_tensor_on_device[i]], [golden_tensor[i]])
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
def test_bw_addalpha_with_opt_output_wo_alpha(input_shapes, device, are_required_outputs):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -90, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -70, 90, device)
    input_grad = None
    other_grad = None

    if are_required_outputs[0]:
        _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)
    if are_required_outputs[1]:
        _, other_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_output_tensor_on_device = ttnn.addalpha_bw(
        grad_tensor,
        input_tensor,
        other_tensor,
        are_required_outputs=are_required_outputs,
        input_a_grad=input_grad,
        input_b_grad=other_grad,
        queue_id=cq_id,
    )

    golden_function = ttnn.get_golden_function(ttnn.addalpha_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

    status = True
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            status = status & compare_pcc([tt_output_tensor_on_device[i]], [golden_tensor[i]])
    assert status
