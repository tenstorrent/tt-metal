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

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
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
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

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


# addalpha_bw writes input_a_grad (ttnn::assign) before the multiply that fills input_b_grad
# validates it, so a bad input_b_grad used to raise only after input_a_grad was already
# overwritten. The op must reject the bad grad before any device op runs.
def test_bw_addalpha_bad_other_grad_leaves_input_grad_unwritten(device, expect_error):
    input_shapes = torch.Size([1, 1, 32, 64])
    _, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    _, other_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    _, grad_tensor = data_gen_with_range(input_shapes, -70, 90, device)

    input_grad = ttnn.from_torch(torch.full(input_shapes, 7.0).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    bad_other_grad = ttnn.from_torch(
        torch.full(torch.Size([1, 1, 32, 32]), 1.0).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device
    )

    before = ttnn.to_torch(input_grad)
    with expect_error(RuntimeError, "addalpha_bw requires the preallocated input_b_grad logical shape"):
        ttnn.addalpha_bw(
            grad_tensor,
            input_tensor,
            other_tensor,
            2.0,
            are_required_outputs=[True, True],
            input_a_grad=input_grad,
            input_b_grad=bad_other_grad,
        )
    assert torch.equal(before, ttnn.to_torch(input_grad)), "input_a_grad was written before the call raised"


def test_bw_addalpha_preallocated_grads_valid_control(device):
    input_shapes = torch.Size([1, 1, 32, 64])
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -70, 90, device)
    _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)
    _, other_grad = data_gen_with_range(input_shapes, -1, 1, device)

    ttnn.addalpha_bw(
        grad_tensor,
        input_tensor,
        other_tensor,
        2.0,
        are_required_outputs=[True, True],
        input_a_grad=input_grad,
        input_b_grad=other_grad,
    )

    golden_function = ttnn.get_golden_function(ttnn.addalpha_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data, 2.0)
    assert compare_pcc([input_grad, other_grad], golden_tensor)
