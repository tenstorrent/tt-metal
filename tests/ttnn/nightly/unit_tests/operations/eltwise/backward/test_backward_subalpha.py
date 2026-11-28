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
@pytest.mark.parametrize("alpha", [0.05])
def test_bw_subalpha(input_shapes, alpha, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.subalpha_bw(grad_tensor, input_tensor, other_tensor, alpha)

    golden_function = ttnn.get_golden_function(ttnn.subalpha_bw)
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
def test_bw_subalpha_default(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.subalpha_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.subalpha_bw)
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
@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True]])
def test_bw_subalpha_opt_output(input_shapes, device, are_required_outputs):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    input_grad = None
    other_grad = None
    tt_output_tensor_on_device = None

    if are_required_outputs[0]:
        _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)
    if are_required_outputs[1]:
        _, other_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.subalpha_bw(
        grad_tensor,
        input_tensor,
        other_tensor,
        are_required_outputs=are_required_outputs,
        input_grad=input_grad,
        other_grad=other_grad,
        queue_id=cq_id,
    )
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    tt_output_tensor_on_device = [input_grad, other_grad]

    golden_function = ttnn.get_golden_function(ttnn.subalpha_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

    status = True
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            status = status & compare_pcc([tt_output_tensor_on_device[i]], [golden_tensor[i]])
    assert status


# subalpha_bw writes input_grad (ttnn::assign) before the neg/multiply chain that fills other_grad
# validates it, so a bad other_grad used to raise only after input_grad was already overwritten.
# Both documented failure shapes are covered: a wrong-shape grad and a ROW_MAJOR grad (neg requires
# the preallocated output's layout to match the gradient's).
@pytest.mark.parametrize("bad_grad", ["wrong_shape", "row_major"])
def test_bw_subalpha_bad_other_grad_leaves_input_grad_unwritten(device, expect_error, bad_grad):
    input_shapes = torch.Size([1, 1, 32, 64])
    _, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    _, other_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    _, grad_tensor = data_gen_with_range(input_shapes, -70, 90, device)

    input_grad = ttnn.from_torch(torch.full(input_shapes, 7.0).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    if bad_grad == "wrong_shape":
        bad_other_grad = ttnn.from_torch(
            torch.full(torch.Size([1, 1, 32, 32]), 1.0).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device
        )
        message = "subalpha_bw requires the preallocated other_grad logical shape"
    else:
        bad_other_grad = ttnn.from_torch(
            torch.full(input_shapes, 1.0).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        message = "subalpha_bw requires the preallocated other_grad layout"

    before = ttnn.to_torch(input_grad)
    with expect_error(RuntimeError, message):
        ttnn.subalpha_bw(
            grad_tensor,
            input_tensor,
            other_tensor,
            2.0,
            are_required_outputs=[True, True],
            input_grad=input_grad,
            other_grad=bad_other_grad,
        )
    assert torch.equal(before, ttnn.to_torch(input_grad)), "input_grad was written before the call raised"


def test_bw_subalpha_preallocated_grads_valid_control(device):
    input_shapes = torch.Size([1, 1, 32, 64])
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -70, 90, device)
    _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)
    _, other_grad = data_gen_with_range(input_shapes, -1, 1, device)

    ttnn.subalpha_bw(
        grad_tensor,
        input_tensor,
        other_tensor,
        2.0,
        are_required_outputs=[True, True],
        input_grad=input_grad,
        other_grad=other_grad,
    )

    golden_function = ttnn.get_golden_function(ttnn.subalpha_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data, 2.0)
    assert compare_pcc([input_grad, other_grad], golden_tensor)
