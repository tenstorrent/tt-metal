# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
    data_gen_with_range_dtype,
)
from tests.ttnn.backward_shape_matrix import (
    BINARY_BACKWARD_BROADCAST_MATRIX,
    broadcast_grad_shape,
    matrix_ids,
    matrix_shapes,
)


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

    golden_function = ttnn.get_golden_function(ttnn.mul_bw)
    golden_tensor = golden_function(grad_data, in_data_a, in_data_b)

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
def test_bw_mul_opt(input_shapes, device, are_required_outputs):
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
    ttnn.mul_bw(
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

    golden_function = ttnn.get_golden_function(ttnn.mul_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

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
@pytest.mark.parametrize("scalar", [0.05, 1.0, 0.5, 0.12, 0.0, -0.05, -1.0, -0.5, -0.12])
def test_bw_mul_scalar(input_shapes, scalar, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)

    tt_output_tensor_on_device = ttnn.mul_bw(grad_tensor, input_tensor, scalar)

    golden_function = ttnn.get_golden_function(ttnn.mul_bw)
    golden_tensor = golden_function(grad_data, in_data, scalar)

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
@pytest.mark.parametrize("scalar", [0.05, 1.0, 0.5, 0.12, 0.0, -0.05, -1.0, -0.5, -0.12])
def test_bw_mul_scalar_bf8b(input_shapes, scalar, device):
    in_data, input_tensor = data_gen_with_range_dtype(input_shapes, -100, 100, device, True, False, ttnn.bfloat8_b)
    grad_data, grad_tensor = data_gen_with_range_dtype(input_shapes, -5, 5, device, False, False, ttnn.bfloat8_b)

    tt_output_tensor_on_device = ttnn.mul_bw(grad_tensor, input_tensor, scalar)

    golden_function = ttnn.get_golden_function(ttnn.mul_bw)
    golden_tensor = golden_function(grad_data, in_data, scalar)

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
@pytest.mark.parametrize("scalar", [0.05, 1.0, 0.5, 0.12, 0.0, -0.05, -1.0, -0.5, -0.12])
def test_bw_mul_scalar_opt_output(input_shapes, scalar, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)

    _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.mul_bw(grad_tensor, input_tensor, scalar, input_grad=input_grad, queue_id=cq_id)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))
    tt_output_tensor_on_device = [input_grad]
    golden_function = ttnn.get_golden_function(ttnn.mul_bw)
    golden_tensor = golden_function(grad_data, in_data, scalar)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


# ---------------------------------------------------------------------------
# broadcast triples — grad_out.shape == broadcast(input, other) per autograd contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_shape,other_shape", matrix_shapes(), ids=matrix_ids())
def test_bw_mul_broadcast(input_shape, other_shape, device):
    grad_shape = tuple(broadcast_grad_shape(input_shape, other_shape))

    in_data_a, input_tensor_a = data_gen_with_range(torch.Size(input_shape), -1, 1, device, True)
    in_data_b, input_tensor_b = data_gen_with_range(torch.Size(other_shape), -5, 5, device, True)
    grad_data, grad_tensor = data_gen_with_range(torch.Size(grad_shape), -100, 100, device)

    tt_output_tensor_on_device = ttnn.mul_bw(grad_tensor, input_tensor_a, input_tensor_b)

    golden_function = ttnn.get_golden_function(ttnn.mul_bw)
    golden_tensor = golden_function(grad_data, in_data_a, in_data_b)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize("input_shape,other_shape", matrix_shapes(), ids=matrix_ids())
@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True]])
def test_bw_mul_broadcast_opt(input_shape, other_shape, are_required_outputs, device):
    grad_shape = tuple(broadcast_grad_shape(input_shape, other_shape))

    in_data, input_tensor = data_gen_with_range(torch.Size(input_shape), -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(torch.Size(other_shape), -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(torch.Size(grad_shape), -100, 100, device)

    input_grad = None
    other_grad = None
    if are_required_outputs[0]:
        _, input_grad = data_gen_with_range(torch.Size(input_shape), -1, 1, device)
    if are_required_outputs[1]:
        _, other_grad = data_gen_with_range(torch.Size(other_shape), -1, 1, device)

    cq_id = 0
    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.mul_bw(
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

    golden_function = ttnn.get_golden_function(ttnn.mul_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

    status = True
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            status = status & compare_pcc([tt_output_tensor_on_device[i]], [golden_tensor[i]])
    assert status
