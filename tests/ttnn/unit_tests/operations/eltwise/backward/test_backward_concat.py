# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes, input_shapes_2, dimension",
    (
        ((torch.Size([12, 1, 30, 32])), (torch.Size([2, 1, 30, 32])), 0),
        ((torch.Size([1, 2, 45, 64])), (torch.Size([1, 1, 45, 64])), 1),
        ((torch.Size([1, 1, 125, 32])), (torch.Size([1, 1, 32, 32])), 2),
        (
            (torch.Size([1, 1, 64, 80])),
            (torch.Size([1, 1, 64, 16])),
            3,
        ),  # size must be divisible by sizeof(uint32_t) because buffers hold uint32_t values
        # Tile shape
        ((torch.Size([4, 1, 32, 32])), (torch.Size([5, 1, 32, 32])), 0),
        ((torch.Size([1, 2, 64, 64])), (torch.Size([1, 1, 64, 64])), 1),
        ((torch.Size([1, 1, 64, 32])), (torch.Size([1, 1, 32, 32])), 2),
        ((torch.Size([1, 1, 64, 64])), (torch.Size([1, 1, 64, 32])), 3),
    ),
)
def test_bw_concat(input_shapes, input_shapes_2, dimension, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, True)

    other_data, other_tensor = data_gen_with_range(input_shapes_2, -100, 100, device, True, True)

    pyt_y = torch.concat((in_data, other_data), dim=dimension)

    grad_data, grad_tensor = data_gen_with_range(pyt_y.shape, -100, 100, device, True, True)

    tt_output_tensor_on_device = ttnn.concat_bw(grad_tensor, input_tensor, other_tensor, dimension)

    golden_function = ttnn.get_golden_function(ttnn.concat_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data, dimension)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes, input_shapes_2",
    (
        ((torch.Size([12, 1, 30, 32])), (torch.Size([2, 1, 30, 32]))),
        ((torch.Size([4, 1, 32, 32])), (torch.Size([5, 1, 32, 32]))),
    ),
)
def test_bw_concat_Default(input_shapes, input_shapes_2, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, True)

    other_data, other_tensor = data_gen_with_range(input_shapes_2, -100, 100, device, True, True)

    pyt_y = torch.concat((in_data, other_data))

    grad_data, grad_tensor = data_gen_with_range(pyt_y.shape, -100, 100, device, True, True)

    tt_output_tensor_on_device = ttnn.concat_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.concat_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


def test_bw_concat_default_example(device):
    x1_torch = torch.rand([12, 1, 30, 32], dtype=torch.bfloat16, requires_grad=True)
    x2_torch = torch.rand([2, 1, 30, 32], dtype=torch.bfloat16, requires_grad=True)
    grad_tensor = torch.rand([14, 1, 30, 32], dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.concat_bw)
    golden_tensor = golden_function(grad_tensor, x1_torch, x2_torch)

    grad_tt = ttnn.from_torch(grad_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    x1_tt = ttnn.from_torch(x1_torch, layout=ttnn.TILE_LAYOUT, device=device)
    x2_tt = ttnn.from_torch(x2_torch, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.concat_bw(grad_tt, x1_tt, x2_tt, 0)
    tt_out_1 = ttnn.to_torch(y_tt[1])
    tt_out_0 = ttnn.to_torch(y_tt[0])
    comp_pass_1 = torch.allclose(tt_out_1, golden_tensor[1])
    comp_pass_0 = torch.allclose(tt_out_0, golden_tensor[0])
    assert comp_pass_1 and comp_pass_0


@pytest.mark.parametrize(
    "input_shapes, input_shapes_2",
    (
        ((torch.Size([12, 1, 30, 32])), (torch.Size([2, 1, 30, 32]))),
        ((torch.Size([4, 1, 32, 32])), (torch.Size([5, 1, 32, 32]))),
    ),
)
@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True]])
def test_bw_concat_Default_with_output(input_shapes, input_shapes_2, device, are_required_outputs):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, True)

    other_data, other_tensor = data_gen_with_range(input_shapes_2, -100, 100, device, True, True)

    pyt_y = torch.concat((in_data, other_data))

    grad_data, grad_tensor = data_gen_with_range(pyt_y.shape, -100, 100, device, True, True)

    input_grad = None
    other_grad = None

    opt_tensor1 = torch.zeros(input_shapes, dtype=torch.bfloat16)
    opt_tensor2 = torch.zeros(input_shapes_2, dtype=torch.bfloat16)

    if are_required_outputs[0]:
        input_grad = ttnn.from_torch(
            opt_tensor1, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
    if are_required_outputs[1]:
        other_grad = ttnn.from_torch(
            opt_tensor2, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
    cq_id = 0

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.concat_bw(
        grad_tensor,
        input_tensor,
        other_tensor,
        are_required_outputs=are_required_outputs,
        input_a_grad=input_grad,
        input_b_grad=other_grad,
        queue_id=cq_id,
    )
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    tt_output_tensor_on_device = [input_grad, other_grad]

    golden_function = ttnn.get_golden_function(ttnn.concat_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

    status = True
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            status = status & compare_pcc([tt_output_tensor_on_device[i]], [golden_tensor[i]])
    assert status


@pytest.mark.parametrize(
    "input_shapes, input_shapes_2, dimension",
    (
        ((torch.Size([12, 1, 30, 32])), (torch.Size([2, 1, 30, 32])), 0),
        ((torch.Size([1, 2, 45, 64])), (torch.Size([1, 1, 45, 64])), 1),
        ((torch.Size([1, 1, 125, 32])), (torch.Size([1, 1, 32, 32])), 2),
        (
            (torch.Size([1, 1, 64, 80])),
            (torch.Size([1, 1, 64, 16])),
            3,
        ),  # size must be divisible by sizeof(uint32_t) because buffers hold uint32_t values
        # Tile shape
        ((torch.Size([4, 1, 32, 32])), (torch.Size([5, 1, 32, 32])), 0),
        ((torch.Size([1, 2, 64, 64])), (torch.Size([1, 1, 64, 64])), 1),
        ((torch.Size([1, 1, 64, 32])), (torch.Size([1, 1, 32, 32])), 2),
        ((torch.Size([1, 1, 64, 64])), (torch.Size([1, 1, 64, 32])), 3),
    ),
)
@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True]])
def test_bw_concat_with_output(input_shapes, input_shapes_2, dimension, device, are_required_outputs):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, True)

    other_data, other_tensor = data_gen_with_range(input_shapes_2, -100, 100, device, True, True)

    pyt_y = torch.concat((in_data, other_data), dim=dimension)

    grad_data, grad_tensor = data_gen_with_range(pyt_y.shape, -100, 100, device, True, True)

    input_grad = None
    other_grad = None

    opt_tensor1 = torch.zeros(input_shapes, dtype=torch.bfloat16)
    opt_tensor2 = torch.zeros(input_shapes_2, dtype=torch.bfloat16)

    if are_required_outputs[0]:
        input_grad = ttnn.from_torch(
            opt_tensor1, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
    if are_required_outputs[1]:
        other_grad = ttnn.from_torch(
            opt_tensor2, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

    cq_id = 0

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.concat_bw(
        grad_tensor,
        input_tensor,
        other_tensor,
        dimension,
        are_required_outputs=are_required_outputs,
        input_a_grad=input_grad,
        input_b_grad=other_grad,
        queue_id=cq_id,
    )
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    tt_output_tensor_on_device = [input_grad, other_grad]

    golden_function = ttnn.get_golden_function(ttnn.concat_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data, dimension)

    status = True
    for i in range(len(are_required_outputs)):
        if are_required_outputs[i]:
            status = status & compare_pcc([tt_output_tensor_on_device[i]], [golden_tensor[i]])
    assert status
