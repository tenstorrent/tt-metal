# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.utility_functions import is_wormhole_b0, is_blackhole
from tests.ttnn.unit_tests.operations.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_val,
    compare_all_close,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
# Pytorch Reference
#   name: fill.Scalar(Tensor self, Scalar value) -> Tensor
#   self: zeros_like(grad)
#   result: at::fill(self_t, 0)
def test_bw_fill(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)

    tt_output_tensor_on_device = ttnn.fill_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.fill_bw)
    golden_tensor = golden_function(grad_data, in_data, value=2.0)

    comp_pass = compare_all_close(tt_output_tensor_on_device, golden_tensor, atol=0, rtol=0)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_fill_opt_tensor(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)

    _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)
    input_grad = ttnn.to_memory_config(input_grad, ttnn.L1_MEMORY_CONFIG)
    cq_id = 0
    pages_before = ttnn._ttnn.reports.get_buffer_pages()
    ttnn.fill_bw(grad_tensor, input_tensor, input_grad=input_grad, queue_id=cq_id)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages())

    golden_function = ttnn.get_golden_function(ttnn.fill_bw)
    golden_tensor = golden_function(grad_data, in_data, value=2.0)

    tt_output_tensor_on_device = [input_grad]
    comp_pass = compare_all_close(tt_output_tensor_on_device, golden_tensor, atol=0, rtol=0)
    assert comp_pass


# Pytorch Reference
#    name: fill.Tensor(Tensor self, Tensor value) -> Tensor
#    self: zeros_like(grad)
#    value: grad.sum()
#    result: at::fill(self_t, value_t)
#   This variant is supported only in Wormhole N300 for non-decimals grad_tensor.
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True]])
def test_bw_fill_tensors(input_shapes, device, are_required_outputs):
    grad_data = torch.zeros(input_shapes)
    grad_tensor = ttnn.from_torch(grad_data, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)
    in_data2 = torch.tensor(2.0, requires_grad=True)
    _, input_tensor2 = data_gen_with_val(torch.Size([1, 1, 32, 32]), device, required_grad=True, val=2.0)

    input_grad = None
    other_grad = None

    if are_required_outputs[0]:
        _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)
    if are_required_outputs[1]:
        _, other_grad = data_gen_with_range(input_shapes, -1, 1, device)

    ttnn.fill_bw(
        grad_tensor,
        input_tensor,
        input_tensor2,
        are_required_outputs=are_required_outputs,
        input_grad=input_grad,
        other_grad=other_grad,
    )

    tt_output_tensor_on_device = [input_grad, other_grad]

    golden_function = ttnn.get_golden_function(ttnn.fill_bw)
    golden_tensor = golden_function(grad_data, in_data, value=in_data2)

    status = True
    if are_required_outputs[0]:
        status = status & compare_all_close([tt_output_tensor_on_device[0]], [golden_tensor[0]], atol=0, rtol=0)
    if are_required_outputs[1]:
        output_tensor = ttnn.from_device(tt_output_tensor_on_device[1])
        tt_output = ttnn.to_torch(output_tensor)
        tt_output = torch.tensor(tt_output[0][0][0][0], dtype=torch.float32)
        status = status & torch.allclose(tt_output, golden_tensor[1], atol=0.001, rtol=0)
    assert status
