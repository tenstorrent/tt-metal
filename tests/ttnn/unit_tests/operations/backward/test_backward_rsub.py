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
def test_bw_rsub(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -5, 5, device, True)

    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, 5, device)

    tt_output_tensor_on_device = ttnn.rsub_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.rsub_bw)
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
@pytest.mark.parametrize("are_required_outputs", [[True, True], [True, False], [False, True], [False, False]])
def test_bw_rsub_opt(input_shapes, device, are_required_outputs):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -5, 5, device, True)

    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, 5, device)

    input_grad = None
    other_grad = None
    tt_output_tensor_on_device = None

    if are_required_outputs[0]:
        _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)
    if are_required_outputs[1]:
        _, other_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0

    if are_required_outputs[0] and are_required_outputs[1]:
        pages_before = ttnn._ttnn.reports.get_buffer_pages()
        ttnn.rsub_bw(
            grad_tensor, input_tensor, other_tensor, input_grad=input_grad, other_grad=other_grad, queue_id=cq_id
        )
        assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages())
        tt_output_tensor_on_device = [input_grad, other_grad]
    else:
        tt_output_tensor_on_device = ttnn.rsub_bw(grad_tensor, input_tensor, other_tensor, queue_id=cq_id)

    golden_function = ttnn.get_golden_function(ttnn.rsub_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
