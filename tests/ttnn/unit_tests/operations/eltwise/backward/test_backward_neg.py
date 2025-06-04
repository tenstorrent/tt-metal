# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc, data_gen_with_range


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_neg(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.neg_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.neg_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_neg_opt(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)
    input_grad = ttnn.to_memory_config(input_grad, ttnn.L1_MEMORY_CONFIG)

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.neg_bw(grad_tensor, input_tensor, input_grad=input_grad)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    golden_function = ttnn.get_golden_function(ttnn.neg_bw)
    golden_tensor = golden_function(grad_data, in_data)

    tt_output_tensor_on_device = [input_grad]
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)

    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_neg_opt_id(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)
    input_grad = ttnn.to_memory_config(input_grad, ttnn.L1_MEMORY_CONFIG)
    cq_id = 0

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.neg_bw(grad_tensor, input_tensor, input_grad=input_grad, queue_id=cq_id)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    golden_function = ttnn.get_golden_function(ttnn.neg_bw)
    golden_tensor = golden_function(grad_data, in_data)

    tt_output_tensor_on_device = [input_grad]
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)

    assert comp_pass
