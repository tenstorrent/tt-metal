# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.utility_functions import is_wormhole_b0, is_blackhole
from tests.ttnn.unit_tests.operations.backward.utility_funcs import (
    data_gen_with_range,
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
# - name: fill.Scalar(Tensor self, Scalar value) -> Tensor
#   self: zeros_like(grad)
#   result: at::fill(self_t, 0)
def test_bw_fill(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device, True)

    tt_output_tensor_on_device = ttnn.fill_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.fill_bw)
    golden_tensor = golden_function(grad_data, in_data)

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
    golden_tensor = golden_function(grad_data, in_data)

    tt_output_tensor_on_device = [input_grad]
    comp_pass = compare_all_close(tt_output_tensor_on_device, golden_tensor, atol=0, rtol=0)
    assert comp_pass
