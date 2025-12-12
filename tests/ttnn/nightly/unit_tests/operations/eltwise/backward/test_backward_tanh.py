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
def test_bw_tanh(input_shapes, device):
    # tt tan supports input range [-1.45, 1.45]
    in_data, input_tensor = data_gen_with_range(input_shapes, -1.45, 1.45, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1e4, 1e4, device)

    tt_output_tensor_on_device = ttnn.tanh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.tanh_bw)
    golden_tensor = golden_function(grad_data, in_data)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, 0.95)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_tanh_with_output(input_shapes, device):
    # tt tan supports input range [-1.45, 1.45]
    in_data, input_tensor = data_gen_with_range(input_shapes, -1.45, 1.45, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1e4, 1e4, device)
    input_grad = None

    _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_output_tensor_on_device = ttnn.tanh_bw(
        grad_tensor,
        input_tensor,
        input_grad=input_grad,
        queue_id=cq_id,
    )

    golden_function = ttnn.get_golden_function(ttnn.tanh_bw)
    golden_tensor = golden_function(grad_data, in_data)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, 0.95)
    assert status
