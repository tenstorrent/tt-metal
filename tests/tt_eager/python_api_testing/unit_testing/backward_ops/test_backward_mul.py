# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_with_range, compare_pcc


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

    tt_output_tensor_on_device = tt_lib.tensor.mul_bw(grad_tensor, input_tensor_a, input_tensor_b)

    in_data_a.retain_grad()
    in_data_b.retain_grad()

    pyt_y = torch.mul(in_data_a, in_data_b)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data_a.grad, in_data_b.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
