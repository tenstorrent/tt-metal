# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
import ttnn
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("scalar", [0.05, 1.0, 0.5, 0.12, 0.0, -0.05, -1.0, -0.5, -0.12])
def test_bw_unary_mul(input_shapes, scalar, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)

    tt_output_tensor_on_device = ttnn.unary_mul_bw(grad_tensor, input_tensor, scalar)

    in_data.retain_grad()

    pyt_y = in_data * torch.tensor(scalar)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
