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
def test_bw_tanh(input_shapes, device):
    # tt tan supports input range [-1.45, 1.45]
    in_data, input_tensor = data_gen_with_range(input_shapes, -1.45, 1.45, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -1e4, 1e4, device)
    pyt_y = torch.tanh(in_data)

    tt_output_tensor_on_device = tt_lib.tensor.tanh_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, 0.95)
    assert status
