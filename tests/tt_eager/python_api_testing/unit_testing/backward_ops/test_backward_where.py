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
def test_bw_where(input_shapes, device):
    condition_data = torch.zeros(input_shapes, dtype=torch.bool)
    condition_data.view(-1)[::2] = True

    condition_tensor = (
        tt_lib.tensor.Tensor(condition_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -1, 1, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -4, 4, device)

    tt_output_tensor_on_device = tt_lib.tensor.where_bw(grad_tensor, condition_tensor, input_tensor, other_tensor)

    in_data.retain_grad()
    other_data.retain_grad()

    pyt_y = torch.where(condition_data, in_data, other_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad, other_data.grad]

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
