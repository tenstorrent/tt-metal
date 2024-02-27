# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_pt_tt, compare_results


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "order",
    [1, 2, 3, 4, 5, 10],
)
def test_bw_polygamma(input_shapes, order, device):
    in_data = torch.rand(input_shapes) * 9 + 1
    in_data.requires_grad = True
    n = order
    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device)

    input_tensor = (
        tt_lib.tensor.Tensor(in_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    tt_output_tensor_on_device = tt_lib.tensor.polygamma_bw(grad_tensor, input_tensor, n)

    in_data.retain_grad()

    pyt_y = torch.polygamma(n, in_data)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]

    status = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert status
