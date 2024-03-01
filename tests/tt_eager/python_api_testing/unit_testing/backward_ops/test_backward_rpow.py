# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import compare_results, data_gen_pt_tt


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "exponent",
    (
        1.5,
        5.7,
        0.0,
        15.2,
    ),
)
def test_bw_rpow(input_shapes, exponent, device):
    grad_data, grad_tensor = data_gen_pt_tt(input_shapes, device)
    in_data = torch.Tensor(size=input_shapes).uniform_()
    in_data.requires_grad = True
    input_tensor = (
        tt_lib.tensor.Tensor(in_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    pyt_y = torch.pow(in_data, exponent)

    tt_output_tensor_on_device = tt_lib.tensor.rpow_bw(grad_tensor, input_tensor, exponent)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]
    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
