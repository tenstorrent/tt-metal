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
    "round_mode",
    (
        None,
        "trunc",
        "floor",
    ),
)
def test_bw_div(input_shapes, round_mode, device):
    in_data = torch.Tensor(size=input_shapes).uniform_()
    in_data.requires_grad = True
    input_tensor = (
        tt_lib.tensor.Tensor(in_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    grad_data = torch.Tensor(size=input_shapes).uniform_()
    grad_tensor = (
        tt_lib.tensor.Tensor(grad_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    other_data, other_tensor = data_gen_pt_tt(input_shapes, device, True)

    pyt_y = torch.div(in_data, other_data, rounding_mode=round_mode)

    if round_mode == None:
        round_mode = "None"
    tt_output_tensor_on_device = tt_lib.tensor.div_bw(grad_tensor, input_tensor, other_tensor, round_mode=round_mode)

    in_data.retain_grad()
    other_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad, other_data.grad]
    status = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert status
