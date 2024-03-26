# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import compare_pcc, data_gen_with_range


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_round(input_shapes, device):
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 199, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, -200, 201, device, required_grad=True)
    pyt_y = torch.round(in_data)

    tt_output_tensor_on_device = tt_lib.tensor.round_bw(grad_tensor)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
