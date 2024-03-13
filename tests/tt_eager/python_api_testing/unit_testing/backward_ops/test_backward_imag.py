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
def test_bw_imag(input_shapes, device):
    torch.manual_seed(0)
    in_data = torch.randn(input_shapes, dtype=torch.complex64)
    in_data.requires_grad = True

    torch.manual_seed(42)
    grad_data = torch.randn(input_shapes, dtype=torch.complex64)
    grad_data = grad_data.imag

    in_data_cplx = torch.cat((in_data.real, in_data.imag), dim=3)
    input_tensor = (
        tt_lib.tensor.Tensor(in_data_cplx, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    grad_tensor = (
        tt_lib.tensor.Tensor(grad_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    tt_output_tensor_on_device = tt_lib.tensor.imag_bw(grad_tensor, input_tensor)

    in_data.retain_grad()

    pyt_y = torch.imag(in_data)

    pyt_y.backward(gradient=grad_data)

    grad_res_real = torch.real(in_data.grad)
    grad_res_imag = torch.imag(in_data.grad)
    golden_tensor = [torch.cat((grad_res_real, grad_res_imag), dim=3)]

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
