# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import compare_pcc, data_gen_with_range
from math import pi


def random_complex_tensor(shape, real_range=(-100, 100), imag_range=(-100, 100)):
    torch.manual_seed(213919)
    real_part = (real_range[1] - real_range[0]) * torch.rand(shape) + real_range[0]
    imag_part = (imag_range[1] - imag_range[0]) * torch.rand(shape) + imag_range[0]
    return torch.complex(real_part, imag_part)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_polar(input_shapes, device):
    in_data = random_complex_tensor(input_shapes, (-90, 110), (0, 2 * pi))
    in_data.requires_grad = True

    grad_data = random_complex_tensor(input_shapes, (-50, 50), (-60, 60))

    input_tensor_a = (
        tt_lib.tensor.Tensor(in_data.real, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    input_tensor_b = (
        tt_lib.tensor.Tensor(in_data.imag, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    grad_data_cplx = torch.cat((grad_data.real, grad_data.imag), dim=-1)
    grad_tensor = (
        tt_lib.tensor.Tensor(grad_data_cplx, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    tt_output_tensor_on_device = tt_lib.tensor.polar_bw(grad_tensor, input_tensor_a, input_tensor_b)

    in_data.retain_grad()

    pyt_y = torch.polar(in_data.real, in_data.imag)

    pyt_y.backward(gradient=grad_data)

    grad_res_real = torch.real(in_data.grad)
    grad_res_imag = torch.imag(in_data.grad)
    golden_tensor = [torch.cat((grad_res_real, grad_res_imag), dim=-1)]

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
