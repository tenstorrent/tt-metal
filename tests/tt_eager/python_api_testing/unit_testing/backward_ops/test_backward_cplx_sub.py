# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import compare_results


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
@pytest.mark.parametrize(
    "alpha",
    [-2, 0.5, 5, 10],
)
def test_bw_cplx_sub(input_shapes, alpha, device):
    in_data = random_complex_tensor(input_shapes, (-110, 100), (-80, 80))
    in_data.requires_grad = True

    other_data = random_complex_tensor(input_shapes, (-90, 90), (-70, 70))
    other_data.requires_grad = True

    grad_data = random_complex_tensor(input_shapes, (-50, 50), (-20, 60))

    in_data_cplx = torch.cat((in_data.real, in_data.imag), dim=3)
    other_data_cplx = torch.cat((other_data.real, other_data.imag), dim=3)
    input_tensor = (
        tt_lib.tensor.Tensor(in_data_cplx, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    other_tensor = (
        tt_lib.tensor.Tensor(other_data_cplx, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    grad_data_cplx = torch.cat((grad_data.real, grad_data.imag), dim=3)
    grad_tensor = (
        tt_lib.tensor.Tensor(grad_data_cplx, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    tt_output_tensor_on_device = tt_lib.tensor.complex_sub_bw(grad_tensor, input_tensor, other_tensor, alpha=alpha)

    in_data.retain_grad()
    other_data.retain_grad()

    pyt_y = torch.sub(in_data, other_data, alpha=alpha)

    pyt_y.backward(gradient=grad_data)

    grad_self_real = torch.real(in_data.grad)
    grad_self_imag = torch.imag(in_data.grad)
    grad_other_real = torch.real(other_data.grad)
    grad_other_imag = torch.imag(other_data.grad)
    golden_tensor = [
        torch.cat((grad_self_real, grad_self_imag), dim=3),
        torch.cat((grad_other_real, grad_other_imag), dim=3),
    ]

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
