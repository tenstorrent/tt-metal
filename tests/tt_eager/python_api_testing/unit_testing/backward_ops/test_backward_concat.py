# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib

from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_pt_tt, compare_results


@pytest.mark.parametrize(
    "input_shapes, input_shapes_2, dimension",
    (
        ((torch.Size([12, 1, 30, 32])), (torch.Size([2, 1, 30, 32])), 0),
        ((torch.Size([1, 2, 45, 64])), (torch.Size([1, 1, 45, 64])), 1),
        ((torch.Size([1, 1, 125, 32])), (torch.Size([1, 1, 32, 32])), 2),
        (
            (torch.Size([1, 1, 64, 80])),
            (torch.Size([1, 1, 64, 16])),
            3,
        ),  # size must be divisible by sizeof(uint32_t) because buffers hold uint32_t values
        # Tile shape
        ((torch.Size([4, 1, 32, 32])), (torch.Size([5, 1, 32, 32])), 0),
        ((torch.Size([1, 2, 64, 64])), (torch.Size([1, 1, 64, 64])), 1),
        ((torch.Size([1, 1, 64, 32])), (torch.Size([1, 1, 32, 32])), 2),
        ((torch.Size([1, 1, 64, 64])), (torch.Size([1, 1, 64, 32])), 3),
    ),
)
def test_bw_add(input_shapes, input_shapes_2, dimension, device):
    in_data = torch.randn(input_shapes, requires_grad=True).bfloat16()
    input_tensor = (
        tt_lib.tensor.Tensor(in_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.ROW_MAJOR).to(device)
    )

    other_data = torch.randn(input_shapes_2, requires_grad=True).bfloat16()
    other_tensor = (
        tt_lib.tensor.Tensor(other_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.ROW_MAJOR).to(device)
    )

    in_data.retain_grad()
    other_data.retain_grad()

    pyt_y = torch.cat((in_data, other_data), dim=dimension)

    grad_data = torch.randn(pyt_y.shape, requires_grad=True).bfloat16()
    grad_tensor = (
        tt_lib.tensor.Tensor(grad_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.ROW_MAJOR).to(device)
    )

    tt_output_tensor_on_device = tt_lib.tensor.concat_bw(grad_tensor, input_tensor, other_tensor, dimension)

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad, other_data.grad]

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
