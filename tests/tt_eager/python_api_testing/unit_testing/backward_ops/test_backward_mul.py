# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)
from loguru import logger


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_mul(input_shapes, device):
    torch.manual_seed(0)
    in_data_a = torch.randn(input_shapes, requires_grad=True).bfloat16()
    in_data_b = torch.randn(input_shapes, requires_grad=True).bfloat16()

    grad_data = torch.randn(input_shapes).bfloat16()

    grad_tensor = (
        tt_lib.tensor.Tensor(grad_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    input_tensor_a = (
        tt_lib.tensor.Tensor(in_data_a, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    input_tensor_b = (
        tt_lib.tensor.Tensor(in_data_b, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    tt_output_tensor_on_device = tt_lib.tensor.mul_bw(grad_tensor, input_tensor_a, input_tensor_b)
    tt_output_tensor_a = tt_output_tensor_on_device[0].cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
    tt_output_tensor_b = tt_output_tensor_on_device[1].cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    in_data_a.retain_grad()
    in_data_b.retain_grad()

    pyt_y = in_data_a * in_data_b

    pyt_y.backward(gradient=grad_data)

    golden_output_tensor_a = in_data_a.grad
    golden_output_tensor_b = in_data_b.grad

    comp_pass_a, comp_out_a = comparison_funcs.comp_pcc(golden_output_tensor_a, tt_output_tensor_a)
    comp_pass_b, comp_out_b = comparison_funcs.comp_pcc(golden_output_tensor_b, tt_output_tensor_b)

    logger.info(comp_out_a)
    logger.info(comp_out_b)
    assert comp_pass_a & comp_pass_b
