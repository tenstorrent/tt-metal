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
@pytest.mark.parametrize("value", [0.05, 1.0, 0.5, 5.0])
def test_bw_addcdiv(input_shapes, value, device):
    torch.manual_seed(0)
    in_data = torch.randn(input_shapes, requires_grad=True).bfloat16()
    tensor1_data = torch.randn(input_shapes, requires_grad=True).bfloat16()
    tensor2_data = torch.randn(input_shapes, requires_grad=True).bfloat16()

    grad_data = torch.randn(input_shapes).bfloat16()

    grad_tensor = (
        tt_lib.tensor.Tensor(grad_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    input_tensor = (
        tt_lib.tensor.Tensor(in_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    tensor1_tensor = (
        tt_lib.tensor.Tensor(tensor1_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    tensor2_tensor = (
        tt_lib.tensor.Tensor(tensor2_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    tt_output_tensor_on_device = tt_lib.tensor.addcdiv_bw(
        grad_tensor, input_tensor, tensor1_tensor, tensor2_tensor, value
    )
    tt_output_tensor_a = tt_output_tensor_on_device[0].cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
    tt_output_tensor_b = tt_output_tensor_on_device[1].cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
    tt_output_tensor_c = tt_output_tensor_on_device[2].cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    in_data.retain_grad()
    tensor1_data.retain_grad()
    tensor2_data.retain_grad()

    pyt_y = torch.addcdiv(in_data, tensor1_data, tensor2_data, value=value)

    pyt_y.backward(gradient=grad_data)

    golden_output_tensor_a = in_data.grad
    golden_output_tensor_b = tensor1_data.grad
    golden_output_tensor_c = tensor2_data.grad

    comp_pass_a, _ = comparison_funcs.comp_pcc(golden_output_tensor_a, tt_output_tensor_a, 0.99)
    _, comp_out_a = comparison_funcs.comp_allclose_and_pcc(golden_output_tensor_a, tt_output_tensor_a)

    comp_pass_b, _ = comparison_funcs.comp_pcc(golden_output_tensor_b, tt_output_tensor_b, 0.99)
    _, comp_out_b = comparison_funcs.comp_allclose_and_pcc(golden_output_tensor_b, tt_output_tensor_b)

    comp_pass_c, _ = comparison_funcs.comp_pcc(golden_output_tensor_c, tt_output_tensor_c, 0.99)
    _, comp_out_c = comparison_funcs.comp_allclose_and_pcc(golden_output_tensor_c, tt_output_tensor_c)

    logger.info(comp_out_a)
    logger.info(comp_out_b)
    logger.info(comp_out_c)
    assert comp_pass_a & comp_pass_b & comp_pass_c
