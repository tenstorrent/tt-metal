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
@pytest.mark.parametrize("min", (-1.0, 1.0, 0.0, -0.5, -10.0, 10.0))
def test_bw_clamp_min(input_shapes, min, device):
    torch.manual_seed(12386)
    in_data = torch.randn(input_shapes, requires_grad=True).bfloat16()
    grad_data = torch.randn(input_shapes).bfloat16()
    grad_tensor = (
        tt_lib.tensor.Tensor(grad_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    input_tensor = (
        tt_lib.tensor.Tensor(in_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )

    pyt_y = torch.clamp(in_data, min=min)

    tt_output_tensor_on_device = tt_lib.tensor.clamp_min_bw(grad_tensor, input_tensor, min)
    tt_output_tensor = tt_output_tensor_on_device[0].cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_output_tensor = in_data.grad
    comp_pass, comp_out = comparison_funcs.comp_pcc(golden_output_tensor, tt_output_tensor)
    logger.info(comp_out)
    assert comp_pass
