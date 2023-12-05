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
# Pytorch Reference
# - name: fill.Scalar(Tensor self, Scalar value) -> Tensor
#   self: zeros_like(grad)
#   result: at::fill(self_t, 0)
def test_bw_fill_zero(input_shapes, device):
    torch.manual_seed(12386)
    grad_data = torch.randn(input_shapes).bfloat16()

    grad_tensor = (
        tt_lib.tensor.Tensor(grad_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
    )
    tt_output_tensor_on_device = tt_lib.tensor.fill_zero_bw(grad_tensor)
    pyt_y = torch.zeros_like(grad_data)

    tt_output_tensor = tt_output_tensor_on_device[0].cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    golden_output_tensor = pyt_y

    comp_pass, _ = comparison_funcs.comp_pcc(golden_output_tensor, tt_output_tensor, 0.99)
    _, comp_out = comparison_funcs.comp_allclose_and_pcc(golden_output_tensor, tt_output_tensor)
    logger.info(comp_out)
    assert comp_pass
