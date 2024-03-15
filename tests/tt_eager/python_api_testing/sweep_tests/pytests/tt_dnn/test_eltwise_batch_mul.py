# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import tt_lib
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs


@pytest.mark.parametrize(
    ("input_shape_a", "input_shape_b"),
    [
        (torch.Size([1, 1, 32, 32]), torch.Size([32, 1, 32, 32])),
        (torch.Size([4, 3, 320, 384]), torch.Size([1, 3, 320, 384])),
        (torch.Size([16, 3, 320, 384]), torch.Size([1, 3, 320, 384])),
        (torch.Size([32, 1, 32, 1024]), torch.Size([1, 1, 32, 1024])),
        # (torch.Size([1, 1, 320, 384]), torch.Size([64, 1, 320, 384])), # error w.r.t arg is more than 1kb #issue 6361
    ],
)
class TestBatchMul:
    def test_batch_mul(self, input_shape_a, input_shape_b, device):
        torch.manual_seed(0)

        input_data_a = torch.randn(input_shape_a).bfloat16()
        input_tensor_a = (
            tt_lib.tensor.Tensor(input_data_a, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
        )
        input_data_b = torch.randn(input_shape_b).bfloat16()
        input_tensor_b = (
            tt_lib.tensor.Tensor(input_data_b, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
        )
        tt_output_tensor_on_device = tt_lib.tensor.batch_mul(input_tensor_a, input_tensor_b)

        tt_out_tensor = tt_output_tensor_on_device.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()
        pt_out_tensor = torch.mul(input_data_a, input_data_b)
        comp_pass, comp_out = comparison_funcs.comp_pcc(pt_out_tensor, tt_out_tensor, pcc=0.99)
        comp_all, _ = comparison_funcs.comp_allclose(pt_out_tensor, tt_out_tensor, atol=4, rtol=1e-1)
        logger.info(comp_pass)
        logger.info(comp_all)
        logger.info(comp_out)
        status = comp_pass | comp_all
        assert status
