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
@pytest.mark.parametrize(
    "memcfg",
    (
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM),
        tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1),
    ),
)
class TestMAELoss:
    def test_loss_mae_none(self, input_shapes, memcfg, device):
        torch.manual_seed(0)
        ref_data = torch.randn(input_shapes).bfloat16()
        pred_data = torch.randn(input_shapes).bfloat16()

        ref_tensor = (
            tt_lib.tensor.Tensor(ref_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
        )

        pred_tensor = (
            tt_lib.tensor.Tensor(pred_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
        )

        tt_output_tensor_on_device = tt_lib.tensor.maeloss(
            ref_tensor, pred_tensor, tt_lib.tensor.LossReductionMode.NONE, memcfg
        )
        tt_mae_output = tt_output_tensor_on_device.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

        loss = torch.nn.L1Loss(reduction="none")
        pt_mae_output = loss(ref_data.to(torch.float32), pred_data.to(torch.float32))
        comp_pass_a, comp_out_a = comparison_funcs.comp_pcc(pt_mae_output, tt_mae_output)

        logger.debug(comp_out_a)
        assert comp_pass_a

    def test_loss_mae_sum(self, input_shapes, memcfg, device):
        torch.manual_seed(0)
        ref_data = torch.randn(input_shapes).bfloat16()
        pred_data = torch.randn(input_shapes).bfloat16()

        ref_tensor = (
            tt_lib.tensor.Tensor(ref_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
        )

        pred_tensor = (
            tt_lib.tensor.Tensor(pred_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
        )

        tt_output_tensor_on_device = tt_lib.tensor.maeloss(
            ref_tensor, pred_tensor, tt_lib.tensor.LossReductionMode.SUM, memcfg
        )
        tt_mae_output = tt_output_tensor_on_device.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

        loss = torch.nn.L1Loss(reduction="sum")
        pt_mae_output = loss(ref_data.to(torch.float32), pred_data.to(torch.float32))

        comp_pass_a, comp_out_a = comparison_funcs.comp_allclose(
            pt_mae_output, torch.tensor(tt_mae_output[0, 0, 0, 0]), atol=4, rtol=1e-1
        )

        logger.debug(comp_out_a)
        assert comp_pass_a

    def test_loss_mae_mean(self, input_shapes, memcfg, device):
        if input_shapes[3] == 384:
            pytest.skip("mean not supported in dimensions 3, 4")
        torch.manual_seed(0)
        ref_data = torch.randn(input_shapes).bfloat16()
        pred_data = torch.randn(input_shapes).bfloat16()

        ref_tensor = (
            tt_lib.tensor.Tensor(ref_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
        )

        pred_tensor = (
            tt_lib.tensor.Tensor(pred_data, tt_lib.tensor.DataType.BFLOAT16).to(tt_lib.tensor.Layout.TILE).to(device)
        )

        tt_output_tensor_on_device = tt_lib.tensor.maeloss(
            ref_tensor, pred_tensor, tt_lib.tensor.LossReductionMode.MEAN, memcfg
        )
        tt_mae_output = tt_output_tensor_on_device.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

        loss = torch.nn.L1Loss(reduction="mean")
        pt_mae_output = loss(ref_data.to(torch.float32), pred_data.to(torch.float32))

        comp_pass_a, comp_out_a = comparison_funcs.comp_allclose(
            pt_mae_output, torch.tensor(tt_mae_output[0, 0, 0, 0]), atol=4, rtol=1e-1
        )

        logger.debug(comp_out_a)
        assert comp_pass_a
