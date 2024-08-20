# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
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
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
)
class TestMSELoss:
    def test_loss_mse_none(self, input_shapes, memcfg, device):
        torch.manual_seed(0)
        ref_data = torch.randn(input_shapes).bfloat16()
        pred_data = torch.randn(input_shapes).bfloat16()

        ref_tensor = ttnn.Tensor(ref_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

        pred_tensor = ttnn.Tensor(pred_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

        tt_output_tensor_on_device = ttnn.mse_loss(
            ref_tensor, pred_tensor, reduction=ttnn.LossReductionMode.NONE, memory_config=memcfg
        )
        tt_mse_output = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

        loss = torch.nn.MSELoss(reduction="none")
        pt_mse_output = loss(ref_data.to(torch.float32), pred_data.to(torch.float32))
        comp_pass_a, comp_out_a = comparison_funcs.comp_pcc(pt_mse_output, tt_mse_output)

        logger.debug(comp_out_a)
        assert comp_pass_a

    def test_loss_mse_sum(self, input_shapes, memcfg, device):
        torch.manual_seed(0)
        ref_data = torch.randn(input_shapes).bfloat16()
        pred_data = torch.randn(input_shapes).bfloat16()

        ref_tensor = ttnn.Tensor(ref_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

        pred_tensor = ttnn.Tensor(pred_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

        tt_output_tensor_on_device = ttnn.mse_loss(
            ref_tensor, pred_tensor, reduction=ttnn.LossReductionMode.SUM, memory_config=memcfg
        )
        tt_mse_output = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

        loss = torch.nn.MSELoss(reduction="sum")
        pt_mse_output = loss(ref_data.to(torch.float32), pred_data.to(torch.float32))
        comp_pass_a, comp_out_a = comparison_funcs.comp_allclose(
            pt_mse_output, torch.tensor(tt_mse_output[0, 0, 0, 0]), atol=1e-1, rtol=1e-1
        )

        logger.debug(comp_out_a)
        assert comp_pass_a

    def test_loss_mse_mean(self, input_shapes, memcfg, device):
        if input_shapes[3] == 384:
            pytest.skip("mean not supported in dimensions 3, 4")
        torch.manual_seed(0)
        ref_data = torch.randn(input_shapes).bfloat16()
        pred_data = torch.randn(input_shapes).bfloat16()

        ref_tensor = ttnn.Tensor(ref_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

        pred_tensor = ttnn.Tensor(pred_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

        tt_output_tensor_on_device = ttnn.mse_loss(
            ref_tensor, pred_tensor, reduction=ttnn.LossReductionMode.MEAN, memory_config=memcfg
        )
        tt_mse_output = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

        loss = torch.nn.MSELoss(reduction="mean")
        pt_mse_output = loss(ref_data.to(torch.float32), pred_data.to(torch.float32))
        comp_pass_a, comp_out_a = comparison_funcs.comp_allclose(
            pt_mse_output, torch.tensor(tt_mse_output[0, 0, 0, 0]), atol=1e-1, rtol=1e-1
        )

        logger.debug(comp_out_a)
        assert comp_pass_a
