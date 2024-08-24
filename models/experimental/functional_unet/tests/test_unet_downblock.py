# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_unet.unet_utils import create_unet_input_tensors
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn2
from models.experimental.functional_unet.tt import model_preprocessing


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("perf_mode", [1])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_downblocks(batch, groups, perf_mode, device):
    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=False)
    model = unet_shallow_torch.UNet.from_random_weights(groups=1)

    parameters = model_preprocessing.create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn2.UNet(parameters, device)

    def check_pcc_conv(torch_tensor, ttnn_tensor, pcc=0.999):
        B, C, H, W = torch_tensor.shape
        ttnn_tensor = ttnn.to_torch(ttnn_tensor).reshape(B, H, W, C).permute(0, 3, 1, 2)
        assert_with_pcc(torch_tensor, ttnn_tensor, pcc)

    def check_pcc_pool(torch_tensor, ttnn_tensor, pcc=0.999):
        B, C, H, W = torch_tensor.shape
        ttnn_tensor = ttnn.to_torch(ttnn_tensor).reshape(B, H, W, -1).permute(0, 3, 1, 2)[:, :C, :, :]
        assert_with_pcc(torch_tensor, ttnn_tensor, pcc)

    logger.info("Verifying UNet downblock1")
    torch_output, torch_residual = model.downblock1(torch_input)
    ttnn_output, ttnn_residual = ttnn_model.downblock1(ttnn_input, perf_mode=perf_mode)
    check_pcc_conv(torch_residual, ttnn_residual)
    check_pcc_pool(torch_output, ttnn_output)

    logger.info("Verifying UNet downblock2")
    torch_output, torch_residual = model.downblock2(torch_output)
    ttnn_output, ttnn_residual = ttnn_model.downblock2(ttnn_output, perf_mode=perf_mode)
    check_pcc_conv(torch_residual, ttnn_residual)
    check_pcc_pool(torch_output, ttnn_output)

    logger.info("Verifying UNet downblock3")
    torch_output, torch_residual = model.downblock3(torch_output)
    ttnn_output, ttnn_residual = ttnn_model.downblock3(ttnn_output, perf_mode=perf_mode)
    check_pcc_conv(torch_residual, ttnn_residual)
    check_pcc_pool(torch_output, ttnn_output)

    logger.info("Verifying UNet downblock4")
    torch_output, torch_residual = model.downblock4(torch_output)
    ttnn_output, ttnn_residual = ttnn_model.downblock4(ttnn_output, perf_mode=perf_mode)
    check_pcc_conv(torch_residual, ttnn_residual)
    check_pcc_pool(torch_output, ttnn_output)

    # output_tensor = ttnn_model(device, ttnn_input_tensor, list(torch_input_tensor.shape), perf_mode=perf_mode)
