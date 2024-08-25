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


def check_pcc_conv(torch_tensor, ttnn_tensor, pcc=0.995):
    B, C, H, W = torch_tensor.shape
    ttnn_tensor = ttnn.to_torch(ttnn_tensor).reshape(B, H, W, C).permute(0, 3, 1, 2)
    assert_with_pcc(torch_tensor, ttnn_tensor, pcc)


def check_pcc_pool(torch_tensor, ttnn_tensor, pcc=0.995):
    B, C, H, W = torch_tensor.shape
    ttnn_tensor = ttnn.to_torch(ttnn_tensor).reshape(B, H, W, -1).permute(0, 3, 1, 2)[:, :C, :, :]
    assert_with_pcc(torch_tensor, ttnn_tensor, pcc)


@pytest.mark.parametrize("batch, groups", [(2, 1)])
@pytest.mark.parametrize(
    "block_name, input_channels, input_height, input_width, residual_channels, factor",
    [
        ("upblock1", 64, 66, 10, 32, 5280),
        ("upblock2", 32, 132, 20, 32, 4 * 5280),
        ("upblock3", 32, 264, 40, 16, 4 * 4 * 5280),
        ("upblock4", 16, 528, 80, 16, 4 * 4 * 4 * 5280),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_upblock(
    batch, groups, block_name, input_channels, input_height, input_width, residual_channels, factor, device
):
    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=False)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = model_preprocessing.create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn2.UNet(parameters, device)

    torch_input, ttnn_input = create_unet_input_tensors(
        device,
        batch,
        groups,
        pad_input=True,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
    )
    torch_residual, ttnn_residual = create_unet_input_tensors(
        device,
        batch,
        groups,
        pad_input=True,
        input_channels=residual_channels,
        input_height=input_height * 2,
        input_width=input_width * 2,
    )
    torch_output = getattr(model, block_name)(torch_input, torch_residual)

    ttnn_input, ttnn_residual = ttnn_input.to(device), ttnn_residual.to(device)

    ttnn_output = getattr(ttnn_model, block_name)(ttnn_input, ttnn_residual, factor)
    check_pcc_conv(torch_output, ttnn_output)
