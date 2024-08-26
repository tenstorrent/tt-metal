# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn


def check_pcc_conv(torch_tensor, ttnn_tensor, pcc=0.999):
    B, C, H, W = torch_tensor.shape
    ttnn_tensor = ttnn.to_torch(ttnn_tensor).reshape(B, H, W, C).permute(0, 3, 1, 2)
    assert_with_pcc(torch_tensor, ttnn_tensor, pcc)


def check_pcc_pool(torch_tensor, ttnn_tensor, pcc=0.999):
    B, C, H, W = torch_tensor.shape
    ttnn_tensor = ttnn.to_torch(ttnn_tensor).reshape(B, H, W, -1).permute(0, 3, 1, 2)[:, :C, :, :]
    assert_with_pcc(torch_tensor, ttnn_tensor, pcc)


@pytest.mark.parametrize("batch, groups", [(2, 1)])
@pytest.mark.parametrize(
    "block_name, input_channels, input_height, input_width",
    [
        ("downblock1", 4, 1056, 160),
        ("downblock2", 16, 528, 80),
        ("downblock3", 16, 264, 40),
        ("downblock4", 32, 132, 20),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_downblock(batch, groups, block_name, input_channels, input_height, input_width, device, reset_seeds):
    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=False)
    model = unet_shallow_torch.UNet.from_random_weights(groups=1)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    torch_input, ttnn_input = create_unet_input_tensors(
        device,
        batch,
        groups,
        pad_input=True,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
    )
    torch_output, torch_residual = getattr(model, block_name)(torch_input)

    ttnn_input = ttnn_input.to(device)
    ttnn_output, ttnn_residual = getattr(ttnn_model, block_name)(ttnn_input)

    check_pcc_conv(torch_residual, ttnn_residual)
    check_pcc_pool(torch_output, ttnn_output)
