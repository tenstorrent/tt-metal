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


@pytest.mark.parametrize("batch, groups", [(2, 1)])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_output_layer(batch, groups, device):
    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=False)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=True, input_channels=16)
    torch_output = model.output_layer(torch_input)
    ttnn_output = ttnn_model.output_layer(ttnn_input)

    B, C, H, W = torch_output.shape
    ttnn_output = ttnn.to_torch(ttnn_output)
    assert list(ttnn_output.shape) == [1, 1, B * H * W, 32], "Expected output layer to return padded output"
    ttnn_output = ttnn_output.reshape(B, H, W, 32)[:, :, :, :C].permute(0, 3, 1, 2)
    assert_with_pcc(torch_output, ttnn_output, 0.99)
