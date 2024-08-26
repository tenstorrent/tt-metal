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


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_downblocks(batch, groups, device):
    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=False)
    model = unet_shallow_torch.UNet.from_random_weights(groups=1)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    def check_pcc(torch_tensor, ttnn_tensor, pcc=0.995):
        B, C, H, W = torch_tensor.shape
        ttnn_tensor = ttnn.to_torch(ttnn_tensor).reshape(B, H, W, C).permute(0, 3, 1, 2)
        assert_with_pcc(torch_tensor, ttnn_tensor, pcc)

    torch_input, ttnn_input = create_unet_input_tensors(
        device, batch, groups, pad_input=True, input_channels=32, input_height=66, input_width=10
    )
    torch_output = model.bottleneck(torch_input)

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_output = ttnn_model.bottleneck(ttnn_input)

    check_pcc(torch_output, ttnn_output)
