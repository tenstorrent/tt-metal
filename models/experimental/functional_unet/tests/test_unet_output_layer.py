# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn

from models.experimental.functional_unet.tests.common import verify_with_pcc


@pytest.mark.parametrize("batch, groups", [(1, 2)])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_output_layer(batch, groups, device, reset_seeds):
    torch_input, ttnn_input = create_unet_input_tensors(batch, groups)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    torch_input, ttnn_input = create_unet_input_tensors(
        batch, groups, input_channels=16, channel_order="last", fold=True
    )
    torch_output = model.output_layer(torch_input)

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn_output = ttnn_model.output_layer(ttnn_input)
    ttnn_output = ttnn_model.postprocess_output_tensor(ttnn_output)

    print("OUTPUT SHAPE: ", ttnn_output.shape)
    breakpoint()

    B, C, H, W = torch_output.shape
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.reshape(B, C, H, W)  # .permute(0, 3, 1, 2)
    verify_with_pcc(torch_output, ttnn_output, 0.9998)
