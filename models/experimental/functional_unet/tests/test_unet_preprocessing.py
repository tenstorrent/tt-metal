# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

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


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("groups", [2])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_unet_model(batch, groups, device, use_program_cache, reset_seeds):
    torch_input, ttnn_input = create_unet_input_tensors(batch, groups, channel_order="first", pad=False)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    x = ttnn.to_device(ttnn_input, device=device, memory_config=ttnn_model.input_sharded_memory_config)
    x = ttnn_model.preprocess_input_tensor(x)
    x = ttnn.reshard(x, ttnn_model.preprocessed_input_sharded_memory_config)

    B, C, H, W = torch_input.shape
    ttnn_output = ttnn.to_torch(x)[:, :, :, :C].reshape(B, H, W, C).permute(0, 3, 1, 2)
    verify_with_pcc(torch_input, ttnn_output, pcc=0.9999)
