# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

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
def test_unet_model(batch, groups, perf_mode, device):
    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=False)
    model = unet_shallow_torch.UNet.from_random_weights(groups=1)

    parameters = model_preprocessing.create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn2.UNet(parameters, device)

    torch_output_tensor = model(torch_input)
    output_tensor = ttnn_model(device, ttnn_input, list(torch_input.shape), perf_mode=perf_mode)

    B, C, H, W = torch_output_tensor.shape
    ttnn_tensor = ttnn.to_torch(output_tensor).reshape(B, H, W, C).permute(0, 3, 1, 2)
    assert_with_pcc(torch_output_tensor, ttnn_tensor, 0.99)
