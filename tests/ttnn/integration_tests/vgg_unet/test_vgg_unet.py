# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.utility_functions import skip_for_grayskull

from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
from loguru import logger
import os

from models.experimental.functional_vgg_unet.reference.vgg_unet import UNetVGG19
from models.experimental.functional_vgg_unet.ttnn.model_preprocessing import create_vgg_unet_model_parameters
from models.experimental.functional_vgg_unet.ttnn.ttnn_vgg_unet import Tt_vgg_unet


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_vgg_unet(device, reset_seeds, model_location_generator):
    torch.manual_seed(0)

    # Input creation
    torch_input = torch.randn((1, 3, 256, 256), dtype=torch.bfloat16)
    torch_input = torch_input.float()

    # Model initialisation
    torch_model = UNetVGG19()

    # Pre-trained weights processing
    weights_pth = "models/experimental/functional_vgg_unet/vgg_unet_torch.pth"
    torch_dict = torch.load(weights_pth)
    new_state_dict = dict(zip(torch_model.state_dict().keys(), torch_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    # Model call
    ref = torch_model(torch_input)

    # Weights pre-processing
    parameters = create_vgg_unet_model_parameters(torch_model, torch_input, device)

    ttnn_model = Tt_vgg_unet(device, parameters, parameters.conv_args)

    torch_input = torch_input.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)

    result_ttnn = ttnn_model(ttnn_input)

    result = ttnn.to_torch(result_ttnn)
    result = result.permute(0, 3, 1, 2)
    result = result.reshape(ref.shape)
    pcc_passed, pcc_message = assert_with_pcc(result, ref, 0.99)  # PCC = 0.99
    logger.info(pcc_message)
