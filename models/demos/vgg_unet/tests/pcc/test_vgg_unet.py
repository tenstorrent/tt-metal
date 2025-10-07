# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_grayskull
from models.demos.vgg_unet.common import load_torch_model
from models.demos.vgg_unet.reference.vgg_unet import UNetVGG19
from models.demos.vgg_unet.ttnn.model_preprocessing import create_vgg_unet_model_parameters
from models.demos.vgg_unet.ttnn.ttnn_vgg_unet import Tt_vgg_unet
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_grayskull()
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
        # True,
    ],
    ids=[
        "pretrained_weight_false",
        # "pretrained_weight_true",
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True, ids=["0"])
def test_vgg_unet(device, reset_seeds, model_location_generator, use_pretrained_weight, min_channels=16):
    torch.manual_seed(0)

    # Input creation
    torch_input = torch.randn((1, 3, 256, 256))

    # Model initialisation
    torch_model = UNetVGG19()

    # Pre-trained weights processing
    if use_pretrained_weight:
        torch_model = load_torch_model(torch_model, model_location_generator)
    torch_model.eval()

    # Model call
    ref = torch_model(torch_input)

    # Weights pre-processing
    parameters = create_vgg_unet_model_parameters(torch_model, torch_input, device)

    ttnn_model = Tt_vgg_unet(device, parameters, parameters.conv_args)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    ttnn_input = ttnn_input.to(device, ttnn.L1_MEMORY_CONFIG)
    result = ttnn_model(ttnn_input)

    result = ttnn.to_torch(result)
    result = result.permute(0, 3, 1, 2)
    result = result.reshape(ref.shape)
    pcc_passed, pcc_message = assert_with_pcc(result, ref, 0.98)
    logger.info(pcc_message)
