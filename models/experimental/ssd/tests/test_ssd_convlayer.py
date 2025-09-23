# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.utility_functions import (
    comp_allclose,
    comp_pcc,
)
from models.experimental.ssd.tt.ssd_mobilenetv3_convlayer import (
    TtMobileNetV3ConvLayer,
)
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large as pretrained,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_ssd_convlayer_inference(device, pcc, imagenet_sample_input, reset_seeds):
    TV_model = pretrained(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    TV_model.eval()

    FEATURE_INDEX = 0
    LAYER_INDEX = 0
    # torch convlayer
    torch_model = TV_model.backbone.features[FEATURE_INDEX][LAYER_INDEX]

    # Tt ssd_conv
    config = {"num_channels": 3}
    tt_model = TtMobileNetV3ConvLayer(
        config,
        in_channels=config["num_channels"],
        out_channels=16,
        kernel_size=3,
        stride=2,
        padding=1,
        use_activation=True,
        activation="HS",
        state_dict=TV_model.state_dict(),
        base_address=f"backbone.features.0.0",
        device=device,
    )

    # Run torch model
    torch_output = torch_model(imagenet_sample_input)

    # Run tt model
    tt_input = torch_to_tt_tensor_rm(imagenet_sample_input, device, put_on_device=True)
    tt_output = tt_model(tt_input)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("SSDConvlayer Passed!")

    assert does_pass, f"SSDConvlayer does not meet PCC requirement {pcc}."
