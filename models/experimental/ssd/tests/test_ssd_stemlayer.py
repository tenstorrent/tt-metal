# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.utility_functions import (
    comp_allclose,
    comp_pcc,
)
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large as pretrained,
)
from models.experimental.ssd.tt.ssd_mobilenetv3_stemlayer import TtMobileNetV3Stem


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_ssd_stem_inference(device, pcc, reset_seeds):
    torch_model = pretrained(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    torch_model.eval()

    # torch stemlayer
    model = torch_model.backbone.features[0][1]

    # Tt ssd_stem
    config = {"in_channels": 16}
    tt_model = TtMobileNetV3Stem(
        config,
        in_channels=config["in_channels"],
        expanded_channels=16,
        out_channels=16,
        kernel_size=1,
        stride=1,
        padding=1,
        state_dict=torch_model.state_dict(),
        base_address=f"backbone.features.0.1",
        device=device,
    )

    tt_model.eval()

    # Run torch model
    input_tensor = torch.randn(1, 16, 112, 112)
    torch_output = model(input_tensor)

    # Run tt model
    tt_stem_input = torch_to_tt_tensor_rm(input_tensor, device)
    tt_output = tt_model(tt_stem_input)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    if does_pass:
        logger.info("SSDStemlayer Passed!")

    assert does_pass, f"SSDStemlayer does not meet PCC requirement {pcc}."
