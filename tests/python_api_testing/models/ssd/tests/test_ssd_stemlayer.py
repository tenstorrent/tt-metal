from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../../../..")

import torch
import pytest
from loguru import logger
from torchvision.models import mobilenet_v3_large as pretrained
from torchvision.models import MobileNet_V3_Large_Weights

import tt_lib
from python_api_testing.models.utility_functions_new import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
from python_api_testing.models.ssd.tt.ssd_mobilenetv3_stemlayer import TtMobileNetV3Stem


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_ssd_stem_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    model = pretrained(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    # torch stemlayer
    torch_model = model.features[1]
    torch_model.eval()

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
        state_dict=model.state_dict(),
        base_address=f"features.1",
        device=device,
        host=host,
    )
    tt_model.eval()

    # Run torch model
    input_tensor = torch.randn(1, 16, 112, 112)
    torch_output = torch_model(input_tensor)

    # Run tt model
    tt_stem_input = torch_to_tt_tensor_rm(input_tensor, device)
    tt_output = tt_model(tt_stem_input)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output, host)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("SSDStemlayer Passed!")
    else:
        logger.warning("SSDStemlayer Failed!")

    assert does_pass
