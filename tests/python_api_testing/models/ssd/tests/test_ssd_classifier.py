from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../../..")

import torch
import pytest
from loguru import logger
from torch import nn
from torchvision.models import mobilenet_v3_large as pretrained
from torchvision.models import MobileNet_V3_Large_Weights

import tt_lib
from models.utility_functions_new import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
from models.ssd.tt.ssd_mobilenetv3_classifier import TtClassifier


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_ssd_classifier_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    # Pretrained model
    model = pretrained(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    torch_model = model.classifier

    # Tt ssd_classifier
    tt_model = TtClassifier(state_dict=model.state_dict(), device=device, host=host)

    # Run torch model
    input_tensor = torch.randn(1, 1, 1, 960)
    torch_output = torch_model(input_tensor)

    # Run tt model
    tt_conv_input = torch_to_tt_tensor_rm(input_tensor, device)
    tt_output = tt_model(tt_conv_input)

    # Compare outputs
    tt_output_torch = tt_to_torch_tensor(tt_output, host)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("SSDClassfier Passed!")

    assert does_pass, "SSDClassfier Failed!"
