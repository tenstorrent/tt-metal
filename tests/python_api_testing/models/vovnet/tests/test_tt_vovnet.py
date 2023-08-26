import pytest
import timm

import sys
from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../")

import tt_lib
from loguru import logger

from tt_models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
from tt_models.vovnet.tt.vovnet import vovnet_for_image_classification


@pytest.mark.parametrize(
    "model_name, pcc",
    (("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", 0.99),),
)
def test_vovnet_model_inference(pcc, imagenet_sample_input, model_name, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    model = timm.create_model(model_name, pretrained=True)

    torch_model = model

    tt_model = vovnet_for_image_classification(
        device=device,
    )

    input = imagenet_sample_input
    model_output = torch_model(input)

    tt_input = torch_to_tt_tensor_rm(input, device)
    tt_output = tt_model(tt_input)
    tt_output_torch = tt_to_torch_tensor(tt_output)
    tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)

    passing, pcc_message = comp_pcc(model_output, tt_output_torch, pcc)

    logger.info(comp_allclose(model_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if passing:
        logger.info("VoVNet Passed!")
    else:
        logger.warning("VoVNet Failed!")

    assert passing
