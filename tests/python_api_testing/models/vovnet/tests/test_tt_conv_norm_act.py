from pathlib import Path
import sys
import torch
import pytest
import timm
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

from tt_models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
import tt_lib
from tt.conv_norm_act import TtConvNormAct


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_vovnet_conv_norm_act_inference(pcc, imagenet_sample_input, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    base_address = f"stem.0"

    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)

    torch_model = model.stem[0]
    tt_model = TtConvNormAct(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        apply_act=True,
        norm_kwargs=None,
        act_kwargs=None,
        state_dict=model.state_dict(),
        base_address=base_address,
        device=device,
        host=host,
    )

    # run torch model
    input = imagenet_sample_input
    model_output = torch_model(input)

    # run tt model
    tt_input = torch_to_tt_tensor_rm(input, device)
    tt_output = tt_model(tt_input)
    tt_output_torch = tt_to_torch_tensor(tt_output)

    # compare output
    passing, pcc_message = comp_pcc(model_output, tt_output_torch, pcc)

    logger.info(comp_allclose(model_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)
    if passing:
        logger.info("ConvNormAct Passed!")
    else:
        logger.warning("ConvNormAct Failed!")

    assert passing
