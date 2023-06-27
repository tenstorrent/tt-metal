from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import torch
import pytest
import timm
import tt_lib
from loguru import logger

from utility_functions_new import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
from tt.separable_conv_norm_act import (
    TtSeparableConvNormAct,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_separable_conv_norm_act_inference(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    base_address = f"stem.1"

    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)

    torch_model = model.stem[1]

    tt_model = TtSeparableConvNormAct(
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding=1,
        bias=False,
        groups=64,
        channel_multiplier=1.0,
        state_dict=model.state_dict(),
        base_address=base_address,
        device=device,
        host=host,
    )

    input = torch.randn(1, 64, 112, 112)

    # run torch model
    model_output = torch_model(input)

    # run tt model
    tt_input = torch_to_tt_tensor_rm(input, device)
    tt_output = tt_model(tt_input)
    tt_output_torch = tt_to_torch_tensor(tt_output, host)

    # compare result
    passing, pcc_message = comp_pcc(model_output, tt_output_torch, pcc)

    logger.info(comp_allclose(model_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if passing:
        logger.info("SeparableConvNormAct Passed!")
    else:
        logger.warning("SeparableConvNormAct Failed!")

    assert passing
