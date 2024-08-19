# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import timm

from loguru import logger

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
from models.experimental.vovnet.tt.conv_norm_act import TtConvNormAct


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_vovnet_conv_norm_act_inference(device, pcc, imagenet_sample_input, reset_seeds):
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

    if passing:
        logger.info("ConvNormAct Passed!")
    else:
        logger.warning("ConvNormAct Failed!")

    assert passing
