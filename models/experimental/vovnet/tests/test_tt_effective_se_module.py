# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import timm

from loguru import logger

from models.experimental.vovnet.tt.effective_se_module import TtEffectiveSEModule
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_effective_se_module_inference(device, pcc, reset_seeds):
    base_address = f"stages.0.blocks.0.attn"

    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)

    torch_model = model.stages[0].blocks[0].attn

    tt_model = TtEffectiveSEModule(
        in_channels=256,
        out_channels=256,
        kernel_size=1,
        stride=1,
        dilation=1,
        padding=0,
        bias=None,
        state_dict=model.state_dict(),
        base_address=base_address,
        device=device,
    )

    # run torch model
    input = torch.randn(1, 256, 56, 56)
    model_output = torch_model(input)

    # run tt model
    tt_input = torch_to_tt_tensor_rm(input, device)
    tt_output = tt_model(tt_input)
    tt_output_torch = tt_to_torch_tensor(tt_output)
    tt_output_torch = tt_output_torch.squeeze(0)

    # compare output
    passing, pcc_message = comp_pcc(model_output, tt_output_torch, pcc)

    logger.info(comp_allclose(model_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("EffectiveSEModule Passed!")
    else:
        logger.warning("EffectiveSEModule Failed!")

    assert passing
