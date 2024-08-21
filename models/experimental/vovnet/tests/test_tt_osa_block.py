# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import pytest
import timm

from loguru import logger

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
from models.experimental.vovnet.tt.osa_block import TtOsaBlock


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_osa_block_inference(device, pcc, reset_seeds):
    STAGE_INDEX = 0
    BLOCK_INDEX = 0
    base_address = f"stages.{STAGE_INDEX}.blocks.{BLOCK_INDEX}"
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)

    torch_model = model.stages[STAGE_INDEX].blocks[BLOCK_INDEX]

    tt_model = TtOsaBlock(
        in_chs=1,
        mid_chs=128,
        out_chs=256,
        layer_per_block=3,
        residual=False,
        depthwise=True,
        base_address=base_address,
        state_dict=model.state_dict(),
        device=device,
    )

    # run torch model
    input = torch.randn(1, 64, 56, 56)
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
        logger.info("OsaBlock Passed!")
    else:
        logger.warning("OsaBlock Failed!")

    assert passing
