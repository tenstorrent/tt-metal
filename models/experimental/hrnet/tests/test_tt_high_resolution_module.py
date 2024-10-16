# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger


import timm
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_allclose,
    comp_pcc,
)

from models.experimental.hrnet.tt.high_resolution_module import (
    TtHighResolutionModule,
)
from models.experimental.hrnet.tt.basicblock import TtBasicBlock


@pytest.mark.parametrize(
    "model_name, pcc",
    (("hrnet_w18_small", 0.99),),
)
def test_hrnet_module_inference(device, model_name, pcc, reset_seeds):
    HR_MODULE_INDEX = 0
    base_address = f"stage2.{HR_MODULE_INDEX}"
    model = timm.create_model(model_name, pretrained=True)

    # Torch BasicBlock
    torch_model = model.stage2[HR_MODULE_INDEX]

    # Tt BasicBlock
    tt_model = TtHighResolutionModule(
        num_branches=2,
        block=TtBasicBlock,
        num_blocks=[2, 2],
        num_inchannels=[16, 32],
        num_channels=[16, 32],
        fuse_method="SUM",
        state_dict=model.state_dict(),
        base_address=base_address,
        device=device,
        multi_scale_output=True,
    )

    # As we are testing Stage2, it requires two inputs for two inchannels
    inputs = [torch.rand(1, 16, 56, 56), torch.rand(1, 32, 28, 28)]

    torch_outputs = torch_model(inputs)

    tt_inputs = [torch_to_tt_tensor_rm(inputs[i], device, put_on_device=False) for i in range(len(inputs))]
    tt_outputs = tt_model(tt_inputs)

    tt_outputs_torch = [tt_to_torch_tensor(tt_outputs[i]) for i in range(len(tt_outputs))]

    does_pass_list = []
    for i in range(len(tt_outputs_torch)):
        does_pass, pcc_message = comp_pcc(torch_outputs[i], tt_outputs_torch[i], pcc)
        does_pass_list.append(does_pass)
        logger.info(comp_allclose(torch_outputs[i], tt_outputs_torch[i]))
        logger.info(pcc_message)

    if all(does_pass_list):
        logger.info("HighResolutionModule Passed!")
    else:
        logger.warning("HighResolutionModule Failed!")

    assert all(does_pass_list)
