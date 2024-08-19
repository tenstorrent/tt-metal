# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import timm

from loguru import logger

from models.experimental.vovnet.tt.classifier_head import TtClassifierHead
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
def test_classifier_head_inference(device, pcc, reset_seeds):
    base_address = f"head"
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)

    torch_model = model.head

    tt_model = TtClassifierHead(
        in_features=1024,
        num_classes=1000,
        pool_type="avg",
        use_conv=False,
        input_fmt="NCHW",
        device=device,
        base_address=base_address,
        state_dict=model.state_dict(),
    )

    # run torch model
    input = torch.randn(1, 1024, 7, 7)
    model_output = torch_model(input)

    # run tt model
    tt_input = torch_to_tt_tensor_rm(input, device, put_on_device=False)
    tt_output = tt_model(tt_input)
    tt_output_torch = tt_to_torch_tensor(tt_output)
    tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)

    # compare output
    passing, pcc_message = comp_pcc(model_output, tt_output_torch, pcc)

    logger.info(comp_allclose(model_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("ClassifierHead Passed!")
    else:
        logger.warning("ClassifierHead Failed!")

    assert passing
