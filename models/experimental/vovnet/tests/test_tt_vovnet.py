# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import timm

from loguru import logger

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_allclose,
    comp_pcc,
)
from models.experimental.vovnet.tt.vovnet import vovnet_for_image_classification


@pytest.mark.parametrize(
    "model_name, pcc",
    (("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", 0.99),),
)
def test_vovnet_model_inference(device, pcc, imagenet_sample_input, model_name, reset_seeds):
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

    if passing:
        logger.info("VoVNet Passed!")
    else:
        logger.warning("VoVNet Failed!")

    assert passing
