# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger


import timm
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_allclose,
    comp_pcc,
)
from models.experimental.hrnet.tt.hrnet_model import hrnet_w18_small


@pytest.mark.parametrize(
    "model_name, pcc",
    (("hrnet_w18_small", 0.99),),
)
def test_hrnet_model_inference(device, model_name, pcc, imagenet_sample_input, reset_seeds):
    torch_model = timm.create_model(model_name, pretrained=True)

    tt_model = hrnet_w18_small(device, multi_scale_output=True)

    torch_output = torch_model(imagenet_sample_input)

    tt_input = torch_to_tt_tensor_rm(imagenet_sample_input, device, put_on_device=False)
    tt_output = tt_model(tt_input)

    tt_output_torch = tt_to_torch_tensor(tt_output).view(1, -1)

    passing, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc)
    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("HRNetForImageClassification Passed!")
    else:
        logger.warning("HRNetForImageClassification Failed!")

    assert passing
