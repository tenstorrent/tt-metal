# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import timm
import pytest
import torch
from loguru import logger


from models.experimental.hrnet.reference.hrnet import PytorchHighResolutionNet
from models.utility_functions import comp_pcc, comp_allclose_and_pcc


@pytest.mark.parametrize(
    "model_name, pcc",
    (("hrnet_w18_small", 1.0),),
)
def test_hrnet_image_classification_inference(model_name, pcc, imagenet_sample_input, reset_seeds):
    Timm_model = timm.create_model(model_name, pretrained=True)
    state_dict = Timm_model.state_dict()

    PT_model = PytorchHighResolutionNet()
    res = PT_model.load_state_dict(state_dict)

    with torch.no_grad():
        Timm_output = Timm_model(imagenet_sample_input)
        PT_output = PT_model(imagenet_sample_input)

    Timm_logits, PT_logits = Timm_output[0], PT_output[0]

    passing, info = comp_pcc(Timm_logits, PT_logits, pcc)
    _, info = comp_allclose_and_pcc(Timm_logits, PT_logits, pcc)
    logger.info(info)

    if passing:
        logger.info("HRNetForImageClassification Passed!")
    else:
        logger.warning("HRNetForImageClassification Failed!")

    assert passing
