# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
import pytest

from models.experimental.swin.reference.swin import PytorchSwinForImageClassification

from models.utility_functions import comp_pcc, comp_allclose_and_pcc
from transformers import AutoFeatureExtractor
from transformers import SwinForImageClassification as HF_SwinForImageClassification


@pytest.mark.parametrize(
    "model_name, pcc",
    (("microsoft/swin-tiny-patch4-window7-224", 0.99),),
)
def test_swin_ic_inference(model_name, pcc, imagenet_sample_input, reset_seeds):
    image = imagenet_sample_input

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    HF_model = HF_SwinForImageClassification.from_pretrained(model_name)

    state_dict = HF_model.state_dict()
    config = HF_model.config
    get_head_mask = HF_model.swin.get_head_mask

    PT_model = PytorchSwinForImageClassification(config)
    _ = PT_model.load_state_dict(state_dict)
    PT_model.swin.get_head_mask = get_head_mask

    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        HF_output = HF_model(**inputs)
        torch_output = PT_model(**inputs)

    passing, info = comp_pcc(HF_output[0], torch_output.logits, pcc)
    _, info = comp_allclose_and_pcc(HF_output[0], torch_output.logits)
    logger.info(info)

    torch_logits = torch_output.logits
    HF_logits = HF_output[0]

    torch_predicted_label = torch_logits.argmax(-1).item()
    HF_predicted_label = HF_logits.argmax(-1).item()

    logger.info("PT Model answered")
    logger.info(PT_model.config.id2label[torch_predicted_label])

    logger.info("Torch Model answered")
    logger.info(HF_model.config.id2label[HF_predicted_label])

    if passing:
        logger.info("SwinForImageClassification Passed!")
    else:
        logger.warning("SwinForImageClassification Failed!")

    assert passing
