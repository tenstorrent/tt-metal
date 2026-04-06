# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from transformers import AutoFeatureExtractor
from transformers import SwinForImageClassification as HF_SwinForImageClassification


@pytest.mark.parametrize(
    "model_name",
    (("microsoft/swin-tiny-patch4-window7-224"),),
)
def test_cpu_demo(model_name, imagenet_sample_input, reset_seeds):
    image = imagenet_sample_input

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    HF_model = HF_SwinForImageClassification.from_pretrained(model_name)

    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        torch_output = HF_model(**inputs)

    logits = torch_output[0]
    predicted_label = logits.argmax(-1).item()
    logger.info("HF Model answered")
    logger.info(HF_model.config.id2label[predicted_label])
