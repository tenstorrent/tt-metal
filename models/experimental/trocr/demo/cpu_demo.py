# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from PIL import Image
from loguru import logger
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torchvision.utils import save_image


@pytest.mark.parametrize(
    "model_name",
    (("microsoft/trocr-base-handwritten"),),
)
def test_cpu_demo(model_name):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    iam_ocr_sample_input = Image.open("models/sample_data/iam_ocr_image.jpg")
    pixel_values = processor(images=iam_ocr_sample_input, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    save_image(pixel_values, "trocr_input_image.jpg")
    logger.info("Image is saved under trocr_input_image.jpg for reference.")

    logger.info("HF Model answered")
    logger.info(generated_text)
