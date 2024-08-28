# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from PIL import Image
from loguru import logger
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torchvision.utils import save_image

import ttnn

from models.experimental.trocr.tt.trocr import trocr_causal_llm


@pytest.mark.parametrize(
    "model_name",
    (("microsoft/trocr-base-handwritten"),),
)
def test_gs_demo(model_name):
    device = ttnn.open_device(0)

    ttnn.SetDefaultDevice(device)

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    iam_ocr_sample_input = Image.open("models/sample_data/iam_ocr_image.jpg")
    pixel_values = processor(images=iam_ocr_sample_input, return_tensors="pt").pixel_values

    generationmixin = trocr_causal_llm(device)
    with torch.no_grad():
        torch_generated_ids = model.generate(pixel_values)
        tt_generated_ids = generationmixin.generate(pixel_values)

    torch_generated_text = processor.batch_decode(torch_generated_ids, skip_special_tokens=True)[0]
    tt_generated_text = processor.batch_decode(tt_generated_ids, skip_special_tokens=True)[0]
    save_image(pixel_values, "trocr_input_image.jpg")
    logger.info("Image is saved under trocr_input_image.jpg for reference.")

    logger.info("GS's Predicted Output")
    logger.info(tt_generated_text)
    ttnn.close_device(device)
