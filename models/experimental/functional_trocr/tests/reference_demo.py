# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from PIL import Image
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from models.experimental.functional_trocr.trocr_generate_utils import generate
from models.experimental.functional_trocr.reference import functional_torch_vit


@pytest.mark.parametrize("model_name", ["microsoft/trocr-base-handwritten"])
def test_demo(model_name):
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    processor = TrOCRProcessor.from_pretrained(model_name)
    iam_ocr_sample_input = Image.open("models/sample_data/iam_ocr_image.jpg")
    pixel_values = processor(images=iam_ocr_sample_input, return_tensors="pt").pixel_values

    model_state_dict = model.state_dict()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=functional_torch_vit.custom_preprocessor,
    )

    generated_ids = generate(
        model.config,
        model.generation_config,
        pixel_values,
        parameters,
        model_state_dict["encoder.embeddings.cls_token"],
        model_state_dict["encoder.embeddings.position_embeddings"],
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    logger.info("Model answered")
    logger.info(generated_text)
