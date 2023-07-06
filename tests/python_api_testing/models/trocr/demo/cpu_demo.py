import torch
import pytest
from loguru import logger
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torchvision.utils import save_image


@pytest.mark.parametrize(
    "model_name",
    (("microsoft/trocr-base-handwritten"),),
)
def test_cpu_demo(model_name, iam_ocr_sample_input, reset_seeds):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    pixel_values = processor(
        images=iam_ocr_sample_input, return_tensors="pt"
    ).pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    save_image(iam_ocr_sample_input, "trocr_input_image.jpg")
    logger.info("Image is saved under trocr_input_image.jpg for reference.")

    logger.info("HF Model answered")
    logger.info(generated_text)
