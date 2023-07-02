import torch
import pytest
from loguru import logger
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests


@pytest.mark.parametrize(
    "model_name",
    (("microsoft/trocr-base-handwritten"),),
)
def test_cpu_demo(model_name, reset_seeds):

    url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    image.save("image.jpg")
    logger.info("Image is saved under image.jpg for reference.")

    logger.info("HF Model answered")
    logger.info(generated_text)
