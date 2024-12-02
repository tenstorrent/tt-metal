# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from PIL import Image
from loguru import logger


def test_cpu_demo():
    image = Image.open("models/sample_data/huggingface_cat_image.jpg")

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    image.save("vit_input_image.jpg")
    logger.info(f"Input image savd as input_image.jpg.")
    logger.info(f"CPU's predicted Output: {model.config.id2label[predicted_label]}.")
