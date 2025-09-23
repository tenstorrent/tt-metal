# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import AutoImageProcessor, DeiTForImageClassificationWithTeacher
from loguru import logger


def test_cpu_demo(hf_cat_image_sample_input):
    image = hf_cat_image_sample_input

    image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model = DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    predicted_label = logits.argmax(-1).item()
    image.save("deit_with_teacher_cpu_input_image.jpg")
    logger.info(f"Input image saved as deit_with_teacher_cpu_input_image.jpg")
    logger.info(f"CPU's prediction: {model.config.id2label[predicted_label]}.")
