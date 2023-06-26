from transformers import AutoImageProcessor, DeiTForImageClassificationWithTeacher
import torch
from datasets import load_dataset
from loguru import logger


def test_cpu_demo():
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model = DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    predicted_label = logits.argmax(-1).item()
    image.save("deit_with_teacher_cpu_input_image.jpg")
    logger.info(f"Input image saved as deit_with_teacher_cpu_input_image.jpg")
    logger.info(f"TT's prediction: {model.config.id2label[predicted_label]}.")
