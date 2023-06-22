from transformers import AutoImageProcessor, DeiTForImageClassification
import torch
from PIL import Image
import requests


def test_cpu_demo():
    torch.manual_seed(3)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    print("\ncpu predicted class:\n", model.config.id2label[predicted_class_idx])
