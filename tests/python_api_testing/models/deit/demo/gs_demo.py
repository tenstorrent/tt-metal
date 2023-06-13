from pathlib import Path
import sys
f = f"{Path(__file__).parent}"

sys.path.append(f"{f}")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

from loguru import logger
import torch
from datasets import load_dataset
from modeling_deit import DeitTForImageClassification

model = DeiTForImageClassification(config)
res = model.load_state_dict(state_dict)
model.deit.get_head_mask = get_head_mask

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():

    torch_output = model(**inputs)

# model predicts one of the 1000 ImageNet classes
logits = torch_output[0]
predicted_label = logits.argmax(-1).item()
logger.info(model.config.id2label[predicted_label])


# logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])
