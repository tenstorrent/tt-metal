from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")


from transformers import AutoImageProcessor
from transformers import ViTForImageClassification as HF_ViTForImageClassication
from loguru import logger
import torch
from datasets import load_dataset
from modeling_vit import ViTForImageClassification

from utility_functions import comp_allclose_and_pcc, comp_pcc

def test_torch_model():
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    HF_model = HF_ViTForImageClassication.from_pretrained("google/vit-base-patch16-224")

    state_dict = HF_model.state_dict()
    config = HF_model.config
    get_head_mask = HF_model.vit.get_head_mask

    model = ViTForImageClassification(config)
    res = model.load_state_dict(state_dict)
    model.vit.get_head_mask = get_head_mask

    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        HF_output = HF_model(**inputs)
        torch_output = model(**inputs)


    passing, info = comp_pcc(HF_output[0], torch_output[0])
    _, info = comp_allclose_and_pcc(HF_output[0], torch_output[0])
    logger.info(info)
    assert passing
