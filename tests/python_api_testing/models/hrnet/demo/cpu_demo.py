import torch
from loguru import logger
import pytest

from datasets import load_dataset
import timm


@pytest.mark.parametrize(
    "model_name",
    (("hrnet_w18_small"),),
)
def test_timm_hrnet_image_classification_inference(model_name, imagenet_label_dict, reset_seeds):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    class_labels = imagenet_label_dict

    Timm_model = timm.create_model(model_name, pretrained=True)

    data_config = timm.data.resolve_data_config(model=Timm_model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    inputs = transforms(image).unsqueeze(0)

    with torch.no_grad():
        Timm_output = Timm_model(inputs)

    logger.info("Timm Model answered")
    logger.info(class_labels[Timm_output[0].argmax(-1).item()])
