# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from transformers import AutoImageProcessor, SegformerForImageClassification
import torch
from datasets import load_dataset
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
from loguru import logger
from huggingface_hub import hf_hub_download

import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from models.experimental.functional_segformer.segformer_utils import ade_palette, post_process


def test_demo_image_classification():
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
    model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")

    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    logger.info(model.config.id2label[predicted_label])


def test_demo_semantic_segmentation():
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    repo_id = "hf-internal-testing/fixtures_ade20k"
    image_path = hf_hub_download(repo_id=repo_id, filename="ADE_val_00000001.jpg", repo_type="dataset")
    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(
        logits, size=(512, 512), mode="bilinear", align_corners=False  # (height, width)
    )

    palette = np.array(ade_palette())

    segmentation_mask = post_process(upsampled_logits, palette)

    plt.imshow(segmentation_mask)
    plt.axis("off")

    # Save the visualization as an image file (e.g., PNG)
    plt.savefig("torch_segmentation_mask.png")
