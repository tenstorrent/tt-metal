# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torchvision
from datasets import load_dataset
from loguru import logger
import requests
import os
import logging
import torch


def preprocess():
    """
    Define the transform for the input image/frames.
    Resize, crop, convert to tensor, and apply ImageNet normalization stats.
    """
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform


def download_images(img_path):
    logging.getLogger("datasets").setLevel(logging.ERROR)
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["train"]["image"][0]
    image.save(img_path)
    logger.info(f"Input image saved to {img_path}")


def load_imagenet_labels(imagenet_class_labels_path):
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    os.makedirs(os.path.dirname(imagenet_class_labels_path), exist_ok=True)
    response = requests.get(url)
    if response.status_code == 200:
        with open(imagenet_class_labels_path, "w") as f:
            f.write(response.text)
        logger.info(f"Downloaded to {imagenet_class_labels_path}")
    else:
        logger.info(f"Failed to download: {response.status_code}")

    with open(imagenet_class_labels_path, "r") as f:
        categories = [line.strip() for line in f.readlines() if line.strip()]

    return categories


def get_batch(data_loader):
    loaded_images = next(data_loader)
    images = None
    labels = []
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    for image in loaded_images:
        img = image.image
        labels.append(image.label)
        if img.mode == "L":
            img = img.convert(mode="RGB")
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        if images is None:
            images = img
        else:
            images = torch.cat((images, img), dim=0)

    return images, labels
