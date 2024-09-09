# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from PIL import Image
import torch
import os
import glob
from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES
from datasets import load_dataset
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import torch


class InputExample(object):
    def __init__(self, image, label=None):
        self.image = image
        self.label = label


def get_input(image_path):
    img = Image.open(image_path)
    return img


def get_label(image_path):
    _, image_name = image_path.rsplit("/", 1)
    image_name_exact, _ = image_name.rsplit(".", 1)
    _, label_id = image_name_exact.rsplit("_", 1)
    label = list(IMAGENET2012_CLASSES).index(label_id)
    return label


preprocess = transforms.Compose(
    [
        transforms.Resize(256),  # Resize the shorter side to 256 pixels
        transforms.CenterCrop(224),  # Crop the center to 224x224 pixels
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(  # Normalize using ImageNet's mean and std
            mean=[0.485, 0.456, 0.406],  # These are the mean values for each channel
            std=[0.229, 0.224, 0.225],  # These are the std values for each channel
        ),
    ]
)


def get_batch(data_loader):
    loaded_images = next(data_loader)
    images = None
    labels = []
    transform = transforms.ToTensor()
    resize_transform = transforms.Resize((224, 224))
    for image in loaded_images:
        img = image.image
        labels.append(image.label)
        if img.mode == "L":
            img = img.convert(mode="RGB")

        img = preprocess(img)
        img = img.to(torch.bfloat16)
        img = img.unsqueeze(0)
        if images is None:
            images = img
        else:
            images = torch.cat((images, img), dim=0)
    return images, labels


def get_data_loader(input_loc, batch_size, iterations):
    img_dir = input_loc + "/"
    data_path = os.path.join(img_dir, "*G")
    files = glob.glob(data_path)

    def loader():
        examples = []
        for f1 in files:
            examples.append(
                InputExample(
                    image=get_input(f1),
                    label=get_label(f1),
                )
            )
            if len(examples) == batch_size:
                yield examples
                del examples
                examples = []

    def loader_hf():
        examples = []
        for f1 in files:
            examples.append(
                InputExample(
                    image=f1["image"],
                    label=f1["label"],
                )
            )
            if len(examples) == batch_size:
                yield examples
                del examples
                examples = []

    if len(files) == 0:
        files_raw = iter(load_dataset("imagenet-1k", split="validation", use_auth_token=True, streaming=True))
        files = []
        sample_count = batch_size * iterations
        for _ in range(sample_count):
            files.append(next(files_raw))
        del files_raw
        return loader_hf()

    return loader()


def get_data(input_loc):
    img_dir = input_loc + "/"
    data_path = os.path.join(img_dir, "*G")
    files = sorted(glob.glob(data_path))
    examples = []
    for f1 in files:
        examples.append(
            InputExample(
                image=get_input(f1),
                label=get_label(f1),
            )
        )
    image_examples = examples

    return image_examples
