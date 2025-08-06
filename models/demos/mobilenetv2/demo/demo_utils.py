# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import glob
import os
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from torchvision import transforms

from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES


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


def get_batch(data_loader, res):
    resize_size = res.size["shortest_edge"]
    loaded_images = next(data_loader)
    images = None
    labels = []
    for image in loaded_images:
        img = image.image
        labels.append(image.label)
        if img.mode == "L":
            img = img.convert(mode="RGB")
        transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(resize_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize with ImageNet mean and std
            ]
        )

        img = transform(img)
        img = img.unsqueeze(0)  # Add batch dimension: (3, H, W) -> (1, 3, H, W)

        if images is None:
            images = img
        else:
            images = torch.cat((images, img), dim=0)
    return images, labels


# def get_data_loader(input_loc, batch_size, iterations):
#     input_path = Path(input_loc)
#     img_dir = input_path  # already the full path in Civ2

#     # Match image files with typical extensions
#     data_path = str(img_dir / "*.[jJpP][pPnN][gG]")  # .jpg, .jpeg, .png etc.
#     files = glob.glob(data_path)

#     def loader():
#         examples = []
#         for f1 in files:
#             examples.append(
#                 InputExample(
#                     image=get_input(f1),
#                     label=get_label(f1),
#                 )
#             )
#             if len(examples) == batch_size:
#                 yield examples
#                 examples = []
#         if examples:
#             yield examples

#     def loader_hf():
#         examples = []
#         for f1 in files:
#             examples.append(
#                 InputExample(
#                     image=f1["image"],
#                     label=f1["label"],
#                 )
#             )
#             if len(examples) == batch_size:
#                 yield examples
#                 examples = []
#         if examples:
#             yield examples

#     # If no local files found, fallback to HF streaming
#     if len(files) == 0:
#         files_raw = iter(load_dataset("imagenet-1k", split="validation", use_auth_token=True, streaming=True))

#         files = []
#         sample_count = batch_size * iterations
#         for _ in range(sample_count):
#             files.append(next(files_raw))
#         del files_raw

#         return loader_hf()

#     return loader()


def get_data_loader(input_loc, batch_size, iterations, entire_imagenet_dataset=False):
    print("daat loader is called")
    # img_dir = input_loc + "/"
    # data_path = os.path.join(img_dir, "*G")
    img_dir = Path(input_loc)
    extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    files = []
    for ext in extensions:
        files.extend(img_dir.glob(ext))
    files = [str(f) for f in files]  # Convert to string if needed

    print("Length of files:", len(files), "from path:", img_dir, files)

    def loader():
        examples = []
        for f1 in files:
            print("get label and input is called")
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
        print("hf is triggered")
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
