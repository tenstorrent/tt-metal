# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torchvision
from datasets import load_dataset
import os
import torch
import glob


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


class InputExample(object):
    def __init__(self, image, label=None):
        self.image = image
        self.label = label


def get_data_loader(input_loc, batch_size, iterations, download_entire_dataset=False):
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
        files_raw = iter(
            load_dataset("imagenet-1k", split="validation", use_auth_token=True, streaming=not download_entire_dataset)
        )
        files = []
        sample_count = batch_size * iterations
        for _ in range(sample_count):
            files.append(next(files_raw))
        del files_raw
        return loader_hf()

    return loader()
