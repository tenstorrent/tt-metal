# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import glob

from datasets import load_dataset
from PIL import Image

from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES


def get_input(image_path):
    img = Image.open(image_path)
    return img


class InputExample(object):
    def __init__(self, image, label=None):
        self.image = image
        self.label = label


def get_label(image_path):
    _, image_name = image_path.rsplit("/", 1)
    image_name_exact, _ = image_name.rsplit(".", 1)
    _, label_id = image_name_exact.rsplit("_", 1)
    label = list(IMAGENET2012_CLASSES).index(label_id)
    return label


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
