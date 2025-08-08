# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import glob
import os

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES


def load_coco_class_names():
    namesfile = "models/demos/utils/coco.names"
    class_names = []
    with open(namesfile, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


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


def get_batch(data_loader, image_processor):
    loaded_images = next(data_loader)
    images = None
    labels = []
    for image in loaded_images:
        img = image.image
        labels.append(image.label)
        if img.mode == "L":
            img = img.convert(mode="RGB")
        img = image_processor(img, return_tensors="pt")
        img = img["pixel_values"]

        if images is None:
            images = img
        else:
            images = torch.cat((images, img), dim=0)
    return images, labels


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
        for _ in tqdm(range(sample_count), desc="Loading samples"):
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


def load_imagenet_dataset(model_location_generator=None, model_version="ImageNet_data"):
    # loads LFC dataset path in CIv2 env
    if model_location_generator is not None and "TT_GH_CI_INFRA" in os.environ:
        dataset_path = (
            model_location_generator("vision-models/mobilenetv2/ImageNet_data", model_subdir="", download_if_ci_v2=True)
            / "data"
        )
    else:
        dataset_path = model_version
    return str(dataset_path)
