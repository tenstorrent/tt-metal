# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ast

import pytest
import torchvision.transforms as transforms
from PIL import Image


@pytest.fixture
def imagenet_label_dict():
    path = "models/sample_data/imagenet_class_labels.txt"
    with open(path, "r") as file:
        class_labels = ast.literal_eval(file.read())
    return class_labels


@pytest.fixture
def imagenet_sample_input():
    path = "models/sample_data/ILSVRC2012_val_00048736.JPEG"

    im = Image.open(path)
    im = im.resize((224, 224))
    return transforms.ToTensor()(im).unsqueeze(0)


@pytest.fixture
def mnist_sample_input():
    path = "models/sample_data/torchvision_mnist_digit_7.jpg"
    im = Image.open(path)
    return im


@pytest.fixture
def iam_ocr_sample_input():
    path = "models/sample_data/iam_ocr_image.jpg"
    im = Image.open(path)
    return im


@pytest.fixture
def hf_cat_image_sample_input():
    path = "models/sample_data/huggingface_cat_image.jpg"
    im = Image.open(path)
    return im
