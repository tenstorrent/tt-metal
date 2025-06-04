# SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ast

import pytest
import torchvision.transforms as transforms
from PIL import Image


def pytest_addoption(parser):
    parser.addoption(
        "--start-from",
        action="store",
        default=0,
        help="Start from prompt number (0-4999)",
    )
    parser.addoption(
        "--num-prompts",
        action="store",
        default=5000,
        help="Number of prompts to process (default: 5000)",
    )


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


@pytest.fixture
def sdxl_accuracy_parameters(request):
    start_from = request.config.getoption("--start-from")
    num_prompts = request.config.getoption("--num-prompts")
    if start_from is not None:
        start_from = int(start_from)
    else:
        start_from = 0

    if num_prompts is not None:
        num_prompts = int(num_prompts)
    else:
        num_prompts = 5000

    return start_from, num_prompts
