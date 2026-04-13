# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests
from PIL import Image
from transformers import AutoProcessor, CLIPModel

from tests.ttnn.utils_for_testing import assert_with_pcc

MODEL_NAME = "openai/clip-vit-base-patch32"

# PCC tests run at this batch size. Optimized model sharding configs are built
# for this value, so the inputs fixture must produce exactly this many samples.
PCC_BATCH_SIZE = 7

# A small pool the perf tests can draw from; PCC only uses the first entry.
IMAGE_URLS = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000037777.jpg",
    "http://images.cocodataset.org/val2017/000000252219.jpg",
    "http://images.cocodataset.org/val2017/000000087038.jpg",
    "http://images.cocodataset.org/val2017/000000174482.jpg",
    "http://images.cocodataset.org/val2017/000000403385.jpg",
    "http://images.cocodataset.org/val2017/000000006818.jpg",
]

TEXT_QUERIES = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
    "a photo of a car",
    "a person riding a bicycle",
    "a group of people playing sports",
    "food on a dining table",
    "a scenic landscape with mountains",
    "an airplane flying in the sky",
    "a street scene in a city",
    "a boat on the water",
    "a child playing with a toy",
    "a kitchen with modern appliances",
    "a sunset over the ocean",
    "a horse running in a field",
    "an office workspace with a computer",
]

PCC_THRESHOLD = 0.98
OPT_PCC_THRESHOLD = 0.90


def pcc_check(torch_output, ttnn_output, pcc=PCC_THRESHOLD):
    assert_with_pcc(torch_output, ttnn_output, pcc=pcc)


def opt_pcc_check(torch_output, ttnn_output, pcc=OPT_PCC_THRESHOLD):
    assert_with_pcc(torch_output, ttnn_output, pcc=pcc)


@pytest.fixture(scope="session")
def torch_model():
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model


@pytest.fixture(scope="session")
def processor():
    return AutoProcessor.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="session")
def inputs(processor):
    """Tokenized inputs with PCC_BATCH_SIZE images + text queries."""
    images = []
    texts = []
    for i in range(PCC_BATCH_SIZE):
        url = IMAGE_URLS[i % len(IMAGE_URLS)]
        images.append(Image.open(requests.get(url, stream=True).raw))
        text = TEXT_QUERIES[i % len(TEXT_QUERIES)]
        texts.append(text)
    return processor(
        texts,
        images,
        padding="max_length",
        max_length=77,
        return_tensors="pt",
    )
