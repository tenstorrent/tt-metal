# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

torchvision = pytest.importorskip("torchvision")
from torchvision import models  # noqa: E402
from torchvision.transforms import InterpolationMode  # noqa: E402


@pytest.fixture(scope="module")
def get_batch():
    # The demo utils import device and dataset packages that the CPU preprocessing does not need.
    stubs = {
        name: mock.MagicMock() for name in ("ttnn", "datasets", "cv2", "tqdm", "loguru") if name not in sys.modules
    }
    with mock.patch.dict(sys.modules, stubs):
        module = importlib.import_module("models.demos.utils.common_demo_utils")
        yield module.get_batch


def _synthetic_image():
    rng = np.random.default_rng(0)
    base = (rng.random((60, 80, 3)) * 255).astype("uint8")
    return Image.fromarray(base).resize((500, 375), Image.BICUBIC)


def _loader(img):
    yield [SimpleNamespace(image=img, label=1)]


@pytest.mark.parametrize(
    "weights, resolution, resize_size, interpolation",
    [
        (models.MobileNet_V2_Weights.IMAGENET1K_V1, 224, 256, None),
        (models.Swin_S_Weights.IMAGENET1K_V1, 224, 246, InterpolationMode.BICUBIC),
        (models.Swin_V2_S_Weights.IMAGENET1K_V1, 256, 260, InterpolationMode.BICUBIC),
    ],
)
def test_get_batch_matches_torchvision_weights_transforms(get_batch, weights, resolution, resize_size, interpolation):
    img = _synthetic_image()
    expected = weights.transforms()(img)
    actual, labels = get_batch(_loader(img), resolution, resize_size=resize_size, interpolation=interpolation)
    assert labels == [1]
    assert torch.equal(actual[0], expected)


def test_get_batch_default_is_unchanged(get_batch):
    img = _synthetic_image()
    from torchvision import transforms as T

    reference = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    actual, _ = get_batch(_loader(img))
    assert torch.equal(actual[0], reference(img))
