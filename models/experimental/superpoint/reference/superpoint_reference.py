# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import SuperPointForKeypointDetection, AutoImageProcessor


MODEL_ID = "magic-leap-community/superpoint"
DEFAULT_NATURAL_IMAGE = Path(
    "/home/ttuser/experiments/superpoint/tt-metal/models/sample_data/house_in_field_1080p.jpg"
)


def load_reference_model():
    model = SuperPointForKeypointDetection.from_pretrained(MODEL_ID)
    model.eval()
    return model


def load_image_processor():
    return AutoImageProcessor.from_pretrained(MODEL_ID)


def get_dummy_input(batch_size: int = 1, height: int = 480, width: int = 640):
    torch.manual_seed(0)
    return torch.rand(batch_size, 3, height, width)


def get_natural_input(
    path: Path = DEFAULT_NATURAL_IMAGE,
    batch_size: int = 1,
    height: int = 480,
    width: int = 640,
) -> torch.Tensor:
    """Load and letterbox a natural image into a (B, 3, H, W) tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    transform = T.Compose([T.Resize((height, width)), T.ToTensor()])
    t = transform(img)  # (3, H, W) in [0, 1]
    return t.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
