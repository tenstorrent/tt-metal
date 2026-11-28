# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The PCC tests' input: a seeded random image, so no test depends on an image file."""

import torch

IMAGE_SIZE = 512
# The per-channel mean and std the pretrained weights were trained with.
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def synthetic_image(seed: int = 0) -> torch.Tensor:
    """A random RGB image as the model takes it: (1, 3, IMAGE_SIZE, IMAGE_SIZE) float32, pixels
    uniform in [0, 1) from ``seed``, normalized with PIXEL_MEAN and PIXEL_STD."""
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    mean = torch.tensor(PIXEL_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(PIXEL_STD).view(1, 3, 1, 1)
    return (pixels - mean) / std
