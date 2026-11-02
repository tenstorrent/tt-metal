# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Deterministic synthetic images for the vision tower tests: a seed and a size give the same uint8 pixels on every
host (torch CPU generator; only arithmetic, ``randn`` and comparisons), so the device and the CPU reference see one
input without a binary fixture in the tree.  ``PINNED_FIXTURES`` names the two grids the tests use and the SHA-256
of their pixel bytes; ``pixel_patches`` runs the module's preprocessing (no resize: the sides are multiples of 32)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from models.demos.blackhole.qwen38_flash_next.vision_reference import VisionTowerConfig, preprocess_image, smart_resize


@dataclass(frozen=True)
class SyntheticImageSpec:
    name: str
    seed: int
    height: int
    width: int
    sha256: str  # of the uint8 [3, H, W] pixel bytes
    reference: bool = True  # False: too large for the CPU reference; the device test measures time and memory only

    @property
    def resized(self) -> tuple[int, int]:
        """The processor's target size (an image under the minimum pixel count is upscaled)."""

        config = VisionTowerConfig()
        return smart_resize(
            self.height,
            self.width,
            factor=config.resize_factor,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )

    @property
    def grid(self) -> tuple[int, int]:
        height, width = self.resized
        return height // 16, width // 16

    @property
    def patches(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def merged_tokens(self) -> int:
        return self.patches // 4


# grid32: 32x32 patches = 1024 patches = 256 merged tokens; grid64: 64x64 = 4096 patches = 1024 tokens;
# grid24x40: a non-square 24x40 grid = 960 patches = 240 tokens (a tile multiple, no padding);
# grid22x18: 22x18 = 396 patches = 99 tokens, padded to 416 rows on the device (the windowed attention path);
# grid128 / grid256: 16,384 and 65,536 patches (the stock maximum, 4096x4096 px): device time and memory only.
PINNED_FIXTURES: tuple[SyntheticImageSpec, ...] = (
    SyntheticImageSpec("grid32", 38, 512, 512, "1ed818d48bb850d5158ad7fc02303c8b533feb43002633fb9dd5e6dd65f02f6e"),
    SyntheticImageSpec("grid64", 64, 1024, 1024, "4013185ad4db9d706334a59f7aa9297903e7d10914176e7aa49eb04f3c128393"),
    SyntheticImageSpec("grid24x40", 2440, 384, 640, "14d309cb8f279d51e20752c6c44588a76f97117b670ea7f9d2ce90d43e095bd8"),
    SyntheticImageSpec("grid22x18", 2218, 352, 288, "0f3121dd8729492570ecdd161e9a1cbd9062fcf5f27aee089abf0fad278e3125"),
    SyntheticImageSpec(
        "grid128", 128, 2048, 2048, "db3af80ba2622d14d9fba13e25c28163e9c13834821af86ab3bc6873467d2375", reference=False
    ),
    SyntheticImageSpec(
        "grid256", 256, 4096, 4096, "f3f0011ebee211c371c8f67a0cb2277f2c95ad5a83b2f7a2f8e6605671a14360", reference=False
    ),
)


def synthetic_image(seed: int, height: int, width: int) -> torch.Tensor:
    """``[3, H, W]`` uint8: colour gradients, seeded rectangles and discs, seeded noise."""

    generator = torch.Generator().manual_seed(seed)
    ys = torch.linspace(0.0, 1.0, height).unsqueeze(1).expand(height, width)
    xs = torch.linspace(0.0, 1.0, width).unsqueeze(0).expand(height, width)
    red = 0.55 * xs + 0.25 * (1 - ys)
    green = 0.15 + 0.5 * ys * (1 - xs)
    blue = 0.6 * (0.5 + 0.5 * torch.sin(6.0 * xs + 3.0 * ys))
    image = torch.stack([red, green, blue], dim=0)
    for _ in range(6):
        params = torch.rand(7, generator=generator)
        cy, cx = params[0] * height, params[1] * width
        h_half, w_half = (0.05 + 0.2 * params[2]) * height, (0.05 + 0.2 * params[3]) * width
        colour = params[4:7].reshape(3, 1, 1)
        rows = torch.arange(height, dtype=torch.float32).unsqueeze(1)
        cols = torch.arange(width, dtype=torch.float32).unsqueeze(0)
        if params[0] < 0.5:  # rectangle
            mask = ((rows - cy).abs() <= h_half) & ((cols - cx).abs() <= w_half)
        else:  # disc
            mask = ((rows - cy) / h_half) ** 2 + ((cols - cx) / w_half) ** 2 <= 1.0
        image = torch.where(mask.unsqueeze(0), 0.7 * colour + 0.3 * image, image)
    noise = torch.randn(3, height, width, generator=generator) * (12.0 / 255.0)
    return ((image + noise).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def image_sha256(image_uint8: torch.Tensor) -> str:
    return hashlib.sha256(image_uint8.contiguous().numpy().tobytes()).hexdigest()


def fixture(name: str) -> SyntheticImageSpec:
    for spec in PINNED_FIXTURES:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown vision fixture {name!r}; known: {[spec.name for spec in PINNED_FIXTURES]}")


def fixture_image(spec: SyntheticImageSpec, *, verify: bool = True) -> torch.Tensor:
    image = synthetic_image(spec.seed, spec.height, spec.width)
    if verify:
        digest = image_sha256(image)
        if digest != spec.sha256:
            raise ValueError(f"fixture {spec.name}: pixel sha256 {digest} differs from the pin {spec.sha256}")
    return image


def pixel_patches(
    spec: SyntheticImageSpec, config: VisionTowerConfig = VisionTowerConfig(), *, verify: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """(``[N, 1536]`` FP32 pixel patches, ``grid_thw`` ``[1, 3]``) of a pinned fixture."""

    return preprocess_image(fixture_image(spec, verify=verify), config)
