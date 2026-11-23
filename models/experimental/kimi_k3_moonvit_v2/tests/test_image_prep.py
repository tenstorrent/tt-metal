# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU: `tt.image_prep` (image -> patches, layer-1 padding) against the HF processor copied into
`reference/`, and its hand-off to `host_prep.prepare_host_inputs` (layer-2 padding)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from models.experimental.kimi_k3_moonvit_v2.reference.kimi_k3_vision_processing import KimiK3VisionProcessor
from models.experimental.kimi_k3_moonvit_v2.reference.media_utils import navit_resize_image
from models.experimental.kimi_k3_moonvit_v2.tt import host_prep, image_prep
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import IN_PATCH_LIMIT, MAX_PROCESSOR_PATCHES, RING_SIZE

PREPROCESSOR_CONFIG = Path(__file__).parents[1] / "reference" / "preprocessor_config.json"


@pytest.fixture(scope="module")
def media_proc_cfg() -> dict:
    return json.loads(PREPROCESSOR_CONFIG.read_text())["media_proc_cfg"]


@pytest.fixture(scope="module")
def processor(media_proc_cfg) -> KimiK3VisionProcessor:
    return KimiK3VisionProcessor(media_proc_cfg=media_proc_cfg)


def hf_preprocess(processor: KimiK3VisionProcessor, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    # A new dict per call: HF overwrites media["image"] in place.
    out = processor.preprocess([{"type": "image", "image": image}])
    return out["pixel_values"], out["grid_thws"]


def random_image(width: int, height: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.frombytes("RGB", (width, height), rng.bytes(3 * width * height))


def hf_plan(width: int, height: int, cfg: dict) -> image_prep.ResizePlan:
    r = navit_resize_image(
        width,
        height,
        cfg["patch_size"],
        cfg["merge_kernel_size"],
        cfg["in_patch_limit"],
        cfg["patch_limit_on_one_side"],
        cfg["fixed_output_tokens"],
    )
    p, m = cfg["patch_size"], cfg["merge_kernel_size"]
    return image_prep.ResizePlan(
        r["new_width"],
        r["new_height"],
        r["pad_width"],
        r["pad_height"],
        (r["new_height"] + r["pad_height"]) // p,
        (r["new_width"] + r["pad_width"]) // p,
    )


# (width, height) -> padded grid (h, w)
KNOWN_GRIDS = {
    (1889, 7377): (510, 132),  # the largest HF grid, 67,320 patches
    (3590, 3590): (258, 258),  # over `in_patch_limit` from padding alone
    (6000, 4000): (210, 316),  # area-limited, over `in_patch_limit`
    (4662, 4662): (256, 256),  # area-limited, within `in_patch_limit`
    (4032, 3024): (216, 288),
    (3840, 2160): (156, 276),
    (1920, 1080): (78, 138),
    (8000, 300): (20, 512),
    (35, 35): (4, 4),
}
EDGE_SIDES = (1, 27, 28, 29, 7168, 7169)


def test_resize_plan_matches_hf(config, media_proc_cfg):
    rng = np.random.default_rng(0)
    sizes = [tuple(map(int, s)) for s in rng.integers(1, 20001, size=(20000, 2))]
    sizes += list(KNOWN_GRIDS) + [(a, b) for a in EDGE_SIDES for b in EDGE_SIDES]
    for width, height in sizes:
        assert image_prep.resize_plan(width, height, config) == hf_plan(width, height, media_proc_cfg), (width, height)
    for (width, height), grid in KNOWN_GRIDS.items():
        plan = image_prep.resize_plan(width, height, config)
        assert (plan.grid_h, plan.grid_w) == grid, (width, height)


def test_max_patches_holds_the_largest_grid(config):
    """The default `max_patches` holds the largest grid HF makes for any image size, with the least
    padding `resolve_parallel` allows."""
    plan = image_prep.resize_plan(1889, 7377, config)
    assert plan.grid_h * plan.grid_w == MAX_PROCESSOR_PATCHES
    assert all(h * w <= MAX_PROCESSOR_PATCHES for h, w in KNOWN_GRIDS.values())
    row_multiple = RING_SIZE * config.merge_area * 32
    assert config.max_patches % row_multiple == 0
    assert 0 <= config.max_patches - MAX_PROCESSOR_PATCHES < row_multiple


@pytest.mark.parametrize("size, grid", [((6000, 4000), "210x316"), ((3590, 3590), "258x258")], ids=["area", "padding"])
def test_rejects_over_limit(config, expect_error, monkeypatch, size, grid):
    """With a `max_patches` under the processor's largest grid, an image over it is rejected before the resize."""
    config = dataclasses.replace(config, max_patches=IN_PATCH_LIMIT)
    image = Image.new("RGB", size)

    def no_resize(*args, **kwargs):
        raise AssertionError("the grid check must come before the resize")

    monkeypatch.setattr(Image.Image, "resize", no_resize)
    with expect_error(ValueError, f"image {size[0]}x{size[1]}: grid {grid} exceeds .* {config.max_patches} patches"):
        image_prep.image_to_patches(image, config)


UNDER_LIMIT = ((4032, 3024), (3584, 3584), (4662, 4662), (8000, 300), (6000, 4000), (3590, 3590), (1889, 7377))


@pytest.mark.parametrize("size", UNDER_LIMIT, ids=[f"{w}x{h}" for w, h in UNDER_LIMIT])
def test_accepts_under_limit(config, size):
    pixel_values, grid_thws = image_prep.image_to_patches(Image.new("RGB", size), config)
    _, h, w = grid_thws[0].tolist()
    assert h * w <= config.max_patches
    assert pixel_values.shape == (h * w, 3, config.patch_size, config.patch_size)


NON_RGB_MODES = ("RGBA", "LA", "PA", "L", "P", "1", "CMYK", "I;16")


@pytest.mark.parametrize("mode", NON_RGB_MODES)
def test_rejects_non_rgb(config, expect_error, monkeypatch, mode):
    def no_resize(*args, **kwargs):
        raise AssertionError("the mode check must come before the resize")

    monkeypatch.setattr(Image.Image, "resize", no_resize)
    with expect_error(ValueError, f"image mode '{mode}' is not supported"):
        image_prep.image_to_patches(Image.new(mode, (35, 35)), config)


@pytest.mark.parametrize(
    "pixels,message",
    [
        (np.zeros((28, 28, 3), np.float32), "uint8"),
        (np.zeros((28, 28), np.uint8), "uint8"),
        (np.zeros((28, 28, 4), np.uint8), "uint8"),
        (np.zeros((28, 42, 3), np.uint8), "W of 28"),  # whole patches, not whole merge windows
        (np.zeros((14, 28, 3), np.uint8), "H a multiple of 28"),
    ],
)
def test_pixels_to_patches_rejects(config, expect_error, pixels, message):
    with expect_error(ValueError, message):
        image_prep.pixels_to_patches(pixels, config)


SIZES = ((1, 1), (35, 35), (100, 57), (1920, 1080), (8000, 300), (300, 8000), (4662, 4662))


@pytest.mark.parametrize("size", SIZES, ids=[f"{w}x{h}" for w, h in SIZES])
def test_image_to_patches_matches_hf(config, processor, size):
    image = random_image(*size)
    pixel_values, grid_thws = image_prep.image_to_patches(image, config)
    hf_pixel_values, hf_grid_thws = hf_preprocess(processor, image)
    assert grid_thws.dtype == hf_grid_thws.dtype and torch.equal(grid_thws, hf_grid_thws)
    assert pixel_values.dtype == hf_pixel_values.dtype and torch.equal(pixel_values, hf_pixel_values)


def test_normalization_matches_hf_for_every_value(config):
    """pixels_to_patches normalizes in float32 where HF divides in float64 and casts: the same for all 256 values."""
    pixels = np.resize(np.arange(256, dtype=np.uint8), (56, 56, 3))  # every value, repeated
    expected = (pixels / 255.0).astype(np.float32)
    expected -= image_prep.IMAGE_MEAN
    expected *= 1.0 / image_prep.IMAGE_STD
    pixel_values, _ = image_prep.pixels_to_patches(pixels, config)
    raster = pixel_values.reshape(4, 4, 3, 14, 14).permute(0, 3, 1, 4, 2).reshape(56, 56, 3)
    assert torch.equal(raster, torch.from_numpy(expected))


@pytest.mark.parametrize("size", SIZES, ids=[f"{w}x{h}" for w, h in SIZES])
def test_resize_array_matches_resize_image(config, size):
    image = random_image(*size)
    expected = image_prep.resize_image(image, config)
    actual = image_prep.resize_array(np.asarray(image), config)
    assert actual.dtype == np.uint8 and np.array_equal(actual, expected)


def test_resize_array_skips_pil_at_scale_one(config, monkeypatch):
    """An image the plan does not shrink is only padded: at scale 1 Pillow's resize returns a copy."""

    def no_resize(*args, **kwargs):
        raise AssertionError("an image that needs no resize must not go through PIL")

    pixels = np.asarray(random_image(100, 57))
    monkeypatch.setattr(Image.Image, "resize", no_resize)
    monkeypatch.setattr(Image, "fromarray", no_resize)
    padded = image_prep.resize_array(pixels, config)
    assert padded.shape == (84, 112, 3)
    assert np.array_equal(padded[:57, :100], pixels) and not padded[57:].any() and not padded[:, 100:].any()


@pytest.mark.parametrize(
    "media",
    [
        np.zeros((28, 28, 3), np.float32),
        np.zeros((28, 28), np.uint8),
        np.zeros((28, 28, 4), np.uint8),
        np.zeros((0, 28, 3), np.uint8),
    ],
    ids=["float32", "2d", "rgba", "empty"],
)
def test_resize_array_rejects(config, expect_error, media):
    with expect_error(ValueError, "expected a non-empty uint8 RGB"):
        image_prep.resize_array(media, config)


def test_padding_is_black(config):
    """35x35 is padded to 56x56: patches 0 and 1 of each side are real, 2 is half pad, 3 is all pad."""
    image = random_image(35, 35)
    pixel_values, grid_thws = image_prep.image_to_patches(image, config)

    padded = np.pad(np.asarray(image), ((0, 21), (0, 21), (0, 0)), constant_values=0)
    x = (torch.from_numpy(padded).float() / 255 - 0.5) / 0.5
    expected = torch.stack(
        [x[14 * i : 14 * i + 14, 14 * j : 14 * j + 14].permute(2, 0, 1) for i in range(4) for j in range(4)]
    )
    assert torch.equal(grid_thws, torch.tensor([[1, 4, 4]]))
    assert torch.equal(pixel_values, expected)

    grid = pixel_values.reshape(4, 4, 3, 14, 14)
    assert (grid[3] == -1.0).all() and (grid[:, 3] == -1.0).all()
    assert (grid[2, :3, :, 7:] == -1.0).all() and (grid[:3, 2, ..., 7:] == -1.0).all()
    assert (grid[:2, :2] > -1.0).any()


def test_preprocessor_defaults(config, media_proc_cfg):
    assert config.in_patch_limit == media_proc_cfg["in_patch_limit"]
    assert config.patch_size == media_proc_cfg["patch_size"]
    assert config.max_side_patches == media_proc_cfg["patch_limit_on_one_side"]
    m = media_proc_cfg["merge_kernel_size"]
    assert config.merge_kernel_size == (m, m)
    assert media_proc_cfg["image_mean"] == [image_prep.IMAGE_MEAN] * 3
    assert media_proc_cfg["image_std"] == [image_prep.IMAGE_STD] * 3


def test_prepare_host_inputs_from_image(config):
    pixel_values, grid_thws = image_prep.image_to_patches(random_image(35, 35), config)
    pos_emb_weight = torch.randn(64, 64, config.hidden_size)  # the checkpoint's 64x64 position table
    inputs = host_prep.prepare_host_inputs(config, pos_emb_weight, pixel_values, grid_thws)
    n = inputs["logical_n"]
    assert n == 16

    raster = host_prep.from_window_order(inputs["patches"][:n], 4, 4, config.merge_kernel_size)
    assert torch.equal(raster, host_prep.patch_rows(config, pixel_values))
    all_pad = [4 * i + j for i in range(4) for j in range(4) if i == 3 or j == 3]
    assert len(all_pad) == 7
    assert (raster[all_pad, : config.patch_dim] == -1.0).all()
    assert not raster[:, config.patch_dim :].any()
    assert not inputs["patches"][n:].any()
