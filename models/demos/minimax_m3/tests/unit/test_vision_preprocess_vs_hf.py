# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the vision encoder's image code (tt/vision/preprocess.py) against the Hugging Face MiniMax-M3 image
processor, and of its input rules.

Each step must give what the processor gives, bit for bit:
  - resize_image: the processor's resized image, which is uint8 (captured where the processor rescales and
    normalizes it), and its grid;
  - patch_rows of that resized image: the processor's pixel_values and image_grid_thw;
  - preprocess, both steps: the same, from the image;
  - resized_size: the processor's resize rule (smart_resize) for 1024 sizes, rejecting the same ones.
The images are seeded noise, the hardest input for the resize, at sizes that take each branch of the resize rule
(under the 56 x 56 minimum area, exactly 672 x 672, over the 672 x 672 maximum, tall, wide, a 25 x 4000 strip, the
most stretched result) and in RGB, RGBA, grayscale and palette modes, plus the cat photo of models/sample_data and
numpy arrays of RGB and RGBA values.

Two processors: the one in transformers (MiniMaxM3VLImageProcessor; tt-metal pins transformers 5.12.1), and, only
when HF_MODEL is a checkpoint with a preprocessor_config.json, the checkpoint's own, loaded with trust_remote_code.
Without HF_MODEL the "checkpoint" cases are not generated, so CI shows no skipped test.

The other tests check check_pixels' rules (every size the resize gives passes, the bounds are reached, and the sizes
outside them fail), the errors of resize_image and patch_rows, a read-only input, that resize_image never returns a
view of its input, and that the module imports with ttnn blocked, as an encoder service's main process needs.

No device and no weights. These tests have no mesh id; the 1-card CI leg selects them by name (its -k includes
"test_vision_").

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_preprocess_vs_hf.py
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from models.demos.minimax_m3.tt.vision.image_input import MERGE_SIZE, num_tokens
from models.demos.minimax_m3.tt.vision.preprocess import (
    BLOCK,
    MAX_RESIZED_ASPECT_RATIO,
    MAX_TOKENS,
    PATCH_SIZE,
    check_pixels,
    patch_rows,
    preprocess,
    resize_image,
    resized_size,
)

REPO = Path(__file__).resolve().parents[5]
# (height, width, PIL mode) of each noise image.
IMAGES = {
    "h20w30": (20, 30, "RGB"),  # under the minimum area: scaled up to 4 patches down and 6 across
    "h672w672": (672, 672, "RGB"),  # the maximum area: not resized
    "h480w640": (480, 640, "RGB"),
    "h1080w1920": (1080, 1920, "RGB"),  # over the maximum area: scaled down to 36 patches down and 64 across
    "h1000w300": (1000, 300, "RGB"),
    "h25w4000": (25, 4000, "RGB"),  # aspect ratio 160
    "h48w9600": (48, 9600, "RGB"),  # aspect ratio 200, the most the processor takes: resized to 28 by 9492, 339:1
    "h333w517_rgba": (333, 517, "RGBA"),
    "h224w224_gray": (224, 224, "L"),
    "h100w150_palette": (100, 150, "P"),
}
PHOTOS = {"cat": "models/sample_data/huggingface_cat_image.jpg"}  # 640x480 RGB JPEG, resized to 644x476: 391 tokens
# Image sides for the resize rule: around the patch and resize-factor multiples, the limits and large photos.
SIDES = [1, 2, 13, 14, 27, 28, 29, 41, 42, 43, 55, 56, 57, 84, 100, 224, 333, 448, 480, 500, 640, 671, 672, 673, 700]
SIDES += [896, 1000, 1080, 1344, 1920, 2016, 4000]


def _processor_kinds():
    checkpoint = os.getenv("HF_MODEL")
    has_processor = bool(checkpoint) and os.path.isfile(os.path.join(checkpoint, "preprocessor_config.json"))
    return ["transformers"] + (["checkpoint"] if has_processor else [])


@pytest.fixture(scope="module", params=_processor_kinds())
def hf_processor(request):
    """transformers' MiniMax-M3 image processor, or the checkpoint's own (HF_MODEL)."""
    if request.param == "transformers":
        from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import MiniMaxM3VLImageProcessor

        return MiniMaxM3VLImageProcessor()
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(os.environ["HF_MODEL"], trust_remote_code=True)


def _noise_image(height: int, width: int, mode: str) -> Image.Image:
    rng = np.random.default_rng(10007 * height + width)
    shape = {"L": (height, width), "RGBA": (height, width, 4)}.get(mode, (height, width, 3))
    image = Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8))
    return image if image.mode == mode else image.convert(mode)


def _photo(name: str) -> Image.Image:
    with Image.open(REPO / PHOTOS[name]) as image:
        image.load()
        return image.copy()


def _run_processor(processor, image, monkeypatch):
    """The processor's pixel_values and image_grid_thw for one image, and the resized image it normalized, as
    (height, width, 3), or None for a processor without transformers' rescale_and_normalize step."""
    resized = []
    with monkeypatch.context() as patch:
        if hasattr(processor, "rescale_and_normalize"):
            normalize = processor.rescale_and_normalize

            def record(images, *args, **kwargs):
                resized.append(images.clone())
                return normalize(images, *args, **kwargs)

            patch.setattr(processor, "rescale_and_normalize", record)
        out = processor(images=[image], return_tensors="pt")
    if not resized:
        return out["pixel_values"], out["image_grid_thw"], None
    (batch,) = resized
    assert batch.shape[0] == 1, f"the processor normalized {batch.shape[0]} images at once"
    return out["pixel_values"], out["image_grid_thw"], batch[0].permute(1, 2, 0).contiguous()


def _check_equal(name: str, ours: torch.Tensor, hf: torch.Tensor):
    ours_type, hf_type = (ours.dtype, tuple(ours.shape)), (hf.dtype, tuple(hf.shape))
    assert ours_type == hf_type, f"{name}: {ours_type} != HF {hf_type}"
    assert torch.equal(ours, hf), f"{name}: not bit-identical to HF, max abs diff {(ours - hf).abs().max().item():.3e}"


def _check_steps(image, processor, monkeypatch):
    """resize_image, patch_rows on the processor's resized image, and preprocess, each equal to the processor."""
    pixel_values, grid_thw, hf_resized = _run_processor(processor, image, monkeypatch)
    resized = resize_image(image)
    assert resized.pixels.dtype == np.uint8 and resized.pixels.flags.c_contiguous
    assert [[1, 2 * resized.grid_h, 2 * resized.grid_w]] == grid_thw.tolist()
    if hf_resized is not None:
        _check_equal("resized image", torch.from_numpy(resized.pixels), hf_resized)
        rows, rows_grid = patch_rows(hf_resized.numpy())
        _check_equal("pixel_values from HF's resized image", rows, pixel_values)
        _check_equal("grid_thw from HF's resized image", rows_grid, grid_thw)
    rows, rows_grid = preprocess(image)
    _check_equal("pixel_values", rows, pixel_values)
    _check_equal("grid_thw", rows_grid, grid_thw)
    return pixel_values


@pytest.mark.parametrize("image_id", list(IMAGES) + list(PHOTOS))
def test_vision_preprocess_vs_hf(hf_processor, image_id, monkeypatch):
    """Each step equals the processor bit for bit: resize_image its resized uint8 image, patch_rows of that image and
    preprocess its pixel_values and image_grid_thw; resized_size and num_tokens give its grid and token count from the
    image size alone."""
    if image_id in IMAGES:
        height, width, mode = IMAGES[image_id]
        image = _noise_image(height, width, mode)
    else:
        image = _photo(image_id)
        width, height = image.size
    _check_steps(image, hf_processor, monkeypatch)
    grid = [[1, *(side // PATCH_SIZE for side in resized_size(height, width))]]
    assert num_tokens(grid) == hf_processor.get_number_of_image_patches(height, width) // MERGE_SIZE**2


def test_vision_preprocess_array_vs_hf(hf_processor, monkeypatch):
    """A uint8 RGB array (height, width, 3) gives what the processor gives for it and for the same PIL image; an RGBA
    array (height, width, 4) gives what the processor gives for the PIL RGBA image, whose alpha channel it drops."""
    photo = np.array(_photo("cat"))
    assert photo.shape == (480, 640, 3)
    from_array = _check_steps(photo, hf_processor, monkeypatch)
    _check_equal("pixel_values of the array against the PIL image", from_array, preprocess(Image.fromarray(photo))[0])
    rgba = np.concatenate([photo, np.full((480, 640, 1), 7, dtype=np.uint8)], axis=2)  # nearly transparent
    rows, _ = preprocess(rgba)
    hf_rows = _run_processor(hf_processor, Image.fromarray(rgba), monkeypatch)[0]
    _check_equal("pixel_values of an RGBA array", rows, hf_rows)
    _check_equal("pixel_values of an RGBA array against its RGB values", rows, from_array)


def test_vision_resized_size_vs_hf(hf_processor):
    """resized_size follows the processor's resize rule for 1024 sizes and rejects the same ones (aspect ratio over
    200); every size it gives passes check_pixels, at most 576 tokens. The rejections are checked without
    expect_error, which would log two lines for each of about a hundred sizes."""
    smart_resize = inspect.getmodule(type(hf_processor)).smart_resize
    factor, max_pixels = hf_processor.patch_size * hf_processor.merge_size, hf_processor.max_pixels
    for height in SIDES:
        for width in SIDES:
            try:
                expected = smart_resize(height, width, factor=factor, max_pixels=max_pixels)
            except ValueError:
                expected = None
            try:
                ours = resized_size(height, width)
            except ValueError as error:
                ours = str(error)
            if expected is None:
                assert "aspect ratio must be at most 200" in str(ours), (height, width, ours)
                continue
            assert ours == expected, (height, width)
            grid_h, grid_w = check_pixels(np.zeros((*expected, 3), dtype=np.uint8))
            assert (grid_h * BLOCK, grid_w * BLOCK) == expected and grid_h * grid_w <= MAX_TOKENS, (height, width)


def test_vision_resized_aspect_ratio_bound(hf_processor):
    """MAX_RESIZED_ASPECT_RATIO is the most the processor's resize stretches an image: over every image with an aspect
    ratio of 190 to 200 and a short side of up to 60 pixels, where the rounding stretches most (its module docstring
    has the bound), the most stretched resized image is 339 tokens long and 1 token wide, as for 9600x48 pixels."""
    smart_resize = inspect.getmodule(type(hf_processor)).smart_resize
    ratios = []
    for short in range(1, 61):
        for long in range(190 * short, 200 * short + 1):
            height, width = smart_resize(short, long)
            ratios.append(width / height)
            assert smart_resize(long, short) == (width, height)
    assert max(ratios) == MAX_RESIZED_ASPECT_RATIO
    assert smart_resize(48, 9600) == (BLOCK, MAX_RESIZED_ASPECT_RATIO * BLOCK)


# id: (height, width) of resized images at the bounds of check_pixels, which must pass.
GOOD_PIXELS = {
    "fewest_tokens": (28, 112),  # 1 token down and 4 across, the area of 56 x 56 pixels
    "most_tokens": (672, 672),  # 24 tokens down and 24 across
    "most_stretched_wide": (28, 9492),  # 1 token down and 339 across
    "most_stretched_tall": (9492, 28),
}


@pytest.mark.parametrize("case", GOOD_PIXELS)
def test_vision_check_pixels_bounds(case):
    """check_pixels takes the resized images at its bounds and gives their token grid."""
    height, width = GOOD_PIXELS[case]
    assert check_pixels(np.zeros((height, width, 3), dtype=np.uint8)) == (height // BLOCK, width // BLOCK)


# id: (pixels, error, message). Every case is the resize of nothing: no image resizes to it.
BAD_PIXELS = {
    "list": ([[[0, 0, 0]]], TypeError, "numpy array"),
    "torch": (torch.zeros(56, 56, 3, dtype=torch.uint8), TypeError, "numpy array"),
    "float32": (np.zeros((56, 56, 3), dtype=np.float32), TypeError, "uint8"),
    "int64": (np.zeros((56, 56, 3), dtype=np.int64), TypeError, "uint8"),
    "gray": (np.zeros((56, 56), dtype=np.uint8), ValueError, r"\(height, width, 3\)"),
    "rgba": (np.zeros((56, 56, 4), dtype=np.uint8), ValueError, r"\(height, width, 3\)"),
    "channels_first": (np.zeros((3, 56, 56), dtype=np.uint8), ValueError, r"\(height, width, 3\)"),
    "not_whole_blocks": (np.zeros((56, 60, 3), dtype=np.uint8), ValueError, "got height 56 and width 60"),
    "half_block": (np.zeros((14, 112, 3), dtype=np.uint8), ValueError, "multiples of 28"),
    "empty": (np.zeros((0, 56, 3), dtype=np.uint8), ValueError, "multiples of 28"),
    "one_token": (np.zeros((28, 28, 3), dtype=np.uint8), ValueError, r"4 to 576 vision tokens .* got 1 \(1 down"),
    "three_tokens": (np.zeros((28, 84, 3), dtype=np.uint8), ValueError, r"got 3 \(1 down, 3 across\)"),
    "too_many_tokens": (np.zeros((700, 700, 3), dtype=np.uint8), ValueError, r"got 625 \(25 down, 25 across\)"),
    "too_stretched": (np.zeros((28, 9520, 3), dtype=np.uint8), ValueError, "aspect ratio must be at most 339"),
    "too_stretched_tall": (np.zeros((9520, 28, 3), dtype=np.uint8), ValueError, "aspect ratio must be at most 339"),
}


@pytest.mark.parametrize("case", BAD_PIXELS)
def test_vision_check_pixels_rejects(case, expect_error):
    """check_pixels, and patch_rows through it, raise TypeError for pixels that are not a uint8 numpy array and
    ValueError for one that no image resizes to: not (height, width, 3), sides that are not whole 28-pixel blocks, fewer
    than 4 or more than 576 tokens, or more stretched than 339:1."""
    pixels, error, message = BAD_PIXELS[case]
    for check in (check_pixels, patch_rows):
        with expect_error(error, message):
            check(pixels)


# id: (image, error, message)
BAD_IMAGES = {
    "torch": (torch.zeros(3, 56, 56, dtype=torch.uint8), TypeError, "PIL image or a uint8 numpy array"),
    "float_array": (np.zeros((56, 56, 3), dtype=np.float32), TypeError, "must be uint8"),
    "gray_array": (np.zeros((56, 56), dtype=np.uint8), ValueError, "RGB or RGBA"),
    "two_channels": (np.zeros((56, 56, 2), dtype=np.uint8), ValueError, "RGB or RGBA"),
    "empty_array": (np.zeros((0, 56, 3), dtype=np.uint8), ValueError, "must be positive"),
    "aspect_ratio": (np.zeros((10, 2010, 3), dtype=np.uint8), ValueError, "aspect ratio must be at most 200"),
}


@pytest.mark.parametrize("case", BAD_IMAGES)
def test_vision_resize_image_rejects(case, expect_error):
    """resize_image and preprocess raise TypeError for an input that is not a PIL image or a uint8 array and ValueError
    for an array that is not RGB or RGBA, an empty image and an aspect ratio over 200."""
    image, error, message = BAD_IMAGES[case]
    for run in (resize_image, preprocess):
        with expect_error(error, message):
            run(image)


def test_vision_resize_image_returns_new_memory():
    """resize_image never returns a view of the caller's array, even for an image already at its resized size, and
    patch_rows takes a read-only array, as one unpickled from shared memory may be, without a warning or a change."""
    image = np.random.default_rng(0).integers(0, 256, (672, 672, 3), dtype=np.uint8)
    resized = resize_image(image)
    assert np.array_equal(resized.pixels, image) and not np.shares_memory(resized.pixels, image)
    assert (resized.grid_h, resized.grid_w) == (24, 24)

    read_only = image.copy()
    read_only.setflags(write=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rows, grid = patch_rows(read_only)
    expected_rows, expected_grid = patch_rows(image)
    assert torch.equal(rows, expected_rows) and torch.equal(grid, expected_grid)
    assert np.array_equal(read_only, image) and not read_only.flags.writeable


def test_vision_preprocess_imports_without_ttnn():
    """The module imports in a fresh interpreter in which importing ttnn fails, so no package on its path imports ttnn
    (as a package __init__.py could), and it does not import transformers either."""
    code = "\n".join(
        [
            "import sys",
            "sys.modules['ttnn'] = None  # import ttnn now raises ImportError",
            f"sys.path.insert(0, {str(REPO)!r})",
            "from models.demos.minimax_m3.tt.vision import image_input, preprocess",
            "loaded = sorted(m for m in sys.modules if m.split('.')[0] in ('ttnn', 'transformers'))",
            "assert loaded == ['ttnn'] and sys.modules['ttnn'] is None, loaded",
            f"assert all(m.__file__.startswith({str(REPO)!r}) for m in (image_input, preprocess))",
            "print(preprocess.MAX_TOKENS)",
        ]
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0 and out.stdout.split() == [str(MAX_TOKENS)], out.stderr[-2000:]
