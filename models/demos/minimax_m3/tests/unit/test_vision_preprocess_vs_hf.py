# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the vision encoder's input against the Hugging Face MiniMax-M3 image processor: the media module's
prepare (vision/media.py), which resizes an image, and patch_rows (tt/vision/preprocess.py), which turns the resized
image into the encoder's patch rows; and of check_size's and check_pixels' rules.

Each step must give what the processor gives, bit for bit:
  - prepare: the processor's resized image, which is uint8 (captured where the processor rescales and normalizes it),
    and its grid;
  - patch_rows, of that resized image and of prepare's: the processor's pixel_values and image_grid_thw;
  - smart_resize, prepare's resize rule: the processor's for 1024 sizes and for the most stretched images, rejecting
    the same ones.
The images are seeded noise, the hardest input for the resize, at sizes that take each branch of the resize rule
(under the 56 x 56 minimum area, exactly 672 x 672, over the 672 x 672 maximum, tall, wide, a 25 x 4000 strip, the
most stretched result) and in RGB, RGBA, grayscale and palette modes, plus the cat photo of models/sample_data. The
processor takes the PIL image, which it converts to RGB, and prepare the uint8 RGB array of it that an encoder service
passes, converted the same way (PIL's convert).

Two processors: the one in transformers (MiniMaxM3VLImageProcessor; tt-metal pins transformers 5.12.1), and, only
when HF_MODEL is a checkpoint with a preprocessor_config.json, the checkpoint's own, loaded with trust_remote_code.
Without HF_MODEL the "checkpoint" cases are not generated, so CI shows no skipped test.

The other tests check check_size's and check_pixels' rules (every size the resize gives passes, the bounds are
reached, and the sizes outside them fail), that prepare never returns a view of its input, patch_rows on a read-only
input, and that tt/vision/preprocess.py and image_input.py, which the media module imports, import with ttnn blocked.

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
    FACTOR,
    MAX_RESIZED_ASPECT_RATIO,
    MAX_TOKENS,
    PATCH_SIZE,
    check_pixels,
    check_size,
    patch_rows,
)
from models.demos.minimax_m3.vision.media import MiniMaxM3Media, smart_resize

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
# prepare reads none of the config's values.
MEDIA = MiniMaxM3Media(
    {"image_token_index": 0, "text_config": {"hidden_size": 1}, "vision_config": {"patch_size": PATCH_SIZE}}
)


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


def _check_steps(image: Image.Image, processor, monkeypatch):
    """prepare's resized image and grid, patch_rows of the processor's resized image, and patch_rows of prepare's,
    each equal to the processor."""
    pixel_values, grid_thw, hf_resized = _run_processor(processor, image, monkeypatch)
    item = MEDIA.prepare("image", np.array(image.convert("RGB")))
    assert item.pixels.dtype == np.uint8 and item.pixels.flags.c_contiguous
    assert [[1, 2 * item.grid_h, 2 * item.grid_w]] == grid_thw.tolist()
    if hf_resized is not None:
        _check_equal("resized image", torch.from_numpy(item.pixels), hf_resized)
        rows, rows_grid = patch_rows(hf_resized.numpy())
        _check_equal("pixel_values from HF's resized image", rows, pixel_values)
        _check_equal("grid_thw from HF's resized image", rows_grid, grid_thw)
    rows, rows_grid = patch_rows(item.pixels)
    _check_equal("pixel_values", rows, pixel_values)
    _check_equal("grid_thw", rows_grid, grid_thw)


@pytest.mark.parametrize("image_id", list(IMAGES) + list(PHOTOS))
def test_vision_preprocess_vs_hf(hf_processor, image_id, monkeypatch):
    """Each step equals the processor bit for bit: prepare its resized uint8 image, patch_rows its pixel_values and
    image_grid_thw; smart_resize and num_tokens give its grid and token count from the image size alone."""
    if image_id in IMAGES:
        height, width, mode = IMAGES[image_id]
        image = _noise_image(height, width, mode)
    else:
        image = _photo(image_id)
        width, height = image.size
    _check_steps(image, hf_processor, monkeypatch)
    grid = [[1, *(side // PATCH_SIZE for side in smart_resize(height, width))]]
    assert num_tokens(grid) == hf_processor.get_number_of_image_patches(height, width) // MERGE_SIZE**2


def test_vision_smart_resize_vs_hf(hf_processor):
    """smart_resize, prepare's resize rule, gives the processor's size for 1024 sizes and rejects the same ones (aspect
    ratio over 200); every size it gives passes check_pixels, at most 576 tokens. The rejections are checked without
    expect_error, which would log two lines for each of about a hundred sizes."""
    hf_smart_resize = inspect.getmodule(type(hf_processor)).smart_resize
    factor, max_pixels = hf_processor.patch_size * hf_processor.merge_size, hf_processor.max_pixels
    for height in SIDES:
        for width in SIDES:
            try:
                expected = hf_smart_resize(height, width, factor=factor, max_pixels=max_pixels)
            except ValueError:
                expected = None
            try:
                got = smart_resize(height, width)
            except ValueError as error:
                got = str(error)
            if expected is None:
                assert "aspect ratio" in str(got), (height, width, got)
                continue
            assert got == expected, (height, width)
            grid_h, grid_w = check_pixels(np.zeros((*expected, 3), dtype=np.uint8))
            assert (grid_h * FACTOR, grid_w * FACTOR) == expected and grid_h * grid_w <= MAX_TOKENS, (height, width)


def test_vision_resized_aspect_ratio_bound(hf_processor):
    """MAX_RESIZED_ASPECT_RATIO is the most the resize stretches an image: over every image with an aspect ratio of
    190 to 200 and a short side of up to 60 pixels, where the rounding stretches most (preprocess.py has the bound),
    smart_resize gives the processor's size, and the most stretched resized image is 339 tokens long and 1 token wide,
    as for 9600x48 pixels."""
    hf_smart_resize = inspect.getmodule(type(hf_processor)).smart_resize
    ratios = []
    for short in range(1, 61):
        for long in range(190 * short, 200 * short + 1):
            height, width = hf_smart_resize(short, long)
            assert smart_resize(short, long) == (height, width), (short, long)
            assert hf_smart_resize(long, short) == (width, height)
            ratios.append(width / height)
    assert max(ratios) == MAX_RESIZED_ASPECT_RATIO
    assert smart_resize(48, 9600) == (FACTOR, MAX_RESIZED_ASPECT_RATIO * FACTOR)


# id: (height, width) of resized images at the bounds of check_size and check_pixels, which must pass.
GOOD_PIXELS = {
    "fewest_tokens": (28, 112),  # 1 token down and 4 across, the area of 56 x 56 pixels
    "most_tokens": (672, 672),  # 24 tokens down and 24 across
    "most_stretched_wide": (28, 9492),  # 1 token down and 339 across
    "most_stretched_tall": (9492, 28),
}


@pytest.mark.parametrize("case", GOOD_PIXELS)
def test_vision_check_pixels_bounds(case):
    """check_size and check_pixels take the resized images at their bounds and give their token grid."""
    height, width = GOOD_PIXELS[case]
    grid = (height // FACTOR, width // FACTOR)
    assert check_size(height, width) == grid
    assert check_pixels(np.zeros((height, width, 3), dtype=np.uint8)) == grid


# id: ((height, width), message). No image resizes to these sizes.
BAD_SIZES = {
    "not_whole_blocks": ((56, 60), "got height 56 and width 60"),
    "half_block": ((14, 112), "multiples of 28"),
    "empty": ((0, 56), "multiples of 28"),
    "one_token": ((28, 28), r"4 to 576 vision tokens .* got 1 \(1 down"),
    "three_tokens": ((28, 84), r"got 3 \(1 down, 3 across\)"),
    "too_many_tokens": ((700, 700), r"got 625 \(25 down, 25 across\)"),
    "too_stretched": ((28, 9520), "aspect ratio must be at most 339"),
    "too_stretched_tall": ((9520, 28), "aspect ratio must be at most 339"),
}
# id: (pixels, error, message). Every case is the resize of nothing: no image resizes to it.
BAD_PIXELS = {
    "list": ([[[0, 0, 0]]], TypeError, "numpy array"),
    "torch": (torch.zeros(56, 56, 3, dtype=torch.uint8), TypeError, "numpy array"),
    "float32": (np.zeros((56, 56, 3), dtype=np.float32), TypeError, "uint8"),
    "int64": (np.zeros((56, 56, 3), dtype=np.int64), TypeError, "uint8"),
    "gray": (np.zeros((56, 56), dtype=np.uint8), ValueError, r"\(height, width, 3\)"),
    "rgba": (np.zeros((56, 56, 4), dtype=np.uint8), ValueError, r"\(height, width, 3\)"),
    "channels_first": (np.zeros((3, 56, 56), dtype=np.uint8), ValueError, r"\(height, width, 3\)"),
    # RGB images of the sizes in BAD_SIZES
    **{
        case: (np.zeros((*size, 3), dtype=np.uint8), ValueError, message) for case, (size, message) in BAD_SIZES.items()
    },
}


@pytest.mark.parametrize("case", BAD_SIZES)
def test_vision_check_size_rejects(case, expect_error):
    """check_size raises ValueError for a size that no image resizes to: sides that are not whole 28-pixel blocks,
    fewer than 4 or more than 576 tokens, or more stretched than 339:1."""
    (height, width), message = BAD_SIZES[case]
    with expect_error(ValueError, message):
        check_size(height, width)


@pytest.mark.parametrize("case", BAD_PIXELS)
def test_vision_check_pixels_rejects(case, expect_error):
    """check_pixels, and patch_rows through it, raise TypeError for pixels that are not a uint8 numpy array and
    ValueError for one that no image resizes to: not (height, width, 3), sides that are not whole 28-pixel blocks, fewer
    than 4 or more than 576 tokens, or more stretched than 339:1."""
    pixels, error, message = BAD_PIXELS[case]
    for check in (check_pixels, patch_rows):
        with expect_error(error, message):
            check(pixels)


def test_vision_prepare_returns_new_memory():
    """prepare never returns a view of the caller's array, even for an image already at its resized size, so a
    service may reuse the array's buffer once prepare returns."""
    rgb = np.random.default_rng(0).integers(0, 256, (672, 672, 3), dtype=np.uint8)
    item = MEDIA.prepare("image", rgb)
    assert np.array_equal(item.pixels, rgb) and not np.shares_memory(item.pixels, rgb)
    assert (item.grid_h, item.grid_w) == (24, 24)


def test_vision_patch_rows_read_only():
    """patch_rows takes a read-only array, as one unpickled from shared memory may be, without a warning or a
    change."""
    image = np.random.default_rng(0).integers(0, 256, (672, 672, 3), dtype=np.uint8)
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
