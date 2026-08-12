# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only gate for MiniMax-H3 FL2VA keyframe conditioning. Four silent contracts: ImageNet
pixel statistics (not [-1, 1]), a sampled posterior under a request-independent seed, an fp16
round trip on the sampled latent, and conditioning noise as the request generator's first draw."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import transformers
from PIL import Image

from ....pipelines.minimax_h3 import conditioning as c
from ....pipelines.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    build_packed_sequence,
    patchify_video_latents,
    prepare_keyframe_image,
    resolve_canvas_size,
)
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline, draw_request_latents
from ....pipelines.minimax_h3.scheduler import MiniMaxH3Scheduler

LATENT_CHANNELS = 24
LATENT_HEIGHT, LATENT_WIDTH = 34, 60
PATCH = (1, 2, 2)
CONDITION_SHAPES = ((1, LATENT_HEIGHT, LATENT_WIDTH),) * 2

LATENTS_MEAN = tuple(0.1 * index for index in range(LATENT_CHANNELS))
LATENTS_STD = tuple(1.0 + 0.05 * index for index in range(LATENT_CHANNELS))


def _latents(seed=5, frames=1):
    return torch.randn(
        1, LATENT_CHANNELS, frames, LATENT_HEIGHT, LATENT_WIDTH, generator=torch.Generator().manual_seed(seed)
    )


def _image(height=544, width=960, seed=3):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def test_pixels_use_imagenet_statistics():
    assert c.MINIMAX_H3_PIXEL_MEAN == (0.485, 0.456, 0.406)
    assert c.MINIMAX_H3_PIXEL_STD == (0.229, 0.224, 0.225)

    image = _image()
    pixels = c.normalize_keyframe_pixels(image)
    assert pixels.shape == (1, 3, 1, image.size[1], image.size[0])
    assert pixels.dtype == torch.float32

    mean = torch.tensor(c.MINIMAX_H3_PIXEL_MEAN).view(1, -1, 1, 1, 1)
    std = torch.tensor(c.MINIMAX_H3_PIXEL_STD).view(1, -1, 1, 1, 1)
    raw = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None, :, None].float().div(255.0)
    assert torch.equal(pixels, (raw - mean) / std)
    assert pixels.min() < -1.0


def test_posterior_seed_is_independent_of_request_seed():
    assert c.MINIMAX_H3_KEYFRAME_ENCODE_SEED == 42
    moments = torch.randn(1, 2 * LATENT_CHANNELS, 1, 8, 8, generator=torch.Generator().manual_seed(1))

    torch.manual_seed(999)
    first = c.sample_posterior(moments)
    torch.manual_seed(12345)
    second = c.sample_posterior(moments)
    assert torch.equal(first, second)

    assert not torch.equal(first, c.sample_posterior(moments, seed=43))

    # a sample, not the mean -- the mean would also be request-independent
    mean = moments.chunk(2, dim=1)[0]
    assert first.shape == mean.shape
    assert not torch.equal(first, mean)


def test_float16_round_trip_is_load_bearing():
    """The fp16 round trip is load-bearing; asserted to change the rows so it cannot be optimized away."""
    latents = _latents()
    rows = c.keyframe_condition_rows(latents, LATENTS_MEAN, LATENTS_STD, PATCH)

    expected_rows = (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2)
    assert rows.shape == (expected_rows, LATENT_CHANNELS * 4)
    assert rows.dtype == torch.float32

    mean = torch.tensor(LATENTS_MEAN).view(1, LATENT_CHANNELS, 1, 1, 1)
    std = torch.tensor(LATENTS_STD).view(1, LATENT_CHANNELS, 1, 1, 1)
    exact = patchify_video_latents((latents - mean) / std, PATCH)
    rounded = patchify_video_latents((latents.to(torch.float16).float() - mean) / std, PATCH)

    assert torch.equal(rows, rounded)
    assert not torch.equal(rows, exact)
    assert (rows - exact).abs().max() > 1e-5


def test_encode_keyframes_is_one_encode_per_image():
    seen = []

    def encode_clip(pixels):
        seen.append(tuple(pixels.shape))
        batch, _, frames, height, width = pixels.shape
        return torch.zeros(batch, 2 * LATENT_CHANNELS, frames, height // 16, width // 16)

    images = [_image(seed=1), _image(seed=2)]
    rows = c.encode_keyframes(images, encode_clip, LATENTS_MEAN, LATENTS_STD, PATCH)

    assert len(seen) == len(images)
    assert all(shape[2] == 1 for shape in seen)
    rows_per_frame = (544 // 16 // 2) * (960 // 16 // 2)
    assert rows.shape == (len(images) * rows_per_frame, LATENT_CHANNELS * 4)


def test_condition_noise_is_deterministic_per_seed():
    seed = 42
    first = c.keyframe_condition_noise(CONDITION_SHAPES, LATENT_CHANNELS, PATCH, torch.Generator().manual_seed(seed))
    second = c.keyframe_condition_noise(CONDITION_SHAPES, LATENT_CHANNELS, PATCH, torch.Generator().manual_seed(seed))
    assert torch.equal(first, second)

    rows_per_frame = (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2)
    assert first.shape == (len(CONDITION_SHAPES) * rows_per_frame, LATENT_CHANNELS * 4)


def test_condition_noise_draws_per_condition_in_order():
    """One concatenated draw would consume the same values, assign them differently, and leave the same generator state."""
    generator = torch.Generator().manual_seed(0)
    both = c.keyframe_condition_noise(CONDITION_SHAPES, LATENT_CHANNELS, PATCH, generator)

    per_condition = torch.cat(
        [
            c.keyframe_condition_noise((shape,), LATENT_CHANNELS, PATCH, torch.Generator().manual_seed(0))
            for shape in CONDITION_SHAPES[:1]
        ]
    )
    assert torch.equal(both[: per_condition.shape[0]], per_condition)


def test_condition_noise_is_the_first_draw_of_a_request():
    ordered = torch.Generator().manual_seed(0)
    c.keyframe_condition_noise(CONDITION_SHAPES, LATENT_CHANNELS, PATCH, ordered)
    video_after_conditioning = torch.randn(3, 64, generator=ordered)

    skipped = torch.Generator().manual_seed(0)
    video_without_conditioning = torch.randn(3, 64, generator=skipped)

    assert not torch.equal(video_after_conditioning, video_without_conditioning)


def test_noise_augmentation_has_no_second_implementation():
    """A local t*x0 + (1-t)*noise drifted 2.4e-7 against the scheduler (Python-double 1-t)."""
    assert not hasattr(c, "noise_augment")
    assert MINIMAX_H3_KEYFRAME_NOISE_AUG == 0.999

    rows = torch.randn(20, 96, generator=torch.Generator().manual_seed(1))
    noise = torch.randn(20, 96, generator=torch.Generator().manual_seed(2))
    out = MiniMaxH3Scheduler(12.0).scale_noise(rows, MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)

    timestep = torch.tensor(MINIMAX_H3_KEYFRAME_NOISE_AUG, dtype=torch.float32)
    assert torch.equal(out, timestep * rows + (1.0 - timestep) * noise)
    drifted = MINIMAX_H3_KEYFRAME_NOISE_AUG * rows + (1.0 - MINIMAX_H3_KEYFRAME_NOISE_AUG) * noise
    assert not torch.equal(out, drifted)


PROD_LATENT_FRAMES = 37
PROD_LATENT_H, PROD_LATENT_W = 48, 84
PROD_AUDIO_LATENTS = 207
PROD_AUDIO_CHANNELS = 32
PROD_ROWS_PER_FRAME = (PROD_LATENT_H // 2) * (PROD_LATENT_W // 2)  # 1008


def _draw(num_keyframes: int, seed: int = 0):
    return draw_request_latents(
        torch.Generator().manual_seed(seed),
        condition_latent_shapes=((1, PROD_LATENT_H, PROD_LATENT_W),) * num_keyframes,
        latent_channels=LATENT_CHANNELS,
        num_latent_frames=PROD_LATENT_FRAMES,
        latent_height=PROD_LATENT_H,
        latent_width=PROD_LATENT_W,
        num_audio_latents=PROD_AUDIO_LATENTS,
        audio_latent_channels=PROD_AUDIO_CHANNELS,
        patch_size=PATCH,
    )


def test_t2va_draws_are_unchanged_by_the_keyframe_argument():
    """The t2va no-regression proof: nothing may be drawn before the video draw."""
    condition_noise, video_rows, audio_rows = _draw(0)
    assert condition_noise is None, "t2va must draw no conditioning noise at all"

    generator = torch.Generator().manual_seed(0)
    expected_video = patchify_video_latents(
        torch.randn(
            (1, LATENT_CHANNELS, PROD_LATENT_FRAMES, PROD_LATENT_H, PROD_LATENT_W),
            generator=generator,
            dtype=torch.float32,
        ),
        PATCH,
    )
    expected_audio = torch.randn(
        (PROD_AUDIO_LATENTS * 2, PROD_AUDIO_CHANNELS), generator=generator, dtype=torch.float32
    )

    assert torch.equal(video_rows, expected_video), "t2va video draw moved"
    assert torch.equal(audio_rows, expected_audio), "t2va audio draw moved"


@pytest.mark.parametrize("num_keyframes", [1, 2], ids=["first", "first_and_last"])
def test_conditioning_noise_is_drawn_before_video_and_audio(num_keyframes):
    condition_noise, video_rows, audio_rows = _draw(num_keyframes)
    _, t2va_video, t2va_audio = _draw(0)

    assert condition_noise is not None
    assert condition_noise.shape == (num_keyframes * PROD_ROWS_PER_FRAME, LATENT_CHANNELS * PATCH[1] * PATCH[2])
    expected_noise = c.keyframe_condition_noise(
        ((1, PROD_LATENT_H, PROD_LATENT_W),) * num_keyframes,
        LATENT_CHANNELS,
        PATCH,
        torch.Generator().manual_seed(0),
    )
    assert torch.equal(condition_noise, expected_noise), "conditioning noise is not the first draw"

    # fl2va at a given seed does NOT reproduce t2va at that seed -- expected, not a regression
    assert not torch.equal(video_rows, t2va_video), "the conditioning draw did not advance the stream"
    assert not torch.equal(audio_rows, t2va_audio)


def test_keyframe_rows_per_anchor_match_the_layout():
    text_tags = torch.ones(39, dtype=torch.long)
    for anchors in (("first",), ("last",), ("first", "last")):
        layout = build_packed_sequence(
            text_tags, PROD_LATENT_FRAMES, PROD_LATENT_H, PROD_LATENT_W, PROD_AUDIO_LATENTS, PATCH, anchors
        )
        assert layout.num_condition_video_rows == len(anchors) * PROD_ROWS_PER_FRAME
        assert layout.num_condition_audio_rows == 0, "fl2va conditions video only; audio conditioning is ref2va"
        expected = torch.arange(39, 39 + layout.num_condition_video_rows)
        assert torch.equal(layout.video_indices[: layout.num_condition_video_rows], expected)

        condition_noise, _, _ = _draw(len(anchors))
        assert condition_noise.shape[0] == layout.num_condition_video_rows


PROMPT = (
    "A red fox trots across a snowy field at dawn, its breath visible in the cold air. "
    "The low sun throws long blue shadows behind it, and loose snow lifts from each footfall."
)


def _weights_dir() -> str:
    base = os.environ.get("MINIMAX_H3_MODEL_PATH")
    if not base:
        pytest.skip("set MINIMAX_H3_MODEL_PATH to a MiniMax-H3 diffusers snapshot")
    return os.path.join(base, "text_encoder")


def _snapshot_root() -> str:
    return os.path.dirname(_weights_dir())


def _presentation(prompt, keyframes):
    """The real method called unbound: a reimplementation here would gate a second copy of the presentation."""
    root = _snapshot_root()
    stub = SimpleNamespace(
        tokenizer=transformers.AutoTokenizer.from_pretrained(root, subfolder="tokenizer"),
        image_processor=transformers.AutoImageProcessor.from_pretrained(root, subfolder="text_encoder"),
    )
    return MiniMaxH3Pipeline._build_presentation(stub, prompt, keyframes), stub


@pytest.mark.parametrize("num_keyframes", [1, 2], ids=["first", "first_and_last"])
def test_fl2va_presentation_matches_the_reference(num_keyframes):
    """Token ids, H3 tags (whole vision block video) and Qwen's mm_token_type_ids (pads only) each fail silently alone."""
    source = Image.fromarray(
        (torch.rand(720, 1280, 3, generator=torch.Generator().manual_seed(0)) * 255).to(torch.uint8).numpy()
    )
    height, width = resolve_canvas_size(*source.size)
    assert (height, width) == (768, 1344), f"expected the production canvas, got {width}x{height}"
    keyframes = [prepare_keyframe_image(source, height, width, stretch=(index == 0)) for index in range(num_keyframes)]

    (input_ids, tags, type_ids, pixel_values, grid_thw), stub = _presentation(PROMPT, keyframes)
    tokenizer, processor = stub.tokenizer, stub.image_processor
    merge = processor.merge_size**2

    # the reference's builder: `encoders.py::MiniMaxH3TextEncoderStep.encode_prompt`
    ref_ids, ref_tags = [], []
    for index in range(num_keyframes):
        num_image_tokens = int(grid_thw[index].prod()) // merge
        label = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
        block = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        ref_ids += label + block
        ref_tags += [MINIMAX_H3_TEXT_TAG] * len(label) + [MINIMAX_H3_VIDEO_TAG] * len(block)
    prompt_ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
    ref_ids += prompt_ids
    ref_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)

    assert input_ids.shape == (1, len(ref_ids)), f"{tuple(input_ids.shape)} vs {(1, len(ref_ids))}"
    assert input_ids[0].tolist() == ref_ids, "token ids differ from the reference presentation"
    assert tags.tolist() == ref_tags, "H3 row tags differ from the reference presentation"

    full_processor = transformers.AutoProcessor.from_pretrained(_snapshot_root(), subfolder="processor")
    expected_type_ids = torch.tensor(full_processor.create_mm_token_type_ids([input_ids[0].tolist()]))
    assert torch.equal(type_ids, expected_type_ids), "mm_token_type_ids differ from the processor's own"
    assert int(type_ids.sum()) == num_keyframes * 1008

    assert int((tags == MINIMAX_H3_VIDEO_TAG).sum()) == num_keyframes * (1008 + 2)
    assert grid_thw.shape[0] == num_keyframes
    assert pixel_values.shape[0] == num_keyframes * 1008 * merge
