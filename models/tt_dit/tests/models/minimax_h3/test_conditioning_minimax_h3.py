# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only gate for MiniMax-H3 FL2VA keyframe conditioning.

Four things here reproduce nothing if dropped, and none of them fails loudly:
ImageNet pixel statistics rather than ``[-1, 1]``, a *sampled* posterior under a
seed independent of the request, a float16 round trip on the sampled latent, and
the conditioning noise being the first draw off the request generator.

The float16 round trip in particular looks like a precision bug and is not, so
``test_float16_round_trip_is_load_bearing`` asserts that it actually changes the
output -- a refactor that "cleans it up" has to fail here.
"""

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
# Two conditions: an fl2va request with both a first and a last keyframe.
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
    """Not [-1, 1]: H3 conditions on VLM-style normalized pixels."""
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
    # A [-1, 1] normalization would land in [-1, 1]; this does not.
    assert pixels.min() < -1.0


def test_posterior_seed_is_independent_of_request_seed():
    """Seeded to 42 internally, so a keyframe encodes identically every request."""
    assert c.MINIMAX_H3_KEYFRAME_ENCODE_SEED == 42
    moments = torch.randn(1, 2 * LATENT_CHANNELS, 1, 8, 8, generator=torch.Generator().manual_seed(1))

    # Perturbing the global RNG must not change the result.
    torch.manual_seed(999)
    first = c.sample_posterior(moments)
    torch.manual_seed(12345)
    second = c.sample_posterior(moments)
    assert torch.equal(first, second)

    assert not torch.equal(first, c.sample_posterior(moments, seed=43))

    # And it is a *sample*, not the posterior mean -- taking the mean would also be
    # request-independent, so that shortcut has to be ruled out here too.
    mean = moments.chunk(2, dim=1)[0]
    assert first.shape == mean.shape
    assert not torch.equal(first, mean)


def test_float16_round_trip_is_load_bearing():
    """The sampled latent is rounded through fp16 before normalization.

    This keeps ~11 bits of every conditioning latent. Asserted to actually change
    the rows so it cannot be silently optimized away.
    """
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
    """A keyframe is a single frame, so no temporal chunking applies."""
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
    """One draw per condition, off the request generator, in packed order.

    Drawing once at the concatenated shape would consume the same number of
    values but assign them differently, and would leave the generator in the same
    state -- so this checks the per-condition split explicitly.
    """
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
    """Conditioning noise precedes the video and audio draws.

    If the order slipped, every latent in the request would change while still
    looking like valid noise.
    """
    ordered = torch.Generator().manual_seed(0)
    c.keyframe_condition_noise(CONDITION_SHAPES, LATENT_CHANNELS, PATCH, ordered)
    video_after_conditioning = torch.randn(3, 64, generator=ordered)

    skipped = torch.Generator().manual_seed(0)
    video_without_conditioning = torch.randn(3, 64, generator=skipped)

    assert not torch.equal(video_after_conditioning, video_without_conditioning)


def test_noise_augmentation_has_no_second_implementation():
    """Noise augmentation must go through the scheduler, not a local copy.

    A local ``t*x0 + (1-t)*noise`` drifted by 2.4e-7 against the scheduler because
    it evaluated ``1 - t`` in Python double instead of float32.
    """
    assert not hasattr(c, "noise_augment")
    # The level itself is the released default; a changed constant would silently retune every anchor.
    assert MINIMAX_H3_KEYFRAME_NOISE_AUG == 0.999

    rows = torch.randn(20, 96, generator=torch.Generator().manual_seed(1))
    noise = torch.randn(20, 96, generator=torch.Generator().manual_seed(2))
    out = MiniMaxH3Scheduler(12.0).scale_noise(rows, MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)

    timestep = torch.tensor(MINIMAX_H3_KEYFRAME_NOISE_AUG, dtype=torch.float32)
    assert torch.equal(out, timestep * rows + (1.0 - timestep) * noise)
    # Python-double arithmetic is what to avoid; prove it is observably different.
    drifted = MINIMAX_H3_KEYFRAME_NOISE_AUG * rows + (1.0 - MINIMAX_H3_KEYFRAME_NOISE_AUG) * noise
    assert not torch.equal(out, drifted)


# ---------------------------------------------------------------------------
# The pipeline's request-draw order, at the production working point
# ---------------------------------------------------------------------------

# 1344x768 / 124 frames: 37 latent frames over a 48x84 latent grid, 207 audio latents. The same shape
# every fl2va and t2va gate runs at, so a t2va bit-identity claim here is about the shape that ships.
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
    """With no keyframes, `draw_request_latents` reproduces the pre-fl2va t2va stream bit-for-bit.

    The t2va no-regression proof, and it costs no device time. Video and audio are drawn inline off
    one generator, so a conditioning draw ahead of them would shift both streams; the empty case must
    consume the generator exactly as the pre-fl2va order did.
    """
    condition_noise, video_rows, audio_rows = _draw(0)
    assert condition_noise is None, "t2va must draw no conditioning noise at all"

    # Exactly what the pipeline did before `draw_request_latents` existed.
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
    """The conditioning noise is the first draw, and it displaces everything behind it.

    Two claims, because only the pair pins the order. The conditioning noise must equal what a fresh
    generator produces first (so nothing is drawn ahead of it), and the video and audio rows must
    *differ* from the no-keyframe case (so it really was consumed from the same stream).
    """
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

    # fl2va at a given seed does NOT reproduce t2va at that seed. Expected, and asserted so nobody
    # reads it as a regression later.
    assert not torch.equal(video_rows, t2va_video), "the conditioning draw did not advance the stream"
    assert not torch.equal(audio_rows, t2va_audio)


def test_keyframe_rows_per_anchor_match_the_layout():
    """One anchor contributes exactly `rows_per_frame` conditioning rows, and the layout agrees.

    The pipeline raises if these disagree; this pins the arithmetic on both sides at the production
    canvas so that check can never be the first place a mismatch is noticed.
    """
    text_tags = torch.ones(39, dtype=torch.long)
    for anchors in (("first",), ("last",), ("first", "last")):
        layout = build_packed_sequence(
            text_tags, PROD_LATENT_FRAMES, PROD_LATENT_H, PROD_LATENT_W, PROD_AUDIO_LATENTS, PATCH, anchors
        )
        assert layout.num_condition_video_rows == len(anchors) * PROD_ROWS_PER_FRAME
        assert layout.num_condition_audio_rows == 0, "fl2va conditions video only; audio conditioning is ref2va"
        # The conditioning rows are the LEADING entries of video_indices and contiguous right after
        # the text rows -- which is what lets the pipeline prepend them and slice them off again.
        expected = torch.arange(39, 39 + layout.num_condition_video_rows)
        assert torch.equal(layout.video_indices[: layout.num_condition_video_rows], expected)

        condition_noise, _, _ = _draw(len(anchors))
        assert condition_noise.shape[0] == layout.num_condition_video_rows


# --- fl2va presentation ---------------------------------------------------------------------------
# Relocated here from `test_text_encoder_minimax_h3.py`, whose other five tests were redundant once
# jonathansu's consolidated conditioner test landed (it covers both the tap comparison and the
# post-norm assertion) and the dedicated mrope tests live in `test_qwen3vl_mrope.py`. This one
# had no equivalent anywhere, and it is host-only, so it belongs in the fast suite rather than behind
# a device fixture.

PROMPT = (
    "A red fox trots across a snowy field at dawn, its breath visible in the cold air. "
    "The low sun throws long blue shadows behind it, and loose snow lifts from each footfall."
)


def _weights_dir() -> str:
    """The text-encoder weights directory, as the device conditioner gates resolve it."""
    base = os.environ.get("MINIMAX_H3_MODEL_PATH") or os.environ.get(
        "MINIMAX_H3_DIFFUSERS_DIR", "/data/cglagovich/MiniMax-H3-diffusers"
    )
    return os.path.join(base, "text_encoder")


def _snapshot_root() -> str:
    """The snapshot directory, i.e. `_weights_dir()`'s parent. Tokenizer and processor live under it."""
    return os.path.dirname(_weights_dir())


def _presentation(prompt, keyframes):
    """`MiniMaxH3Pipeline._build_presentation` called unbound, so no mesh is needed.

    It touches only `self.tokenizer` and `self.image_processor`, so a stub carrying those is enough.
    Calling the real method rather than reimplementing it is the point: a reimplementation here would
    gate a second copy of the presentation and pass while the pipeline's copy was wrong.
    """
    root = _snapshot_root()
    stub = SimpleNamespace(
        tokenizer=transformers.AutoTokenizer.from_pretrained(root, subfolder="tokenizer"),
        image_processor=transformers.AutoImageProcessor.from_pretrained(root, subfolder="text_encoder"),
    )
    return MiniMaxH3Pipeline._build_presentation(stub, prompt, keyframes), stub


@pytest.mark.parametrize("num_keyframes", [1, 2], ids=["first", "first_and_last"])
def test_fl2va_presentation_matches_the_reference(num_keyframes):
    """Token ids, H3 row tags and `mm_token_type_ids` all match the diffusers reference.

    At the production canvas: a 16:9 source resolves to 1344x768, whose patch grid is [1, 48, 84] =
    1008 merged vision patches, so one keyframe is a 1010-row vision block inside a 1028-row
    presentation. Note that 1008 is also `rows_per_frame` -- the same (H/32) x (W/32) grid is read by
    the conditioner as image tokens and by the DiT as conditioning rows.

    Three things are checked because each fails silently on its own:

    - the token ids, including that there is no chat template and no BOS/EOS;
    - **H3's** `token_tags`, where the whole vision block (start/end sentinels included) is
      video-tagged. That tag is what the DiT's AdaLN keys off, so text-tagging it mis-modulates 1010
      rows and no PCC gate anywhere would see it;
    - **Qwen3-VL's** `mm_token_type_ids`, a *different* tagging over the same tokens, which marks only
      the `<|image_pad|>` run. Conflating the two is the easy mistake and this pins both.
    """
    source = Image.fromarray(
        (torch.rand(720, 1280, 3, generator=torch.Generator().manual_seed(0)) * 255).to(torch.uint8).numpy()
    )
    height, width = resolve_canvas_size(*source.size)
    assert (height, width) == (768, 1344), f"expected the production canvas, got {width}x{height}"
    keyframes = [prepare_keyframe_image(source, height, width, stretch=(index == 0)) for index in range(num_keyframes)]

    (input_ids, tags, type_ids, pixel_values, grid_thw), stub = _presentation(PROMPT, keyframes)
    tokenizer, processor = stub.tokenizer, stub.image_processor
    merge = processor.merge_size**2

    # --- the reference's builder, `encoders.py::MiniMaxH3TextEncoderStep.encode_prompt` ---
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

    # `mm_token_type_ids` is Qwen3-VL's own tagging and marks ONLY the image pads -- the vision
    # start/end sentinels are text there, while H3 tags them video. Checked against the processor's own
    # `create_mm_token_type_ids` rather than against our derivation of it.
    full_processor = transformers.AutoProcessor.from_pretrained(_snapshot_root(), subfolder="processor")
    expected_type_ids = torch.tensor(full_processor.create_mm_token_type_ids([input_ids[0].tolist()]))
    assert torch.equal(type_ids, expected_type_ids), "mm_token_type_ids differ from the processor's own"
    assert int(type_ids.sum()) == num_keyframes * 1008

    # The vision block is video-tagged but is NOT the same set of rows as the image pads: it also
    # covers the two sentinels. If these ever coincided, one of the two taggings would be wrong.
    assert int((tags == MINIMAX_H3_VIDEO_TAG).sum()) == num_keyframes * (1008 + 2)
    assert grid_thw.shape[0] == num_keyframes
    assert pixel_values.shape[0] == num_keyframes * 1008 * merge
