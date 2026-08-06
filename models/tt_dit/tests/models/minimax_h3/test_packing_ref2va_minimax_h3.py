# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only parity gate for the MiniMax-H3 ``ref2va`` packed sequence.

Every quantity here is a checkpoint contract, and the failure mode of all of them
is silent: a drifted rotary coordinate desynchronizes audio from video, a
mis-tagged row gets the wrong AdaLN modulation, and a reference block placed at
the wrong offset conditions on the wrong thing -- none of which fails a shape or
finiteness check, and all of which still produce plausible-looking video. So the
assertions are ``torch.equal`` against the installed reference, not PCC.

The reference is ``diffusers.modular_pipelines.minimax_h3.packing_ref2va`` at PR #14355.
Compared against its classes rather than against hand-written expectations, so no part of
the contract is re-derived here.

Host-only, no mesh, a few seconds.
"""

import numpy as np
import pytest
import torch

from ....pipelines.minimax_h3 import packing as p
from ....pipelines.minimax_h3 import packing_ref2va as r

reference_packing = pytest.importorskip(
    "diffusers.modular_pipelines.minimax_h3.packing_ref2va",
    reason="requires the minimax-h3 diffusers branch",
)
# The base half too: the two `_temporal_position_span` implementations live one in
# each module, and this gate pins ours against *both*.
reference_base_packing = pytest.importorskip(
    "diffusers.modular_pipelines.minimax_h3.packing",
    reason="requires the minimax-h3 diffusers branch",
)

PATCH_SIZE = (1, 2, 2)

# The target working point every ref2va gate runs at, and the one the measured padded
# lengths of am. 114 were taken at: 1344x768, 124 frames.
TARGET_HEIGHT, TARGET_WIDTH, TARGET_FRAMES = 768, 1344, 124
VAE_RATIO = 16
AUDIO_HOP = 800
AUDIO_RATE = 32000


def _target():
    """``(num_latent_frames, latent_height, latent_width, num_audio_latents)`` of the generated rows."""
    return (
        p.video_latent_num_frames(TARGET_FRAMES),
        TARGET_HEIGHT // VAE_RATIO,
        TARGET_WIDTH // VAE_RATIO,
        p.audio_latent_num_frames(TARGET_FRAMES),
    )


# ---------------------------------------------------------------- reference specs
#
# A spec says what a reference IS; the geometry is then derived through the
# reference implementation's own sizing rules rather than written down, so a case
# cannot silently drift away from the production geometry. `audio_seconds` is the
# soundtrack duration a video or audio reference carries.


def _image_geometry(source_width: int, source_height: int):
    height, width = r.resolve_reference_image_size(source_width, source_height)
    return dict(num_latent_frames=1, latent_height=height // VAE_RATIO, latent_width=width // VAE_RATIO)


def _video_geometry(source_width: int, source_height: int, num_frames: int = TARGET_FRAMES):
    # Through both passes the reference applies, in order, at 24 fps.
    frames = np.zeros((num_frames, source_height, source_width, 3), dtype=np.uint8)
    frames = r.prepare_reference_frames(frames, TARGET_FRAMES)
    trimmed = r.trim_reference_num_frames(frames.shape[0])
    return dict(
        num_latent_frames=p.video_latent_num_frames(trimmed),
        latent_height=frames.shape[1] // VAE_RATIO,
        latent_width=frames.shape[2] // VAE_RATIO,
    )


def _audio_latents(seconds: float) -> int:
    """What the audio VAE produces: the samples right-padded up to a whole 800-sample hop."""
    return int(np.ceil(int(seconds * AUDIO_RATE) / AUDIO_HOP))


def _spec_to_prepared(spec, dataclass):
    """One spec into either dataclass -- ours or the reference's. Same fields, by design."""
    kind = spec["kind"]
    geometry = {}
    if kind == "image":
        geometry = _image_geometry(*spec["source"])
    elif kind == "video":
        geometry = _video_geometry(*spec["source"], spec.get("num_frames", TARGET_FRAMES))
    audio_seconds = spec.get("audio_seconds")
    return dataclass(
        kind=kind,
        has_audio=audio_seconds is not None,
        num_audio_latents=0 if audio_seconds is None else _audio_latents(audio_seconds),
        **geometry,
    )


def _both(spec_list):
    """``(ours, theirs)`` prepared-reference lists built from one spec list."""
    ours = [_spec_to_prepared(spec, r.MiniMaxH3PreparedReference) for spec in spec_list]
    theirs = [_spec_to_prepared(spec, reference_packing.MiniMaxH3PreparedReference) for spec in spec_list]
    return ours, theirs


DURATION = TARGET_FRAMES / p.MINIMAX_H3_FPS  # 5.1667 s

IMAGE_1TO1 = dict(kind="image", source=(1024, 1024))
IMAGE_16TO9 = dict(kind="image", source=(1920, 1080))
VIDEO_SOUND = dict(kind="video", source=(1344, 768), audio_seconds=DURATION)
# A 1:1 sounded video against the 16:9 target: its latent width differs from the
# target's, so its soundtrack rows must pin to its OWN width grid. A soundtrack pinned
# to the target grid instead keeps every length, tag and index correct.
VIDEO_SOUND_1TO1 = dict(kind="video", source=(768, 768), audio_seconds=DURATION)
VIDEO_SILENT = dict(kind="video", source=(1344, 768))
AUDIO_ONLY = dict(kind="audio", audio_seconds=DURATION)

# `nine_mixed_awkward` is awkward by construction: an audio reference between two videos, a
# video before an image, and a silent video beside a sounded one. That exercises the
# per-modality label counters, the soundtrack-before-video row order and the shared rotary
# clock out of their natural order. Nine references, inside the 9/3/3/12 limits.
#
# `nine_mixed_reversed` is the same nine in the opposite order. Reference order is
# semantic -- it moves both the rotary clock and the row modalities -- so a layout
# assembled order-blind matches the reference on neither ordering.
_NINE_MIXED = [
    VIDEO_SOUND,
    AUDIO_ONLY,
    VIDEO_SILENT,
    IMAGE_1TO1,
    AUDIO_ONLY,
    IMAGE_16TO9,
    VIDEO_SOUND,
    IMAGE_1TO1,
    AUDIO_ONLY,
]

CASES = {
    "one_image": [IMAGE_1TO1],
    "one_audio": [IMAGE_1TO1, AUDIO_ONLY],  # audio may never be the only reference
    "video_with_sound": [VIDEO_SOUND],
    "video_with_sound_1to1": [VIDEO_SOUND_1TO1],
    "video_without_sound": [VIDEO_SILENT],
    "nine_mixed_awkward": _NINE_MIXED,
    "nine_mixed_reversed": list(reversed(_NINE_MIXED)),
}

TEXT_LEN = 1024
# A reference's vision block lands inside the text rows tagged as VIDEO, not text.
VISION_BLOCK = slice(30, 120)


def _text_tags():
    tags = torch.ones(TEXT_LEN, dtype=torch.long)
    tags[VISION_BLOCK] = p.MINIMAX_H3_VIDEO_TAG
    return tags


# ------------------------------------------------------------------- the rotary clock


def test_span_matches_both_reference_summation_orders():
    """Each summation order against its own reference function.

    ``packing._temporal_position_span`` reproduces a numpy pairwise sum;
    ``packing_ref2va._temporal_position_span`` sums sequentially in float64. Exact ``==``:
    the difference under test is a float64 last ulp.
    """
    for n in range(1, 61):
        assert p._temporal_position_span(n) == reference_base_packing._temporal_position_span(
            n
        ), f"pairwise span disagrees with the reference at n={n}"
        assert r._temporal_position_span(n) == reference_packing._temporal_position_span(
            n
        ), f"sequential span disagrees with the reference at n={n}"


def test_the_two_span_orders_actually_differ():
    """The gate above is only a gate if the two orders are distinguishable.

    They agree below 16 latent frames and diverge from 16 on -- and at the
    production 37 they differ by ~2 ulp in the OPPOSITE direction to n=16, so a
    test written only at 16 would not pin the sign. A single shared implementation
    would pass every small case and be wrong at the working point.
    """
    assert p._temporal_position_span(15) == r._temporal_position_span(15), "expected agreement below 16"
    assert p._temporal_position_span(16) != r._temporal_position_span(16), "expected divergence from 16 latent frames"

    production = p.video_latent_num_frames(TARGET_FRAMES)
    assert production == 37
    pairwise, sequential = p._temporal_position_span(production), r._temporal_position_span(production)
    assert pairwise != sequential, "the two orders must differ at the production frame count"
    # Directions differ between n=16 and n=37; assert both so a "fix" that aligns
    # them cannot pass.
    assert r._temporal_position_span(16) > p._temporal_position_span(16)
    assert sequential < pairwise


# --------------------------------------------------------------------- the layout


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_layout_matches_reference(case):
    """Bit-exact against the reference: every field the transformer addresses rows through."""
    ours, theirs = _both(CASES[case])
    num_latent_frames, latent_height, latent_width, num_audio_latents = _target()
    args = (num_latent_frames, latent_height, latent_width, num_audio_latents, PATCH_SIZE)

    got = r.build_ref2va_packed_sequence(_text_tags(), ours, *args)
    want = reference_packing.build_ref2va_packed_sequence(_text_tags(), theirs, *args)

    assert got.sequence_length == want.sequence_length
    assert got.position_ids.dtype == torch.float64
    assert torch.equal(got.position_ids, want.position_ids), "the fp64 rotary grid is not bit-exact"
    assert torch.equal(got.token_tags, want.token_tags)
    assert torch.equal(got.video_indices, want.video_indices)
    assert torch.equal(got.audio_indices, want.audio_indices)
    assert torch.equal(got.text_indices, want.text_indices)
    assert got.num_condition_video_rows == want.num_condition_video_rows
    assert got.num_condition_audio_rows == want.num_condition_audio_rows


# ---------------------------------------------------------------- media preparation


# Note the argument and return orders differ: `(width, height)` in, `(height, width)`
# out, matching the reference. Both orientations are listed so a transposed
# implementation cannot pass.
@pytest.mark.parametrize(
    "source,expected",
    [
        ((1024, 1024), (2048, 2048)),
        ((1920, 1080), (2048, 3648)),  # 16:9, no area cap -- the long edge is free to grow
        ((1080, 1920), (3648, 2048)),
        ((4096, 1024), (2048, 8192)),  # 4:1, the documented extreme: 65536 patches for ONE image
    ],
)
def test_resolve_reference_image_size(source, expected):
    assert r.resolve_reference_image_size(*source) == expected
    assert r.resolve_reference_image_size(*source) == reference_packing.resolve_reference_image_size(*source)
    height, width = expected
    assert height % p.MINIMAX_H3_CANVAS_MULTIPLE == 0 and width % p.MINIMAX_H3_CANVAS_MULTIPLE == 0
    # No area cap, unlike the target canvas: this is 4x the short edge of a canvas
    # and is what makes one reference image cost thousands of rows.
    assert min(height, width) == r.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE


def test_resolve_reference_image_size_rejects_out_of_range(expect_error):
    with expect_error(ValueError, "within 1:4 and 4:1"):
        r.resolve_reference_image_size(5000, 1000)
    with expect_error(ValueError, "positive size"):
        r.resolve_reference_image_size(0, 100)


@pytest.mark.parametrize("fps", [24.0, 30.0, 25.0, 12.0, 60.0, 23.976])
def test_resample_reference_frames_matches_reference(fps):
    """Constant-frame-rate resampling onto 24 fps, frame selection included."""
    frames = np.arange(50 * 2 * 2 * 3, dtype=np.uint8).reshape(50, 2, 2, 3)
    got = r.resample_reference_frames(frames, fps)
    want = reference_packing.resample_reference_frames(frames, fps)
    assert got.shape == want.shape
    assert np.array_equal(got, want)
    if fps == 24.0:
        # The parity-exact route: the same array, not a copy.
        assert got is frames


def test_prepare_reference_frames_uses_its_own_aspect_canvas():
    """A reference video goes onto the canvas of ITS OWN aspect ratio, not the target's."""
    frames = np.zeros((10, 480, 480, 3), dtype=np.uint8)
    prepared = r.prepare_reference_frames(frames, TARGET_FRAMES)
    assert prepared.shape[1:3] == p.resolve_canvas_size(480, 480) == (768, 768)
    assert np.array_equal(prepared, reference_packing.prepare_reference_frames(frames, TARGET_FRAMES))

    # Already at its canvas: no resampling pass and no copy. `shares_memory`, not
    # `is` -- the frame-count cap slices first, so what comes back is a view of the
    # input rather than the input itself, and only a view keeps the pixels bit-exact.
    at_canvas = np.zeros((10, 768, 1344, 3), dtype=np.uint8)
    passthrough = r.prepare_reference_frames(at_canvas, TARGET_FRAMES)
    assert np.shares_memory(passthrough, at_canvas)

    # And truncated to the generated frame count.
    long_clip = np.zeros((TARGET_FRAMES + 40, 768, 1344, 3), dtype=np.uint8)
    assert r.prepare_reference_frames(long_clip, TARGET_FRAMES).shape[0] == TARGET_FRAMES


@pytest.mark.parametrize("num_frames", [1, 5, 12, 24, 25, 124, 192])
def test_sample_reference_video_frames_matches_reference(num_frames):
    """2 fps sampling, pair merging and the round-half-to-even timestamps."""
    frames = np.zeros((num_frames, 4, 4, 3), dtype=np.uint8)
    got_frames, got_ts = r.sample_reference_video_frames(frames)
    want_frames, want_ts = reference_packing.sample_reference_video_frames(frames)
    assert len(got_frames) == len(want_frames)
    assert got_ts == want_ts
    # One vision block per merged pair of sampled frames.
    assert len(got_ts) == -(-len(got_frames) // r.MINIMAX_H3_QWEN_TEMPORAL_PATCH)


def test_block_timestamps_round_half_to_even():
    """ "{:.1f}" on the mean of a 2 fps pair gives "<0.2 seconds>", not "<0.3 seconds>".

    0.25 is exactly representable and Python rounds it to even, so this is a real
    formatting contract rather than a floating-point accident.
    """
    frames = np.zeros((124, 4, 4, 3), dtype=np.uint8)
    _, timestamps = r.sample_reference_video_frames(frames)
    assert timestamps[0] == 0.25
    assert f"{timestamps[0]:.1f}" == "0.2"


@pytest.mark.parametrize("num_frames", [1, 5, 22, 39, 124, 125, 130])
def test_trim_reference_num_frames_matches_reference(num_frames):
    got = r.trim_reference_num_frames(num_frames)
    assert got == reference_packing.trim_reference_num_frames(num_frames)
    assert got % p.MINIMAX_H3_FRAMES_PER_CHUNK == p.MINIMAX_H3_LATENTS_PER_CHUNK


def test_trim_reference_num_frames_has_a_22_frame_floor():
    """``max(1, ...)`` forces at least one whole chunk, so the floor is 22 and not 5.

    Which means the function does *not* only round down, despite reading that way:
    1, 5, 10 and 23 frames all map to 22. For a clip shorter than 22 the caller's
    ``frames[:trim(n)]`` is then a no-op slice and the video VAE's own
    repeat-the-last-frame padding takes over -- so nothing breaks, but "snap down"
    is only true from 22 up. Pinned because a port that "fixed" the floor to 5
    would silently encode a different number of latent frames for short references,
    and the layout is built from that count.
    """
    for num_frames in (1, 5, 10, 22, 23):
        assert r.trim_reference_num_frames(num_frames) == 22
        assert r.trim_reference_num_frames(num_frames) == reference_packing.trim_reference_num_frames(num_frames)
    # From 39 up it genuinely rounds down.
    assert r.trim_reference_num_frames(39) == 39
    assert r.trim_reference_num_frames(55) == 39
    assert r.trim_reference_num_frames(125) == 124


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("sample_rate", [32000, 44100])
def test_prepare_reference_waveform_matches_reference(channels, sample_rate):
    """Stereo upmix, truncation at the native rate, then one resample pass."""
    torch.manual_seed(0)
    waveform = torch.randn(channels, int(6.0 * sample_rate))
    got = r.prepare_reference_waveform(waveform, sample_rate, AUDIO_RATE, DURATION)
    want = reference_packing.prepare_reference_waveform(waveform, sample_rate, AUDIO_RATE, DURATION)
    assert got.shape == want.shape
    assert torch.equal(got, want)
    assert got.shape[0] == p.MINIMAX_H3_AUDIO_CHANNELS
    if sample_rate == AUDIO_RATE:
        # Truncated to the generated duration, at its own rate.
        assert got.shape[1] == int(DURATION * sample_rate)


def test_reference_media_to_uint8_matches_reference():
    """Channels-first for a tensor, channels-last for an array, floats read over [0, 1]."""
    torch.manual_seed(0)
    tensor = torch.rand(3, 8, 6)
    array = (np.random.default_rng(0).random((8, 6, 3)) * 255).astype(np.uint8)
    for media in (tensor, array, [tensor, tensor]):
        got = r.reference_media_to_uint8(media)
        want = reference_packing.reference_media_to_uint8(media)
        assert got.dtype == np.uint8
        assert np.array_equal(got, want)


# ------------------------------------------------------------------- the reference dataclass


def test_reference_requires_exactly_one_medium(expect_error):
    """A video may carry audio; nothing else may carry two media."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    waveform = torch.zeros(2, 800)

    assert r.MiniMaxH3Reference(image=image).kind == "image"
    assert r.MiniMaxH3Reference(video=video).kind == "video"
    assert r.MiniMaxH3Reference(audio=waveform).kind == "audio"
    sounded = r.MiniMaxH3Reference(video=video, audio=waveform)
    assert sounded.kind == "video" and sounded.has_audio

    with expect_error(ValueError, "exactly one"):
        r.MiniMaxH3Reference(image=image, video=video)
    with expect_error(ValueError, "exactly one"):
        r.MiniMaxH3Reference(image=image, audio=waveform)
    with expect_error(ValueError, "exactly one"):
        r.MiniMaxH3Reference()


def test_reference_refuses_a_path(expect_error):
    """This module never opens media files, so a path is an error rather than a decode."""
    with expect_error(ValueError, "never opens media files"):
        r.MiniMaxH3Reference(image="subject.png")


def test_reference_defaults_fps_to_the_models_own():
    """No fps means "already at 24", so the frames flow through untouched."""
    video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    assert r.MiniMaxH3Reference(video=video).fps == float(p.MINIMAX_H3_FPS)


def test_documented_limits_match_the_reference():
    """The per-request ceilings are a checkpoint contract, not our choice."""
    assert r.MINIMAX_H3_MAX_REFERENCE_IMAGES == reference_packing.MINIMAX_H3_MAX_REFERENCE_IMAGES == 9
    assert r.MINIMAX_H3_MAX_REFERENCE_VIDEOS == reference_packing.MINIMAX_H3_MAX_REFERENCE_VIDEOS == 3
    assert r.MINIMAX_H3_MAX_REFERENCE_AUDIOS == reference_packing.MINIMAX_H3_MAX_REFERENCE_AUDIOS == 3
    assert r.MINIMAX_H3_MAX_REFERENCES == reference_packing.MINIMAX_H3_MAX_REFERENCES == 12
    assert r.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE == reference_packing.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE == 2048


# --------------------------------------------------------------------- the presentation


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_presentation_matches_reference(case):
    """Token ids AND tags, against the reference's own builder, on the real tokenizer.

    The tags are the load-bearing half and the easier one to get wrong: H3 tags the
    **whole** vision block as video, sentinels included, and that tag is what the
    DiT's AdaLN keys off.
    """
    tokenizer = _tokenizer()
    ours, theirs = _both(CASES[case])
    # Vision-token counts the conditioner's processor would produce. The exact
    # values do not matter to the presentation's structure, but they must be the
    # same on both sides, and one per image / per video.
    image_counts = [4096 for reference in ours if reference.kind == "image"]
    video_counts = [1008 for reference in ours if reference.kind == "video"]
    for prepared in (ours, theirs):
        for reference in prepared:
            if reference.kind == "video":
                reference.block_timestamps = [0.25, 1.25, 2.25, 3.25, 4.25, 5.0]

    prompt = "a red fox stepping through wet grass at dawn"
    got_ids, got_tags = r.build_ref2va_presentation(tokenizer, prompt, ours, image_counts, video_counts)
    want_ids, want_tags = reference_packing.build_ref2va_presentation(
        tokenizer, prompt, theirs, image_counts, video_counts
    )
    assert got_ids == want_ids
    assert got_tags == want_tags
    assert len(got_tags) == len(got_ids)


def _tokenizer():
    """The checkpoint's own tokenizer. It decides the token ids, so nothing else may."""
    import os
    from pathlib import Path

    weights = os.environ.get("MINIMAX_H3_DIFFUSERS_DIR") or os.environ.get("MINIMAX_H3_MODEL_PATH")
    if not weights or not (Path(weights) / "tokenizer").is_dir():
        pytest.skip("needs MINIMAX_H3_DIFFUSERS_DIR pointing at a diffusers snapshot with tokenizer/")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(weights), subfolder="tokenizer")
