# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 ``ref2va`` reference conditioning: everything except the e2e pipeline.

One file for the whole subsystem -- request validation, per-reference preparation, media
preparation, the packed-sequence layout, the presentation, and the device VAE encode -- because
every one of these gates the same contract: the reference implementation's own ``ref2va`` path
(``diffusers.modular_pipelines.minimax_h3`` at PR #14355), imported and compared against rather
than re-derived. Three kinds of gate, split by what they need:

* **Host, bit-exact** -- preparation, layout, media prep, presentation. Every quantity is a
  checkpoint contract whose failure mode is silent: a drifted rotary coordinate desynchronizes
  audio from video, a mis-tagged row gets the wrong AdaLN modulation, and a reference block
  placed at the wrong offset conditions on the wrong thing -- none of which fails a shape or
  finiteness check, and all of which still produce plausible-looking video. So the assertions
  are ``torch.equal`` / ``np.array_equal`` against the installed reference, not PCC. No mesh,
  seconds.
* **Device, PCC** (``test_encode_references_matches_reference``) -- the real VAE encode against
  ``MiniMaxH3Ref2VAReferenceEncoderStep``, on **real media**, parametrized per modality. This is
  the per-modality gate that lets the e2e suite run a single (``mixed``) case: each modality's
  encode is held to the reference here, so the e2e only has to show the composition reaches the
  output. The image runs at its full production resolution and the soundtrack at its full
  duration; the video runs at a *reduced frame count* on the production canvas, because the
  reference's video VAE runs on CPU (see the note above that case). ``randn`` would not exercise
  the fp16 round trip on natural statistics, which is the thing under test.
* **Presentation, two layers** -- bit-exact token/tag parity against the reference's own builder
  (``test_presentation_matches_reference``), and structural gates on the pipeline's real
  presentation path (vision-run ordering, token-type ids). Complementary: parity pins the
  contract, structure pins the failure modes parity cannot name.

The three encode recipes differ per modality and none of them is guessable from the others:
image and video posteriors are *sampled* under a generator seeded 42 and rounded through
float16, while a soundtrack takes the posterior *mean* untouched.

This module ``importorskip``s the minimax-h3 diffusers branch at import time.
``test_packing_minimax_h3.py`` (t2va/fl2va) is a separate file on purpose: its golden digests
are designed to stand in when that branch is absent, which the skip here would defeat.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from loguru import logger

from ....pipelines.minimax_h3 import packing as p
from ....pipelines.minimax_h3 import packing_ref2va as rp
from ....pipelines.minimax_h3 import references as R
from ....utils.check import assert_quality
from ....utils.test import ring_params_req_exact_devices

# The base `packing` half too: the two `_temporal_position_span` implementations live one
# in each module, and the rotary-clock gate pins ours against *both*.
reference_before_encoder, reference_packing, reference_encoders, reference_base_packing = (
    pytest.importorskip(module, reason="requires the minimax-h3 diffusers branch")
    for module in (
        "diffusers.modular_pipelines.minimax_h3.before_encoder",
        "diffusers.modular_pipelines.minimax_h3.packing_ref2va",
        "diffusers.modular_pipelines.minimax_h3.encoders",
        "diffusers.modular_pipelines.minimax_h3.packing",
    )
)

AUDIO_RATE = 32000
TARGET_FRAMES = 124
DURATION = TARGET_FRAMES / p.MINIMAX_H3_FPS  # 5.1667 s

# The target working point every ref2va gate runs at, and the one the measured padded
# lengths were taken at: 1344x768, 124 frames.
TARGET_HEIGHT, TARGET_WIDTH = 768, 1344
VAE_RATIO = 16
AUDIO_HOP = 800
PATCH_SIZE = (1, 2, 2)


def _components():
    """The one attribute the reference's setup step reads off its pipeline."""
    return SimpleNamespace(audio_sampling_rate=AUDIO_RATE)


def _rng():
    return np.random.default_rng(0)


def _image(width: int, height: int) -> np.ndarray:
    """Structured content, not flat: a resize of a constant field cannot show an error."""
    rng = _rng()
    return (rng.random((height, width, 3)) * 255).astype(np.uint8)


def _video(width: int, height: int, num_frames: int) -> np.ndarray:
    rng = _rng()
    return (rng.random((num_frames, height, width, 3)) * 255).astype(np.uint8)


def _waveform(seconds: float, sample_rate: int = AUDIO_RATE, channels: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(channels, int(seconds * sample_rate)) * 0.1


def _pair(specs: list[dict]):
    """``(ours, theirs)`` reference lists built from one spec list, as ``_both`` does below."""
    ours = [rp.MiniMaxH3Reference(**spec) for spec in specs]
    theirs = [reference_packing.MiniMaxH3Reference(**spec) for spec in specs]
    return ours, theirs


SPECS = {
    "image": dict(image=_image(1024, 1024)),
    "image_16to9": dict(image=_image(1920, 1080)),
    "video_24fps_at_canvas": dict(video=_video(1344, 768, TARGET_FRAMES), fps=24.0),
    "video_30fps_off_canvas": dict(video=_video(1920, 1080, 60), fps=30.0),
    "video_with_sound": dict(video=_video(1344, 768, TARGET_FRAMES), fps=24.0, audio=_waveform(DURATION)),
    "audio": dict(audio=_waveform(DURATION)),
    "audio_44k_mono": dict(audio=_waveform(DURATION, 44100, 1), sample_rate=44100),
}


# ------------------------------------------------------------------ request limits


def test_reference_limits_match_reference_and_are_enforced(expect_error):
    """The per-request ceilings are a checkpoint contract, and ``check_references`` enforces them.

    One test for both halves because they are one claim: the constants must equal the
    reference's (they are not our choice), and the enforcement must key off exactly those
    constants -- ceilings that match but are not applied, or are applied at different values,
    both condition a request the checkpoint was never trained on.
    """
    # Parity: the documented ceilings, against the reference's own constants.
    assert rp.MINIMAX_H3_MAX_REFERENCE_IMAGES == reference_packing.MINIMAX_H3_MAX_REFERENCE_IMAGES == 9
    assert rp.MINIMAX_H3_MAX_REFERENCE_VIDEOS == reference_packing.MINIMAX_H3_MAX_REFERENCE_VIDEOS == 3
    assert rp.MINIMAX_H3_MAX_REFERENCE_AUDIOS == reference_packing.MINIMAX_H3_MAX_REFERENCE_AUDIOS == 3
    assert rp.MINIMAX_H3_MAX_REFERENCES == reference_packing.MINIMAX_H3_MAX_REFERENCES == 12
    assert rp.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE == reference_packing.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE == 2048

    # Enforcement: each ceiling, plus the "audio not alone" rule.
    image = rp.MiniMaxH3Reference(image=_image(256, 256))
    video = rp.MiniMaxH3Reference(video=_video(256, 256, 10), fps=24.0)
    audio = rp.MiniMaxH3Reference(audio=_waveform(DURATION))

    assert R.check_references([image, video, audio]) == ["image", "video", "audio"]

    with expect_error(ValueError, "at least one reference"):
        R.check_references([])
    with expect_error(ValueError, "at most 9 image"):
        R.check_references([image] * 10)
    with expect_error(ValueError, "at most 3 video"):
        R.check_references([video] * 4)
    with expect_error(ValueError, "at most 3 audio"):
        R.check_references([image, audio, audio, audio, audio])
    with expect_error(ValueError, "at most 12 references"):
        R.check_references([image] * 9 + [video] * 3 + [audio])
    with expect_error(ValueError, "paired with at least one image or video"):
        R.check_references([audio])


# --------------------------------------------------------------------- preparation


@pytest.mark.parametrize("name", list(SPECS), ids=list(SPECS))
def test_prepare_references_matches_reference(name):
    """Per-reference preparation, bit-exact against the reference's own setup step.

    Every reference is prepared at **its own** resolution, so this is where a
    reference wrongly forced onto the target canvas would show up -- and that
    mistake would still produce a runnable request, just one conditioned at the
    wrong scale.
    """
    spec = SPECS[name]
    # An audio reference may not be alone, so pair it with an image; the image is
    # prepared identically either way and the pairing is what the reference allows.
    specs = [spec] if "audio" not in name else [SPECS["image"], spec]
    ours, theirs = _pair(specs)

    got, got_frames = R.prepare_references(ours, TARGET_FRAMES, AUDIO_RATE)
    want, want_frames = reference_before_encoder.MiniMaxH3Ref2VASetupStep.prepare_references(
        _components(), theirs, TARGET_FRAMES
    )

    assert got_frames == want_frames == TARGET_FRAMES
    assert len(got) == len(want)
    for index, (a, b) in enumerate(zip(got, want)):
        assert a.kind == b.kind, index
        assert a.has_audio == b.has_audio, index
        if a.kind == "image":
            assert np.array_equal(np.asarray(a.image), np.asarray(b.image)), f"prepared image {index} differs"
            assert a.image.size == b.image.size
        if a.kind == "video":
            assert np.array_equal(a.frames, b.frames), f"prepared frames {index} differ"
        if a.has_audio:
            assert torch.equal(a.waveform, b.waveform), f"prepared waveform {index} differs"
            assert a.waveform.shape[0] == p.MINIMAX_H3_AUDIO_CHANNELS


def test_prepare_references_uses_each_references_own_resolution():
    """An image at 2048 short edge, a video at its own 768 canvas -- neither the target's.

    Two sizing rules feed one request, 2048 px against 768 px on the short edge. Using one
    where the other belongs conditions at the wrong scale and fails no shape check, since
    both produce a valid request.
    """
    references = [
        rp.MiniMaxH3Reference(image=_image(1024, 1024)),
        rp.MiniMaxH3Reference(video=_video(1920, 1080, 30), fps=24.0),
    ]
    prepared, _ = R.prepare_references(references, TARGET_FRAMES, AUDIO_RATE)

    # 2048 px short edge, no area cap.
    assert prepared[0].image.size == (2048, 2048)
    assert min(prepared[0].image.size) == rp.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE
    # The 768 px canvas of the video's own 16:9, which equals the target here. The
    # assertion that matters is 768-something rather than 2048-something.
    assert prepared[1].frames.shape[1:3] == (768, 1344)
    assert min(prepared[1].frames.shape[1:3]) == p.MINIMAX_H3_SHORT_EDGE
    # 2048 against 768: an image reference is encoded at 2.67x a video reference's short
    # edge, so one image costs 4096 rows where one video frame costs 1008. The 4x the
    # vision-tower test quotes is at the aspect extremes, where the area cap pushes the
    # short edge to 32 patches.
    assert min(prepared[0].image.size) > min(prepared[1].frames.shape[1:3])


def test_prepare_references_truncates_a_video_and_its_soundtrack_to_the_target():
    """A longer reference is cut to the generated duration, on both media."""
    long_frames = _video(1344, 768, TARGET_FRAMES + 60)
    references = [rp.MiniMaxH3Reference(video=long_frames, fps=24.0, audio=_waveform(10.0))]
    prepared, num_frames = R.prepare_references(references, TARGET_FRAMES, AUDIO_RATE)

    assert num_frames == TARGET_FRAMES
    assert prepared[0].frames.shape[0] == TARGET_FRAMES
    assert prepared[0].waveform.shape[1] == int(DURATION * AUDIO_RATE)


def test_num_frames_may_be_derived_from_a_single_audio_bearing_reference():
    """Left open, the duration is the one soundtrack's -- and only if there is exactly one."""
    references = [rp.MiniMaxH3Reference(image=_image(512, 512)), rp.MiniMaxH3Reference(audio=_waveform(6.0))]
    _, num_frames = R.prepare_references(references, None, AUDIO_RATE)
    assert num_frames == p.align_num_frames(round(6.0 * p.MINIMAX_H3_FPS))
    assert num_frames % p.MINIMAX_H3_FRAMES_PER_CHUNK == p.MINIMAX_H3_LATENTS_PER_CHUNK
    # And the reference agrees.
    theirs = [
        reference_packing.MiniMaxH3Reference(image=_image(512, 512)),
        reference_packing.MiniMaxH3Reference(audio=_waveform(6.0)),
    ]
    _, want = reference_before_encoder.MiniMaxH3Ref2VASetupStep.prepare_references(_components(), theirs, None)
    assert num_frames == want


def test_num_frames_is_ambiguous_with_two_soundtracks(expect_error):
    references = [
        rp.MiniMaxH3Reference(video=_video(768, 768, 24), fps=24.0, audio=_waveform(6.0)),
        rp.MiniMaxH3Reference(audio=_waveform(7.0)),
    ]
    with expect_error(ValueError, "exactly one of them carries audio"):
        R.prepare_references(references, None, AUDIO_RATE)


@pytest.mark.parametrize("seconds", [4.0, 16.0])
def test_a_derived_duration_outside_the_models_range_is_rejected(seconds, expect_error):
    references = [rp.MiniMaxH3Reference(image=_image(512, 512)), rp.MiniMaxH3Reference(audio=_waveform(seconds))]
    with expect_error(ValueError, "seconds"):
        R.prepare_references(references, None, AUDIO_RATE)


# ------------------------------------------------------------------ the audio hop


@pytest.mark.parametrize("seconds", [DURATION, 5.0, 8.0, 15.0])
def test_pad_waveform_to_hop_reproduces_the_reference_latent_count(seconds):
    """Zero-padding to a whole 800-sample hop, and the latent count it implies.

    Our device audio encoder asserts divisibility where the reference VAE right-pads
    internally, so this padding is the port's own step. It must land on the same
    number of latents the reference produces, because the layout is built from it --
    and the production 5.1667 s is 165333 samples, which is *not* a multiple of 800.
    """
    waveform = _waveform(seconds)
    padded = R.pad_waveform_to_hop(waveform)

    assert padded.shape[-1] % R.MINIMAX_H3_AUDIO_HOP == 0
    assert padded.shape[-1] - waveform.shape[-1] < R.MINIMAX_H3_AUDIO_HOP
    assert torch.equal(padded[:, : waveform.shape[-1]], waveform), "padding must not disturb the samples"
    assert (padded[:, waveform.shape[-1] :] == 0).all(), "the pad is zeros, as F.pad's default is"
    # The latent count the layout will be built from.
    assert padded.shape[-1] // R.MINIMAX_H3_AUDIO_HOP == int(np.ceil(waveform.shape[-1] / R.MINIMAX_H3_AUDIO_HOP))


def test_production_soundtrack_is_not_a_whole_number_of_hops():
    """The case that makes the padding load-bearing rather than defensive."""
    samples = int(DURATION * AUDIO_RATE)
    assert samples == 165333
    assert samples % R.MINIMAX_H3_AUDIO_HOP != 0
    assert R.pad_waveform_to_hop(_waveform(DURATION)).shape[-1] // R.MINIMAX_H3_AUDIO_HOP == 207


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
        assert rp._temporal_position_span(n) == reference_packing._temporal_position_span(
            n
        ), f"sequential span disagrees with the reference at n={n}"


def test_the_two_span_orders_actually_differ():
    """The gate above is only a gate if the two orders are distinguishable.

    They agree below 16 latent frames and diverge from 16 on -- and at the
    production 37 they differ by ~2 ulp in the OPPOSITE direction to n=16, so a
    test written only at 16 would not pin the sign. A single shared implementation
    would pass every small case and be wrong at the working point.
    """
    assert p._temporal_position_span(15) == rp._temporal_position_span(15), "expected agreement below 16"
    assert p._temporal_position_span(16) != rp._temporal_position_span(16), "expected divergence from 16 latent frames"

    production = p.video_latent_num_frames(TARGET_FRAMES)
    assert production == 37
    pairwise, sequential = p._temporal_position_span(production), rp._temporal_position_span(production)
    assert pairwise != sequential, "the two orders must differ at the production frame count"
    # Directions differ between n=16 and n=37; assert both so a "fix" that aligns
    # them cannot pass.
    assert rp._temporal_position_span(16) > p._temporal_position_span(16)
    assert sequential < pairwise


# --------------------------------------------------------------------- the layout
#
# A spec says what a reference IS; the geometry is then derived through the
# reference implementation's own sizing rules rather than written down, so a case
# cannot silently drift away from the production geometry. `audio_seconds` is the
# soundtrack duration a video or audio reference carries.


def _target():
    """``(num_latent_frames, latent_height, latent_width, num_audio_latents)`` of the generated rows."""
    return (
        p.video_latent_num_frames(TARGET_FRAMES),
        TARGET_HEIGHT // VAE_RATIO,
        TARGET_WIDTH // VAE_RATIO,
        p.audio_latent_num_frames(TARGET_FRAMES),
    )


def _image_geometry(source_width: int, source_height: int):
    height, width = rp.resolve_reference_image_size(source_width, source_height)
    return dict(num_latent_frames=1, latent_height=height // VAE_RATIO, latent_width=width // VAE_RATIO)


def _video_geometry(source_width: int, source_height: int, num_frames: int = TARGET_FRAMES):
    # Through both passes the reference applies, in order, at 24 fps.
    frames = np.zeros((num_frames, source_height, source_width, 3), dtype=np.uint8)
    frames = rp.prepare_reference_frames(frames, TARGET_FRAMES)
    trimmed = rp.trim_reference_num_frames(frames.shape[0])
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
    ours = [_spec_to_prepared(spec, rp.MiniMaxH3PreparedReference) for spec in spec_list]
    theirs = [_spec_to_prepared(spec, reference_packing.MiniMaxH3PreparedReference) for spec in spec_list]
    return ours, theirs


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
# assembled order-blind matches the reference on neither ordering. This pair is also the
# host order-sensitivity gate, standing in for an e2e generation pair: the layout change
# is pinned bit-exact here, and `test_ref2va_conditioning_is_not_a_no_op` proves layout
# differences reach the output.
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


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_layout_matches_reference(case):
    """Bit-exact against the reference: every field the transformer addresses rows through."""
    ours, theirs = _both(CASES[case])
    num_latent_frames, latent_height, latent_width, num_audio_latents = _target()
    args = (num_latent_frames, latent_height, latent_width, num_audio_latents, PATCH_SIZE)

    got = rp.build_ref2va_packed_sequence(_text_tags(), ours, *args)
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
#
# The unique edge cases the layout depends on. Happy-path composition is covered
# bit-exact by `test_prepare_references_matches_reference`'s 7-spec composition above;
# this section covers what those specs do not reach -- the fps grid, the 22-frame trim
# floor, the round-half-to-even timestamps, the aspect extremes, and the input-form
# handling of `reference_media_to_uint8`.


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
    assert rp.resolve_reference_image_size(*source) == expected
    assert rp.resolve_reference_image_size(*source) == reference_packing.resolve_reference_image_size(*source)
    height, width = expected
    assert height % p.MINIMAX_H3_CANVAS_MULTIPLE == 0 and width % p.MINIMAX_H3_CANVAS_MULTIPLE == 0
    # No area cap, unlike the target canvas: this is 4x the short edge of a canvas
    # and is what makes one reference image cost thousands of rows.
    assert min(height, width) == rp.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE


def test_resolve_reference_image_size_rejects_out_of_range(expect_error):
    with expect_error(ValueError, "within 1:4 and 4:1"):
        rp.resolve_reference_image_size(5000, 1000)
    with expect_error(ValueError, "positive size"):
        rp.resolve_reference_image_size(0, 100)


@pytest.mark.parametrize("fps", [24.0, 30.0, 25.0, 12.0, 60.0, 23.976])
def test_resample_reference_frames_matches_reference(fps):
    """Constant-frame-rate resampling onto 24 fps, frame selection included."""
    frames = np.arange(50 * 2 * 2 * 3, dtype=np.uint8).reshape(50, 2, 2, 3)
    got = rp.resample_reference_frames(frames, fps)
    want = reference_packing.resample_reference_frames(frames, fps)
    assert got.shape == want.shape
    assert np.array_equal(got, want)
    if fps == 24.0:
        # The parity-exact route: the same array, not a copy.
        assert got is frames


def test_prepare_reference_frames_uses_its_own_aspect_canvas():
    """A reference video goes onto the canvas of ITS OWN aspect ratio, not the target's.

    The 1:1 source resolves to a 768x768 canvas that *differs* from the 768x1344 target,
    which is what makes this a gate; the 7-spec composition's videos resolve to canvases
    that happen to equal the target's.
    """
    frames = np.zeros((10, 480, 480, 3), dtype=np.uint8)
    prepared = rp.prepare_reference_frames(frames, TARGET_FRAMES)
    assert prepared.shape[1:3] == p.resolve_canvas_size(480, 480) == (768, 768)
    assert np.array_equal(prepared, reference_packing.prepare_reference_frames(frames, TARGET_FRAMES))

    # Already at its canvas: no resampling pass and no copy. `shares_memory`, not
    # `is` -- the frame-count cap slices first, so what comes back is a view of the
    # input rather than the input itself, and only a view keeps the pixels bit-exact.
    at_canvas = np.zeros((10, 768, 1344, 3), dtype=np.uint8)
    passthrough = rp.prepare_reference_frames(at_canvas, TARGET_FRAMES)
    assert np.shares_memory(passthrough, at_canvas)


@pytest.mark.parametrize("num_frames", [1, 5, 12, 24, 25, 124, 192])
def test_sample_reference_video_frames_matches_reference(num_frames):
    """2 fps sampling, pair merging and the round-half-to-even timestamps."""
    frames = np.zeros((num_frames, 4, 4, 3), dtype=np.uint8)
    got_frames, got_ts = rp.sample_reference_video_frames(frames)
    want_frames, want_ts = reference_packing.sample_reference_video_frames(frames)
    assert len(got_frames) == len(want_frames)
    assert got_ts == want_ts
    # One vision block per merged pair of sampled frames.
    assert len(got_ts) == -(-len(got_frames) // rp.MINIMAX_H3_QWEN_TEMPORAL_PATCH)


def test_block_timestamps_round_half_to_even():
    """ "{:.1f}" on the mean of a 2 fps pair gives "<0.2 seconds>", not "<0.3 seconds>".

    0.25 is exactly representable and Python rounds it to even, so this is a real
    formatting contract rather than a floating-point accident.
    """
    frames = np.zeros((124, 4, 4, 3), dtype=np.uint8)
    _, timestamps = rp.sample_reference_video_frames(frames)
    assert timestamps[0] == 0.25
    assert f"{timestamps[0]:.1f}" == "0.2"


@pytest.mark.parametrize("num_frames", [1, 5, 22, 39, 124, 125, 130])
def test_trim_reference_num_frames_matches_reference(num_frames):
    got = rp.trim_reference_num_frames(num_frames)
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
        assert rp.trim_reference_num_frames(num_frames) == 22
        assert rp.trim_reference_num_frames(num_frames) == reference_packing.trim_reference_num_frames(num_frames)
    # From 39 up it genuinely rounds down.
    assert rp.trim_reference_num_frames(39) == 39
    assert rp.trim_reference_num_frames(55) == 39
    assert rp.trim_reference_num_frames(125) == 124


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("sample_rate", [32000, 44100])
def test_prepare_reference_waveform_matches_reference(channels, sample_rate):
    """Stereo upmix, truncation at the native rate, then one resample pass."""
    torch.manual_seed(0)
    waveform = torch.randn(channels, int(6.0 * sample_rate))
    got = rp.prepare_reference_waveform(waveform, sample_rate, AUDIO_RATE, DURATION)
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
        got = rp.reference_media_to_uint8(media)
        want = reference_packing.reference_media_to_uint8(media)
        assert got.dtype == np.uint8
        assert np.array_equal(got, want)


# ------------------------------------------------------------------- the reference dataclass


def test_reference_requires_exactly_one_medium(expect_error):
    """A video may carry audio; nothing else may carry two media."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    waveform = torch.zeros(2, 800)

    assert rp.MiniMaxH3Reference(image=image).kind == "image"
    assert rp.MiniMaxH3Reference(video=video).kind == "video"
    assert rp.MiniMaxH3Reference(audio=waveform).kind == "audio"
    sounded = rp.MiniMaxH3Reference(video=video, audio=waveform)
    assert sounded.kind == "video" and sounded.has_audio

    with expect_error(ValueError, "exactly one"):
        rp.MiniMaxH3Reference(image=image, video=video)
    with expect_error(ValueError, "exactly one"):
        rp.MiniMaxH3Reference(image=image, audio=waveform)
    with expect_error(ValueError, "exactly one"):
        rp.MiniMaxH3Reference()


def test_reference_refuses_a_path(expect_error):
    """This module never opens media files, so a path is an error rather than a decode."""
    with expect_error(ValueError, "never opens media files"):
        rp.MiniMaxH3Reference(image="subject.png")


def test_reference_defaults_fps_to_the_models_own():
    """No fps means "already at 24", so the frames flow through untouched."""
    video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    assert rp.MiniMaxH3Reference(video=video).fps == float(p.MINIMAX_H3_FPS)


# ------------------------------------------------------- the typed condition blocks


def _prepared(kind, *, frames=1, height=48, width=84, audio_latents=0):
    return rp.MiniMaxH3PreparedReference(
        kind=kind,
        has_audio=audio_latents > 0,
        num_latent_frames=frames,
        latent_height=height if kind != "audio" else 0,
        latent_width=width if kind != "audio" else 0,
        num_audio_latents=audio_latents,
    )


def test_split_condition_blocks_is_in_packed_order():
    """The block list must interleave exactly as the packed sequence does.

    ``video_rows`` and ``audio_rows`` arrive as two per-modality concatenations, but
    the sequence puts a video reference's soundtrack immediately before its own video
    rows. A block list in per-modality order instead of packed order would keep every
    row count right and desynchronize the whole reference region.
    """
    references = [
        _prepared("video", frames=37, audio_latents=207),  # audio then video
        _prepared("audio", audio_latents=100),  # audio only
        _prepared("image"),  # video only
    ]
    video_rows = torch.arange(sum(r.num_video_rows for r in references if r.kind != "audio")).float()[:, None]
    audio_rows = torch.arange(sum(r.num_audio_rows for r in references)).float()[:, None] + 10000

    blocks = R.split_condition_blocks(references, video_rows, audio_rows)
    assert [kind for _, kind in blocks] == ["audio", "video", "audio", "video"]
    assert [block.shape[0] for block, _ in blocks] == [
        references[0].num_audio_rows,
        references[0].num_video_rows,
        references[1].num_audio_rows,
        references[2].num_video_rows,
    ]

    # Concatenating the blocks reproduces the reference region row for row, and each
    # block is the right slice of its own modality's tensor.
    assert torch.equal(blocks[0][0], audio_rows[: references[0].num_audio_rows])
    assert torch.equal(blocks[1][0], video_rows[: references[0].num_video_rows])
    assert torch.equal(blocks[2][0], audio_rows[references[0].num_audio_rows :])
    assert torch.equal(blocks[3][0], video_rows[references[0].num_video_rows :])


def test_split_condition_blocks_row_count_matches_the_layout():
    """The blocks must span exactly the layout's reference region, with nothing left over."""
    references = [
        _prepared("image", height=128, width=128),
        _prepared("video", frames=37, audio_latents=207),
        _prepared("audio", audio_latents=207),
    ]
    layout = rp.build_ref2va_packed_sequence(torch.ones(64, dtype=torch.long), references, 37, 48, 84, 207, (1, 2, 2))
    video_rows = torch.zeros(layout.num_condition_video_rows, 96)
    audio_rows = torch.zeros(layout.num_condition_audio_rows, 32)

    blocks = R.split_condition_blocks(references, video_rows, audio_rows)
    assert sum(block.shape[0] for block, _ in blocks) == (
        layout.num_condition_video_rows + layout.num_condition_audio_rows
    )


def test_split_condition_blocks_rejects_a_row_count_mismatch(expect_error):
    """A leftover means a reference was skipped, which would shift every row after it."""
    references = [_prepared("image")]
    with expect_error(ValueError, "consumed"):
        R.split_condition_blocks(references, torch.zeros(references[0].num_video_rows + 7, 96), None)


def test_reference_condition_shapes_skips_audio_references():
    """One noise draw per VISUAL reference; an audio reference draws none."""
    references = [_prepared("image"), _prepared("audio", audio_latents=50), _prepared("video", frames=37)]
    shapes = R.reference_condition_shapes(references)
    assert shapes == ((1, 48, 84), (37, 48, 84))


def test_normalize_reference_pixels_uses_imagenet_statistics():
    """Not [-1, 1]: H3 conditions on VLM-style normalized pixels."""
    frames = np.full((1, 32, 32, 3), 128, dtype=np.uint8)
    pixels = R.normalize_reference_pixels(frames)
    assert pixels.shape == (1, 3, 1, 32, 32)
    expected = [(128 / 255.0 - m) / s for m, s in zip(R.MINIMAX_H3_PIXEL_MEAN, R.MINIMAX_H3_PIXEL_STD)]
    for channel, value in enumerate(expected):
        assert torch.allclose(pixels[0, channel], torch.full((1, 32, 32), value), atol=1e-6)
    # A video is the same helper with T > 1, so the two paths cannot drift.
    assert R.normalize_reference_pixels(np.zeros((5, 8, 8, 3), dtype=np.uint8)).shape == (1, 3, 5, 8, 8)


# ------------------------------------------------------------------------- device
#
# The Phase 3 gate. Needs the mesh, the checkpoint and real media; see the module
# docstring on why `randn` will not do.

WEIGHTS_ENV = "MINIMAX_H3_DIFFUSERS_DIR"
MEDIA_ENV = "MINIMAX_H3_REFERENCE_MEDIA"
DEFAULT_MEDIA = Path.home() / "h3_fl2va_artifacts" / "fl2va_first.mp4"


def _real_media() -> Path:
    """A real video with a real soundtrack: a prior calibrated run of this very pipeline.

    1344x768 at 24 fps, 124 frames, already the canvas its own aspect ratio resolves
    to -- so it takes the parity-exact untouched route through
    ``prepare_reference_frames`` and the encode is compared on natural statistics.
    """
    path = Path(os.environ.get(MEDIA_ENV) or DEFAULT_MEDIA)
    if not path.is_file():
        pytest.skip(f"no reference media at {path}; set {MEDIA_ENV} to a video with a soundtrack")
    return path


def _weights_dir() -> Path:
    directory = Path(os.environ.get(WEIGHTS_ENV, ""))
    if not directory.is_dir():
        pytest.skip(f"set {WEIGHTS_ENV} to a diffusers snapshot of the checkpoint")
    return directory


# ---------------------------------------------------------------- the encode, on device
#
# Device encode against `MiniMaxH3Ref2VAReferenceEncoderStep`, on real media. `pcc=0.99`
# is the floor the encoder's thirteen `ttnn.group_norm` calls set -- none has an fp32 path
# -- and the same bar the fl2va keyframe encode holds to.
#
# The image runs at its full production resolution and the soundtrack at its full
# duration. The video runs at a reduced FRAME COUNT on the production canvas: the
# reference's video VAE encode runs on CPU, where 124 frames at 768x1344 is hours. What
# the video path adds is the entry point (`vae.encode`, with its 17-frames-per-5-latents
# chunking) and the frame trim, and 22 frames -> 7 latent frames exercises both.

# 16384, not the 65536 every other MiniMax-H3 gate uses. MEASURED, not chosen: the
# taps=3 video-reference encoder's static circular buffers clash with L1 by 4224 bytes at 65536 and
# still clash at 32768; 16384 is the first value that fits. `l1_small_size` reserves the TOP of L1,
# so a smaller reservation pushes those small allocations above the CB region rather than into it.
# `MINIMAX_H3_L1_SMALL` overrides it, which is how that was measured -- one config per process.
_L1_SMALL = int(os.environ.get("MINIMAX_H3_L1_SMALL", 16384))

# The same ring fabric params the e2e gates use, so the encode is measured in the configuration it
# will actually run in. A LINE config happens to work here -- the VAE encoders use no ring CCL -- but
# "it passed under a config production does not use" is exactly how two earlier L1-clash bugs got missed.
MESH_4X8 = [
    pytest.param(
        (4, 8),
        {**ring_params_req_exact_devices, "l1_small_size": _L1_SMALL},
        id="4x8",
    )
]

REFERENCE_PCC = 0.99
# 22 frames is `17 * 1 + 5`, the smallest count with more than one chunk's worth of
# temporal structure, and it maps to 7 latent frames.
VIDEO_FRAMES = 22


def _reference_components(weights: Path, device: str = "cpu"):
    """A stand-in for the reference pipeline, carrying only what its encoder step reads."""
    from diffusers.models import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(weights), subfolder="vae").to(device).eval()
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(weights), subfolder="audio_vae").to(device).eval()
    return SimpleNamespace(
        vae=vae,
        audio_vae=audio_vae,
        patch_size=(1, 2, 2),
        audio_latent_channels=audio_vae.config.latent_channels,
        _execution_device=torch.device(device),
    )


def _real_frame() -> np.ndarray:
    """One decoded frame of the real clip: natural pixel statistics, as an image reference.

    Not `randn`. The sampled latent is rounded through float16 before normalization,
    discarding about half the mantissa of every conditioning value, and how that lands is a
    property of natural image statistics -- a Gaussian exercises the code path and not the
    thing under test.
    """
    frames, _, _ = rp.decode_reference_video(_real_media())
    return frames[0]


def _reference_case(case: str):
    """`(ours, theirs)` prepared references for one modality's gate.

    Audio is paired with a small image because the reference forbids an audio-only request;
    the pairing is inert for the audio comparison, which slices its own rows out.
    """
    media = _real_media()
    frames, fps, soundtrack = rp.decode_reference_video(media)
    waveform, sample_rate = soundtrack

    if case == "image":
        specs = [dict(image=_real_frame())]
    elif case == "video":
        specs = [dict(video=frames[:VIDEO_FRAMES], fps=fps)]
    elif case == "audio":
        specs = [dict(image=_real_frame()), dict(audio=waveform, sample_rate=sample_rate)]
    else:
        raise ValueError(case)

    ours, _ = R.prepare_references([rp.MiniMaxH3Reference(**s) for s in specs], VIDEO_FRAMES, AUDIO_RATE)
    theirs, _ = reference_before_encoder.MiniMaxH3Ref2VASetupStep.prepare_references(
        _components(), [reference_packing.MiniMaxH3Reference(**s) for s in specs], VIDEO_FRAMES
    )
    return ours, theirs


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("case", ["image", "audio", "video"])
def test_encode_references_matches_reference(mesh_device, case, reset_seeds):
    """One modality's reference encode, against the reference implementation, on real media.

    Parametrized per modality rather than one request carrying all three, because the three
    take different code paths with different device requirements and a single case would
    report the first failure as a failure of all of them. `video` in particular needs the
    taps=3 encoder, whose L1 footprint is a separate question from the other two.

    Also asserts the resolved **geometry**, because the packed layout is built from it: a
    latent frame count or a latent height off by one produces a valid request conditioned on
    the wrong rows, and no PCC on the rows themselves would show it.
    """
    from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    weights = _weights_dir()
    ours, theirs = _reference_case(case)

    # ---- ours, on the mesh. Only the encoders this case needs. ----
    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=weights, task="ref2va")
    needs_video = any(reference.kind == "video" for reference in ours)
    vae = pipeline._prepare_vae()
    vae = pipeline._prepare_vae(encode_shape=(1, vae.tile_size, vae.tile_size), encode_taps=1)
    if needs_video:
        vae = pipeline._prepare_vae(
            encode_shape=(pipeline.vae_config.clip_length, vae.tile_size, vae.tile_size), encode_taps=3
        )
    audio_encoder = pipeline._prepare_audio_encoder() if any(r.has_audio for r in ours) else None

    got_video, got_audio = R.encode_references(
        ours,
        encode_clip=vae.encode_clip,
        encode_video=vae.encode if needs_video else None,
        encode_audio=(lambda waveform: audio_encoder(waveform)[0]) if audio_encoder else None,
        latents_mean=pipeline.vae_config.latents_mean,
        latents_std=pipeline.vae_config.latents_std,
        audio_latents_mean=pipeline.audio_config["latents_mean"],
        audio_latents_std=pipeline.audio_config["latents_std"],
        patch_size=pipeline.patch_size,
        audio_latent_channels=pipeline.audio_config["latent_channels"],
    )

    # ---- theirs, on CPU ----
    components = _reference_components(weights)
    with torch.no_grad():
        want_video, want_audio = reference_encoders.MiniMaxH3Ref2VAReferenceEncoderStep.encode_references(
            components, theirs, device=torch.device("cpu")
        )

    # Geometry first: the layout is built from it, so a mismatch here invalidates everything
    # downstream regardless of how the rows compare.
    for index, (a, b) in enumerate(zip(ours, theirs)):
        assert (a.num_latent_frames, a.latent_height, a.latent_width) == (
            b.num_latent_frames,
            b.latent_height,
            b.latent_width,
        ), f"reference {index} resolved to different visual latent geometry"
        assert a.num_audio_latents == b.num_audio_latents, f"reference {index} resolved a different audio length"
    logger.info(
        f"[{case}] resolved geometry: "
        + ", ".join(
            f"{r.kind}({r.num_latent_frames}x{r.latent_height}x{r.latent_width}, audio {r.num_audio_latents})"
            for r in ours
        )
    )

    assert got_video.shape == want_video.shape, f"{tuple(got_video.shape)} != {tuple(want_video.shape)}"
    logger.info(f"[{case}] video condition rows {tuple(got_video.shape)}")
    assert_quality(want_video, got_video, pcc=REFERENCE_PCC)

    if got_audio is not None:
        assert got_audio.shape == want_audio.shape, f"{tuple(got_audio.shape)} != {tuple(want_audio.shape)}"
        logger.info(f"[{case}] audio condition rows {tuple(got_audio.shape)}")
        # The soundtrack takes the posterior MEAN, so it carries no sampling noise at all.
        assert_quality(want_audio, got_audio, pcc=REFERENCE_PCC)
    else:
        assert want_audio is None, "the reference produced audio rows where we produced none"


# ------------------------------------------------------- the presentation, parity layer
#
# Bit-exact token ids AND tags against the reference's own builder, on the real tokenizer.


def _tokenizer():
    """The checkpoint's own tokenizer. It decides the token ids, so nothing else may."""
    weights = os.environ.get("MINIMAX_H3_DIFFUSERS_DIR") or os.environ.get("MINIMAX_H3_MODEL_PATH")
    if not weights or not (Path(weights) / "tokenizer").is_dir():
        pytest.skip("needs MINIMAX_H3_DIFFUSERS_DIR pointing at a diffusers snapshot with tokenizer/")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(weights), subfolder="tokenizer")


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
    got_ids, got_tags = rp.build_ref2va_presentation(tokenizer, prompt, ours, image_counts, video_counts)
    want_ids, want_tags = reference_packing.build_ref2va_presentation(
        tokenizer, prompt, theirs, image_counts, video_counts
    )
    assert got_ids == want_ids
    assert got_tags == want_tags
    assert len(got_tags) == len(got_ids)


# ------------------------------------------------------- the presentation, structure layer
#
# `_build_ref2va_presentation` needs only the tokenizer and the two processors, so it
# runs on host against a stub. Worth gating here rather than only at e2e: it is where
# the vision patches are put into PRESENTATION order, and the failure mode is one
# reference's tokens landing in another's rows -- which produces a perfectly valid
# request conditioned on the wrong thing.


class _ProcessorStub:
    """Just the three attributes `_build_ref2va_presentation` reads off a pipeline."""

    def __init__(self, weights: Path):
        from transformers import AutoImageProcessor, AutoTokenizer, AutoVideoProcessor

        self.tokenizer = AutoTokenizer.from_pretrained(str(weights), subfolder="tokenizer")
        self.image_processor = AutoImageProcessor.from_pretrained(str(weights), subfolder="text_encoder")
        self.video_processor = AutoVideoProcessor.from_pretrained(str(weights), subfolder="text_encoder")


def _build(stub, prompt, references):
    from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    return MiniMaxH3Pipeline._build_ref2va_presentation(stub, prompt, references)


def test_presentation_orders_vision_patches_by_reference_not_by_batch():
    """A VIDEO reference before an IMAGE one must yield the video's grid first.

    The two media go through different processors, so the natural batching is "all images,
    then all videos". But `_scatter_rows` consumes the tower's merged rows **in run
    order**, and the runs are in sequence order -- so batch order would put the image's
    tokens into the video's first block. This is the whole reason the presentation
    concatenates the patches itself instead of handing over two batches.
    """
    stub = _ProcessorStub(_weights_dir())
    video = rp.reference_from_video_file(_real_media(), with_audio=False)
    image = rp.MiniMaxH3Reference(image=_image(1024, 1024))

    prepared, _ = R.prepare_references([video, image], TARGET_FRAMES, AUDIO_RATE)
    input_ids, tags, type_ids, pixel_values, grid_thw, kinds = _build(stub, "a prompt", prepared)

    assert kinds == ["video", "image"], f"vision entries are in {kinds}, not presentation order"
    # The video's grid is (blocks, 48, 84) and the image's (1, 128, 128): distinguishable,
    # which is what makes this assertion meaningful.
    assert int(grid_thw[0][0]) == len(prepared[0].block_timestamps) > 1
    assert tuple(grid_thw[1].tolist()) == (1, 128, 128)
    # Patches concatenated in the same order, and every patch accounted for.
    assert pixel_values.shape[0] == sum(int(grid.prod()) for grid in grid_thw)

    # And the reversed request puts them the other way round -- so the ordering is really
    # read off the references rather than being a fixed video-first rule.
    reversed_prepared, _ = R.prepare_references([image, video], TARGET_FRAMES, AUDIO_RATE)
    _, _, _, _, reversed_grid, reversed_kinds = _build(stub, "a prompt", reversed_prepared)
    assert reversed_kinds == ["image", "video"]
    assert tuple(reversed_grid[0].tolist()) == (1, 128, 128)


def test_presentation_vision_runs_line_up_with_the_towers_rows():
    """The runs the scatter writes into must cover exactly the tokens the tower emits.

    One run per image and one per merged frame pair of a video, in sequence order, and
    their total length equal to the merged token count. A mismatch here is what the
    pipeline's own assertions catch at e2e; catching it on host is free.
    """
    from ....encoders.qwen3vl.model_qwen3vl import vision_token_runs

    stub = _ProcessorStub(_weights_dir())
    references = [
        rp.MiniMaxH3Reference(image=_image(1024, 1024)),
        rp.reference_from_video_file(_real_media(), with_audio=False),
        rp.MiniMaxH3Reference(audio=_waveform(DURATION)),
    ]
    prepared, _ = R.prepare_references(references, TARGET_FRAMES, AUDIO_RATE)
    input_ids, tags, type_ids, pixel_values, grid_thw, kinds = _build(stub, "a prompt", prepared)

    pad_ids = [stub.tokenizer.convert_tokens_to_ids(t) for t in ("<|image_pad|>", "<|video_pad|>")]
    runs = vision_token_runs(input_ids, pad_ids)
    merge = stub.image_processor.merge_size**2

    # One run per grid entry once a video's `t` is expanded: 1 for the image, 6 for the video.
    assert len(runs) == sum(int(grid[0]) for grid in grid_thw)
    # Total run length == the merged token count the tower will emit.
    assert sum(length for _, length in runs) == sum(int(grid.prod()) for grid in grid_thw) // merge
    # Runs are sorted and disjoint, which `_scatter_rows` requires outright.
    assert all(a[0] + a[1] <= b[0] for a, b in zip(runs, runs[1:]))
    # An audio reference contributes a label and no vision block at all.
    assert stub.tokenizer.decode(input_ids[0]).count("<Audio 1>") == 1


def test_presentation_token_type_ids_distinguish_image_from_video_pads():
    """Qwen3-VL's own tagging: 1 for image pads, 2 for video pads, 0 for everything else.

    Distinct from H3's `token_tags`, which marks the WHOLE vision block including its
    sentinels as video. Conflating the two mis-modulates AdaLN with no PCC signal, and
    collapsing video pads to 1 would put a video reference on the image rotary grid.
    """
    stub = _ProcessorStub(_weights_dir())
    references = [
        rp.MiniMaxH3Reference(image=_image(512, 512)),
        rp.reference_from_video_file(_real_media(), with_audio=False),
    ]
    prepared, _ = R.prepare_references(references, TARGET_FRAMES, AUDIO_RATE)
    input_ids, tags, type_ids, _, grid_thw, _ = _build(stub, "a prompt", prepared)

    image_pad = stub.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_pad = stub.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    assert (type_ids[input_ids == image_pad] == 1).all()
    assert (type_ids[input_ids == video_pad] == 2).all()
    assert (type_ids[(input_ids != image_pad) & (input_ids != video_pad)] == 0).all()
    assert int((type_ids == 2).sum()) > 0, "this request must contain video pads for the check to bite"

    # H3's own tags cover the sentinels too, so there are strictly more video-tagged rows
    # than there are vision pads.
    assert int((tags == p.MINIMAX_H3_VIDEO_TAG).sum()) > int((type_ids > 0).sum())
