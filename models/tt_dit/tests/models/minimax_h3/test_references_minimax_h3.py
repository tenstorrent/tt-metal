# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 ``ref2va`` reference conditioning, gated against the reference implementation's
own path (``diffusers.modular_pipelines.minimax_h3`` at PR #14355): host gates are bit-exact
(drift is silent), the device VAE encode is PCC on real media. importorskips that branch."""

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

# Base `packing` too: the rotary-clock gate pins both `_temporal_position_span` implementations.
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

TARGET_HEIGHT, TARGET_WIDTH = 768, 1344
VAE_RATIO = 16
AUDIO_HOP = 800
PATCH_SIZE = (1, 2, 2)


def _components():
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


def test_reference_limits_match_reference_and_are_enforced(expect_error):
    assert rp.MINIMAX_H3_MAX_REFERENCE_IMAGES == reference_packing.MINIMAX_H3_MAX_REFERENCE_IMAGES == 9
    assert rp.MINIMAX_H3_MAX_REFERENCE_VIDEOS == reference_packing.MINIMAX_H3_MAX_REFERENCE_VIDEOS == 3
    assert rp.MINIMAX_H3_MAX_REFERENCE_AUDIOS == reference_packing.MINIMAX_H3_MAX_REFERENCE_AUDIOS == 3
    assert rp.MINIMAX_H3_MAX_REFERENCES == reference_packing.MINIMAX_H3_MAX_REFERENCES == 12
    assert rp.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE == reference_packing.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE == 2048

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


@pytest.mark.parametrize("name", list(SPECS), ids=list(SPECS))
def test_prepare_references_matches_reference(name):
    spec = SPECS[name]
    # audio may not be alone; pair with an image
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
    references = [
        rp.MiniMaxH3Reference(image=_image(1024, 1024)),
        rp.MiniMaxH3Reference(video=_video(1920, 1080, 30), fps=24.0),
    ]
    prepared, _ = R.prepare_references(references, TARGET_FRAMES, AUDIO_RATE)

    assert prepared[0].image.size == (2048, 2048)
    assert min(prepared[0].image.size) == rp.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE
    assert prepared[1].frames.shape[1:3] == (768, 1344)
    assert min(prepared[1].frames.shape[1:3]) == p.MINIMAX_H3_SHORT_EDGE
    assert min(prepared[0].image.size) > min(prepared[1].frames.shape[1:3])


def test_prepare_references_truncates_a_video_and_its_soundtrack_to_the_target():
    long_frames = _video(1344, 768, TARGET_FRAMES + 60)
    references = [rp.MiniMaxH3Reference(video=long_frames, fps=24.0, audio=_waveform(10.0))]
    prepared, num_frames = R.prepare_references(references, TARGET_FRAMES, AUDIO_RATE)

    assert num_frames == TARGET_FRAMES
    assert prepared[0].frames.shape[0] == TARGET_FRAMES
    assert prepared[0].waveform.shape[1] == int(DURATION * AUDIO_RATE)


def test_num_frames_may_be_derived_from_a_single_audio_bearing_reference():
    references = [rp.MiniMaxH3Reference(image=_image(512, 512)), rp.MiniMaxH3Reference(audio=_waveform(6.0))]
    _, num_frames = R.prepare_references(references, None, AUDIO_RATE)
    assert num_frames == p.align_num_frames(round(6.0 * p.MINIMAX_H3_FPS))
    assert num_frames % p.MINIMAX_H3_FRAMES_PER_CHUNK == p.MINIMAX_H3_LATENTS_PER_CHUNK
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


@pytest.mark.parametrize("seconds", [DURATION, 5.0, 8.0, 15.0])
def test_pad_waveform_to_hop_reproduces_the_reference_latent_count(seconds):
    waveform = _waveform(seconds)
    padded = R.pad_waveform_to_hop(waveform)

    assert padded.shape[-1] % R.MINIMAX_H3_AUDIO_HOP == 0
    assert padded.shape[-1] - waveform.shape[-1] < R.MINIMAX_H3_AUDIO_HOP
    assert torch.equal(padded[:, : waveform.shape[-1]], waveform), "padding must not disturb the samples"
    assert (padded[:, waveform.shape[-1] :] == 0).all(), "the pad is zeros, as F.pad's default is"
    assert padded.shape[-1] // R.MINIMAX_H3_AUDIO_HOP == int(np.ceil(waveform.shape[-1] / R.MINIMAX_H3_AUDIO_HOP))


def test_production_soundtrack_is_not_a_whole_number_of_hops():
    """The case that makes the padding load-bearing rather than defensive."""
    samples = int(DURATION * AUDIO_RATE)
    assert samples == 165333
    assert samples % R.MINIMAX_H3_AUDIO_HOP != 0
    assert R.pad_waveform_to_hop(_waveform(DURATION)).shape[-1] // R.MINIMAX_H3_AUDIO_HOP == 207


def test_span_matches_both_reference_summation_orders():
    for n in range(1, 61):
        assert p._temporal_position_span(n) == reference_base_packing._temporal_position_span(
            n
        ), f"pairwise span disagrees with the reference at n={n}"
        assert rp._temporal_position_span(n) == reference_packing._temporal_position_span(
            n
        ), f"sequential span disagrees with the reference at n={n}"


def test_the_two_span_orders_actually_differ():
    assert p._temporal_position_span(15) == rp._temporal_position_span(15), "expected agreement below 16"
    assert p._temporal_position_span(16) != rp._temporal_position_span(16), "expected divergence from 16 latent frames"

    production = p.video_latent_num_frames(TARGET_FRAMES)
    assert production == 37
    pairwise, sequential = p._temporal_position_span(production), rp._temporal_position_span(production)
    assert pairwise != sequential, "the two orders must differ at the production frame count"
    assert rp._temporal_position_span(16) > p._temporal_position_span(16)
    assert sequential < pairwise


def _target():
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
    frames = np.zeros((num_frames, source_height, source_width, 3), dtype=np.uint8)
    frames = rp.prepare_reference_frames(frames, TARGET_FRAMES)
    trimmed = rp.trim_reference_num_frames(frames.shape[0])
    return dict(
        num_latent_frames=p.video_latent_num_frames(trimmed),
        latent_height=frames.shape[1] // VAE_RATIO,
        latent_width=frames.shape[2] // VAE_RATIO,
    )


def _audio_latents(seconds: float) -> int:
    return int(np.ceil(int(seconds * AUDIO_RATE) / AUDIO_HOP))


def _spec_to_prepared(spec, dataclass):
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
    ours = [_spec_to_prepared(spec, rp.MiniMaxH3PreparedReference) for spec in spec_list]
    theirs = [_spec_to_prepared(spec, reference_packing.MiniMaxH3PreparedReference) for spec in spec_list]
    return ours, theirs


IMAGE_1TO1 = dict(kind="image", source=(1024, 1024))
IMAGE_16TO9 = dict(kind="image", source=(1920, 1080))
VIDEO_SOUND = dict(kind="video", source=(1344, 768), audio_seconds=DURATION)
# 1:1 sounded video: its soundtrack rows must pin to its OWN width grid, not the target's.
VIDEO_SOUND_1TO1 = dict(kind="video", source=(768, 768), audio_seconds=DURATION)
VIDEO_SILENT = dict(kind="video", source=(1344, 768))
AUDIO_ONLY = dict(kind="audio", audio_seconds=DURATION)

# Awkward order on purpose: exercises the label counters, soundtrack-before-video order and the
# shared rotary clock; the reversed copy is the host order-sensitivity gate (order is semantic).
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


# `(width, height)` in, `(height, width)` out; both orientations listed so a transpose cannot pass.
@pytest.mark.parametrize(
    "source,expected",
    [
        ((1024, 1024), (2048, 2048)),
        ((1920, 1080), (2048, 3648)),
        ((1080, 1920), (3648, 2048)),
        ((4096, 1024), (2048, 8192)),
    ],
)
def test_resolve_reference_image_size(source, expected):
    assert rp.resolve_reference_image_size(*source) == expected
    assert rp.resolve_reference_image_size(*source) == reference_packing.resolve_reference_image_size(*source)
    height, width = expected
    assert height % p.MINIMAX_H3_CANVAS_MULTIPLE == 0 and width % p.MINIMAX_H3_CANVAS_MULTIPLE == 0
    assert min(height, width) == rp.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE


def test_resolve_reference_image_size_rejects_out_of_range(expect_error):
    with expect_error(ValueError, "within 1:4 and 4:1"):
        rp.resolve_reference_image_size(5000, 1000)
    with expect_error(ValueError, "positive size"):
        rp.resolve_reference_image_size(0, 100)


@pytest.mark.parametrize("fps", [24.0, 30.0, 25.0, 12.0, 60.0, 23.976])
def test_resample_reference_frames_matches_reference(fps):
    frames = np.arange(50 * 2 * 2 * 3, dtype=np.uint8).reshape(50, 2, 2, 3)
    got = rp.resample_reference_frames(frames, fps)
    want = reference_packing.resample_reference_frames(frames, fps)
    assert got.shape == want.shape
    assert np.array_equal(got, want)
    if fps == 24.0:
        assert got is frames


def test_prepare_reference_frames_uses_its_own_aspect_canvas():
    frames = np.zeros((10, 480, 480, 3), dtype=np.uint8)
    prepared = rp.prepare_reference_frames(frames, TARGET_FRAMES)
    assert prepared.shape[1:3] == p.resolve_canvas_size(480, 480) == (768, 768)
    assert np.array_equal(prepared, reference_packing.prepare_reference_frames(frames, TARGET_FRAMES))

    # `shares_memory`, not `is`: the frame-count cap slices first, so a view comes back.
    at_canvas = np.zeros((10, 768, 1344, 3), dtype=np.uint8)
    passthrough = rp.prepare_reference_frames(at_canvas, TARGET_FRAMES)
    assert np.shares_memory(passthrough, at_canvas)


@pytest.mark.parametrize("num_frames", [1, 5, 12, 24, 25, 124, 192])
def test_sample_reference_video_frames_matches_reference(num_frames):
    frames = np.zeros((num_frames, 4, 4, 3), dtype=np.uint8)
    got_frames, got_ts = rp.sample_reference_video_frames(frames)
    want_frames, want_ts = reference_packing.sample_reference_video_frames(frames)
    assert len(got_frames) == len(want_frames)
    assert got_ts == want_ts
    assert len(got_ts) == -(-len(got_frames) // rp.MINIMAX_H3_QWEN_TEMPORAL_PATCH)


def test_block_timestamps_round_half_to_even():
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
    """1..23 frames all map to 22 -- not a plain round-down; a "fixed" floor of 5 silently changes the latent count."""
    for num_frames in (1, 5, 10, 22, 23):
        assert rp.trim_reference_num_frames(num_frames) == 22
        assert rp.trim_reference_num_frames(num_frames) == reference_packing.trim_reference_num_frames(num_frames)
    assert rp.trim_reference_num_frames(39) == 39
    assert rp.trim_reference_num_frames(55) == 39
    assert rp.trim_reference_num_frames(125) == 124


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("sample_rate", [32000, 44100])
def test_prepare_reference_waveform_matches_reference(channels, sample_rate):
    torch.manual_seed(0)
    waveform = torch.randn(channels, int(6.0 * sample_rate))
    got = rp.prepare_reference_waveform(waveform, sample_rate, AUDIO_RATE, DURATION)
    want = reference_packing.prepare_reference_waveform(waveform, sample_rate, AUDIO_RATE, DURATION)
    assert got.shape == want.shape
    assert torch.equal(got, want)
    assert got.shape[0] == p.MINIMAX_H3_AUDIO_CHANNELS
    if sample_rate == AUDIO_RATE:
        assert got.shape[1] == int(DURATION * sample_rate)


def test_reference_media_to_uint8_matches_reference():
    torch.manual_seed(0)
    tensor = torch.rand(3, 8, 6)
    array = (np.random.default_rng(0).random((8, 6, 3)) * 255).astype(np.uint8)
    for media in (tensor, array, [tensor, tensor]):
        got = rp.reference_media_to_uint8(media)
        want = reference_packing.reference_media_to_uint8(media)
        assert got.dtype == np.uint8
        assert np.array_equal(got, want)


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
    with expect_error(ValueError, "never opens media files"):
        rp.MiniMaxH3Reference(image="subject.png")


def test_reference_defaults_fps_to_the_models_own():
    video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    assert rp.MiniMaxH3Reference(video=video).fps == float(p.MINIMAX_H3_FPS)


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
    """Per-modality order would keep every row count right and desynchronize the whole region."""
    references = [
        _prepared("video", frames=37, audio_latents=207),
        _prepared("audio", audio_latents=100),
        _prepared("image"),
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

    assert torch.equal(blocks[0][0], audio_rows[: references[0].num_audio_rows])
    assert torch.equal(blocks[1][0], video_rows[: references[0].num_video_rows])
    assert torch.equal(blocks[2][0], audio_rows[references[0].num_audio_rows :])
    assert torch.equal(blocks[3][0], video_rows[references[0].num_video_rows :])


def test_split_condition_blocks_row_count_matches_the_layout():
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
    references = [_prepared("image")]
    with expect_error(ValueError, "consumed"):
        R.split_condition_blocks(references, torch.zeros(references[0].num_video_rows + 7, 96), None)


def test_reference_condition_shapes_skips_audio_references():
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
    assert R.normalize_reference_pixels(np.zeros((5, 8, 8, 3), dtype=np.uint8)).shape == (1, 3, 5, 8, 8)


REFERENCE_MEDIA = Path.home() / "h3_fl2va_artifacts" / "fl2va_first.mp4"


def _real_media() -> Path:
    if not REFERENCE_MEDIA.is_file():
        pytest.skip(f"no reference media at {REFERENCE_MEDIA}; place a video with a soundtrack there")
    return REFERENCE_MEDIA


def _weights_dir() -> Path:
    directory = Path(os.environ.get("MINIMAX_H3_MODEL_PATH", ""))
    if not directory.is_dir():
        pytest.skip("set MINIMAX_H3_MODEL_PATH to a diffusers snapshot of the checkpoint")
    return directory


_L1_SMALL = 16384  # 65536 and 32768 clash with the taps=3 encoder CBs

# The e2e ring fabric params: a LINE config happens to pass here but production never runs it.
MESH_4X8 = [
    pytest.param(
        (4, 8),
        {**ring_params_req_exact_devices, "l1_small_size": _L1_SMALL},
        id="4x8",
    )
]

REFERENCE_PCC = 0.99  # floor set by the encoder's ttnn.group_norm calls (no fp32 path)
# 17*1 + 5: smallest multi-chunk count (7 latent frames); the reference video VAE encode is CPU-hours at 124.
VIDEO_FRAMES = 22


def _reference_components(weights: Path, device: str = "cpu"):
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
    """Not randn: the fp16 round trip is only exercised on natural image statistics."""
    frames, _, _ = rp.decode_reference_video(_real_media())
    return frames[0]


def _reference_case(case: str):
    """Audio pairs with an image (audio-only is forbidden); the pairing is inert for the comparison."""
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
    """Per modality (the three encode paths fail independently); geometry asserted too, since the layout is built from it."""
    from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    weights = _weights_dir()
    ours, theirs = _reference_case(case)

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

    components = _reference_components(weights)
    with torch.no_grad():
        want_video, want_audio = reference_encoders.MiniMaxH3Ref2VAReferenceEncoderStep.encode_references(
            components, theirs, device=torch.device("cpu")
        )

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


def _tokenizer():
    weights = os.environ.get("MINIMAX_H3_MODEL_PATH")
    if not weights or not (Path(weights) / "tokenizer").is_dir():
        pytest.skip("needs MINIMAX_H3_MODEL_PATH pointing at a diffusers snapshot with tokenizer/")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(weights), subfolder="tokenizer")


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_presentation_matches_reference(case):
    """The tags are the load-bearing half: H3 tags the WHOLE vision block video, sentinels included."""
    tokenizer = _tokenizer()
    ours, theirs = _both(CASES[case])
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


class _ProcessorStub:
    def __init__(self, weights: Path):
        from transformers import AutoImageProcessor, AutoTokenizer, AutoVideoProcessor

        self.tokenizer = AutoTokenizer.from_pretrained(str(weights), subfolder="tokenizer")
        self.image_processor = AutoImageProcessor.from_pretrained(str(weights), subfolder="text_encoder")
        self.video_processor = AutoVideoProcessor.from_pretrained(str(weights), subfolder="text_encoder")


def _build(stub, prompt, references):
    from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    return MiniMaxH3Pipeline._build_ref2va_presentation(stub, prompt, references)


def test_presentation_orders_vision_patches_by_reference_not_by_batch():
    """`_scatter_rows` consumes tower rows in run order, so processor-batch order would swap blocks."""
    stub = _ProcessorStub(_weights_dir())
    video = rp.reference_from_video_file(_real_media(), with_audio=False)
    image = rp.MiniMaxH3Reference(image=_image(1024, 1024))

    prepared, _ = R.prepare_references([video, image], TARGET_FRAMES, AUDIO_RATE)
    input_ids, tags, type_ids, pixel_values, grid_thw, kinds = _build(stub, "a prompt", prepared)

    assert kinds == ["video", "image"], f"vision entries are in {kinds}, not presentation order"
    assert int(grid_thw[0][0]) == len(prepared[0].block_timestamps) > 1
    assert tuple(grid_thw[1].tolist()) == (1, 128, 128)
    assert pixel_values.shape[0] == sum(int(grid.prod()) for grid in grid_thw)

    reversed_prepared, _ = R.prepare_references([image, video], TARGET_FRAMES, AUDIO_RATE)
    _, _, _, _, reversed_grid, reversed_kinds = _build(stub, "a prompt", reversed_prepared)
    assert reversed_kinds == ["image", "video"]
    assert tuple(reversed_grid[0].tolist()) == (1, 128, 128)


def test_presentation_vision_runs_line_up_with_the_towers_rows():
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

    assert len(runs) == sum(int(grid[0]) for grid in grid_thw)
    assert sum(length for _, length in runs) == sum(int(grid.prod()) for grid in grid_thw) // merge
    # Runs are sorted and disjoint, which `_scatter_rows` requires outright.
    assert all(a[0] + a[1] <= b[0] for a, b in zip(runs, runs[1:]))
    assert stub.tokenizer.decode(input_ids[0]).count("<Audio 1>") == 1


def test_presentation_token_type_ids_distinguish_image_from_video_pads():
    """Distinct from H3's token_tags (whole block video): conflating the two mis-modulates AdaLN silently."""
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
    assert int((tags == p.MINIMAX_H3_VIDEO_TAG).sum()) > int((type_ids > 0).sum())
