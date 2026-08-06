# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 ``ref2va`` reference conditioning: preparation and encode.

Two halves, split by what they need:

* **Host** -- request validation, per-reference preparation at each reference's own
  resolution, the audio hop padding, and the typed condition-block split. All gated
  against the reference implementation's own
  ``MiniMaxH3Ref2VASetupStep.prepare_references``, and all free.
* **Device** (``test_encode_references_matches_reference``) -- the real VAE encode at
  production resolutions against ``MiniMaxH3Ref2VAReferenceEncoderStep``, on **real
  media**. ``randn`` would not exercise the fp16 round trip on natural statistics,
  which is the thing under test.

The three encode recipes differ per modality and none of them is guessable from the
others: image and video posteriors are *sampled* under a generator seeded 42 and
rounded through float16, while a soundtrack takes the posterior *mean* untouched.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn

from ....pipelines.minimax_h3 import packing as p
from ....pipelines.minimax_h3 import packing_ref2va as rp
from ....pipelines.minimax_h3 import references as R
from ....utils.check import assert_quality

reference_before_encoder = pytest.importorskip(
    "diffusers.modular_pipelines.minimax_h3.before_encoder",
    reason="requires the minimax-h3 diffusers branch",
)
reference_packing = pytest.importorskip(
    "diffusers.modular_pipelines.minimax_h3.packing_ref2va",
    reason="requires the minimax-h3 diffusers branch",
)
reference_encoders = pytest.importorskip(
    "diffusers.modular_pipelines.minimax_h3.encoders",
    reason="requires the minimax-h3 diffusers branch",
)

AUDIO_RATE = 32000
TARGET_FRAMES = 124
DURATION = TARGET_FRAMES / p.MINIMAX_H3_FPS


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


def _pair(spec: dict):
    """One spec into ``(our reference, the reference implementation's reference)``."""
    ours = rp.MiniMaxH3Reference(**spec)
    theirs = reference_packing.MiniMaxH3Reference(**spec)
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


# --------------------------------------------------------------------------- host


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
    ours = [_pair(s)[0] for s in specs]
    theirs = [_pair(s)[1] for s in specs]

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

    Two sizing rules feed the same request, 2048 px against 768 px on the short edge,
    and using one where the other belongs is the single easiest way to condition at
    the wrong scale. It would not fail a shape check: both produce a valid request.
    """
    references = [
        rp.MiniMaxH3Reference(image=_image(1024, 1024)),
        rp.MiniMaxH3Reference(video=_video(1920, 1080, 30), fps=24.0),
    ]
    prepared, _ = R.prepare_references(references, TARGET_FRAMES, AUDIO_RATE)

    # 2048 px short edge, no area cap.
    assert prepared[0].image.size == (2048, 2048)
    assert min(prepared[0].image.size) == rp.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE
    # The 768 px canvas of the VIDEO's own 16:9, which happens to equal the target
    # here -- and 768x1344 rather than 2048-anything, which is the point.
    assert prepared[1].frames.shape[1:3] == (768, 1344)
    assert min(prepared[1].frames.shape[1:3]) == p.MINIMAX_H3_SHORT_EDGE
    # 2048 against 768: an image reference is encoded at 2.67x a video reference's
    # short edge, which is why one image costs 4096 rows where one video frame costs
    # 1008. (The 4x the vision-tower test quotes is at the aspect extremes, where the
    # canvas area cap pushes the short edge down to 32 patches.)
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


def test_request_limits(expect_error):
    """The documented per-request ceilings, and the "audio not alone" rule."""
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
# The Phase 3 gate: our device encode against `MiniMaxH3Ref2VAReferenceEncoderStep`, on
# real media. `pcc=0.99` is the floor the encoder's thirteen `ttnn.group_norm` calls set
# -- none has an fp32 path -- and it is the same bar the fl2va keyframe encode holds to.
#
# On lengths: the IMAGE case runs at its full production 2048x2048, and the soundtrack at
# its full production duration. The VIDEO case runs at a reduced FRAME COUNT on the
# production canvas, because the reference's video VAE encode runs on CPU and 124 frames
# at 768x1344 is hours there. What the video path adds over the image path is the entry
# point (`vae.encode`, with its 17-frames-per-5-latents chunking) and the frame trim, and
# both are exercised by 22 frames -> 7 latent frames just as well as by 124 -> 37. The
# production frame count is covered end to end by the e2e gate. Same reasoning as am. 122.

# 16384, not the 65536 every other MiniMax-H3 gate uses. MEASURED, not chosen (am. 124/126): the
# taps=3 video-reference encoder's static circular buffers clash with L1 by 4224 bytes at 65536 and
# still clash at 32768; 16384 is the first value that fits. `l1_small_size` reserves the TOP of L1,
# so a smaller reservation pushes those small allocations above the CB region rather than into it.
# `MINIMAX_H3_L1_SMALL` overrides it, which is how that was measured -- one config per process.
_L1_SMALL = int(os.environ.get("MINIMAX_H3_L1_SMALL", 16384))

MESH_4X8 = [
    pytest.param(
        (4, 8),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": _L1_SMALL, "trace_region_size": 200000000},
        id="blackhole-4x8",
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
    taps=3 encoder, whose L1 footprint is a separate question from the other two
    (campaign am. 124).

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


# ------------------------------------------------------- the presentation, on host
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
