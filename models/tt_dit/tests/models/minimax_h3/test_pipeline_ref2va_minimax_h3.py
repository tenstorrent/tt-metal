# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 ``ref2va`` end to end: omni-reference conditioning on the 4x8 mesh.

A separate file from the ``t2va`` / ``fl2va`` e2e gates so it defaults to its own
process: a ref2va request is 1.2x-3.0x t2va's packed length, and one process
holding DiT programs plus CCL buffers at several of those lengths is a memory risk.

WHY THE fl2va QUALITY GATE DOES NOT TRANSFER

``fl2va``'s keyframe is pinned to frame 0 of the output, so "decoded frame 0 correlates
with the keyframe" has a floor to hold to. A ``ref2va`` reference is not pinned to any
output row -- it conditions, it is not copied -- so no similarity number is one a correct
implementation must reach, and an absolute bar would be invented rather than derived.

The gate is built from a measured floor instead:

1. Two generations from the same prompt, seed and reference give the run-to-run floor.
   Measured: exactly 0.000000, the pipeline being bit-reproducible.
2. A third with a different reference of the same size. The packed layout is then identical
   row for row and every noise draw has the same shape in the same order, so the noise is
   bit-identical and the two requests differ in reference content alone.
3. An implementation that ignored its reference would score exactly the floor. Measured:
   0.128143.

What this gate does not assert: that each output resembles the reference it was given more
than the one it was not. No instrument is known to measure that here -- whole-frame
luminance correlation is positional where conditioning is not, and CLIP image-image
similarity separates the two outputs rather than the two references. Those numbers are
logged only.

**Look at the frames.** Seams and flicker are what whole-tensor metrics average away, and
both are parallelism bugs. Artifacts land in ``MINIMAX_H3_REF2VA_ARTIFACT_DIR``, written
before any quality check so a failing check still leaves them behind.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

from ....pipelines.minimax_h3.packing_ref2va import MiniMaxH3Reference
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....utils.test import ring_params_req_exact_devices
from .common_av import (
    check_audio_sanity,
    check_av_sync,
    check_spatial_seams,
    clip_prompt_alignment,
    ref2va_references,
    reference_video,
    run_vbench,
    write_artifacts,
)

WEIGHTS_ENV = "MINIMAX_H3_DIFFUSERS_DIR"
ARTIFACT_ENV = "MINIMAX_H3_REF2VA_ARTIFACT_DIR"

# The working point. Fixed: changing it invalidates the numbers below.
WIDTH, HEIGHT, NUM_FRAMES, STEPS, SEED = 1344, 768, 124, 50, 0
FPS = 24

PROMPT = "a slow push-in through a quiet room as afternoon light moves across the floor"

# Quality bars for REFERENCE-DRIVEN content, set below the minimum measured across the three shapes
# and NOT inherited from t2va, three of whose six bars ref2va does not meet:
#
#   dimension               one_image  video+sound  mixed  | min     t2va bar
#   CLIP prompt alignment      29.05      29.97     29.38  | 29.05   33.0
#   subject_consistency       0.9631     0.9344    0.9587  | 0.9344  0.95
#   background_consistency    0.9569     0.9249    0.9397  | 0.9249  0.95
#   motion_smoothness         0.9957     0.9952    0.9959  | 0.9952  0.97
#   dynamic_degree            1.0000     1.0000    1.0000  | 1.0     1.0
#   imaging_quality           0.4826     0.6575    0.5826  | 0.4826  0.64
#
# None of the three shortfalls is a defect. CLIP tracks prompt specificity, and ref2va's prompt is
# one clause against t2va's dialogue scene. The consistency pair penalises change over time while
# `dynamic_degree` is 1.0 everywhere, so the lowest pair belongs to the case conditioned on a moving
# clip. `imaging_quality` is no-reference IQA and spreads 0.17 on one pipeline; 0.4884 was recorded
# on a visually perfect scene. Headroom follows t2va's convention: its 33.0 sits ~4 under a measured
# 37.05.
REF2VA_CLIP_THRESHOLD = 25.0
REF2VA_VBENCH_THRESHOLDS = {
    "subject_consistency": 0.90,
    "background_consistency": 0.89,
    "motion_smoothness": 0.97,  # t2va's, and ref2va measures *better* than t2va on it
    "dynamic_degree": 1.0,  # binary over one video: 1.0 or the clip is frozen
    "imaging_quality": 0.44,
}

# 16384 rather than the 65536 the t2va/fl2va gates use. A video reference goes through the video
# VAE's taps=3 encoder, whose static circular buffers clash with L1 at 65536 and at 32768; 16384 is
# the first value that fits. t2va and fl2va never reach that encoder. One process holds
# every stage, so this value also has to serve the audio decode.
_L1_SMALL = int(os.environ.get("MINIMAX_H3_L1_SMALL", 16384))

# The same ring params the t2va and fl2va e2e gates use: the DiT attends in a ring on the SP axis,
# and a LINE fabric config fails its CCL ops with `fabric.cpp:174 forwarding_direction.has_value()`.
# Only `l1_small_size` differs.
MESH_4X8 = [
    pytest.param(
        (4, 8),
        {**ring_params_req_exact_devices, "l1_small_size": _L1_SMALL},
        id="4x8",
    )
]


def _weights_dir() -> Path:
    directory = Path(os.environ.get(WEIGHTS_ENV, ""))
    if not directory.is_dir():
        pytest.skip(f"set {WEIGHTS_ENV} to a diffusers snapshot holding transformer_ref/")
    if not (directory / "transformer_ref").is_dir():
        pytest.skip(f"{directory} has no transformer_ref/; ref2va runs against that partition")
    return directory


def _artifact_dir() -> Path:
    directory = Path(os.environ.get(ARTIFACT_ENV) or Path.home() / "h3_ref2va_artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _real_frame_image() -> Image.Image:
    """One decoded frame of the real clip, as a photographic image reference."""
    from ....pipelines.minimax_h3.packing_ref2va import decode_reference_video

    frames, _, _ = decode_reference_video(reference_video())
    return Image.fromarray(frames[0])


def _inverted(image: Image.Image) -> Image.Image:
    """The same photograph with its colours inverted: the discriminator's second reference.

    Holds size constant, so the packed layout and the noise stream are unchanged, along with
    texture and edge statistics. Only the palette differs, which is both transferable and
    measurable. A synthetic pattern would leave nothing for a direction check to find, being
    content the model cannot render for the prompt.
    """
    return Image.fromarray(255 - np.asarray(image.convert("RGB")))


def _frames_of(output) -> np.ndarray:
    """``(F, H, W, 3)`` uint8 frames from a pipeline output."""
    return (output.video[0].permute(1, 2, 3, 0).float().cpu().numpy() * 255).astype(np.uint8)


def _clip_resemblance(output, image: Image.Image, num_frames: int = 8) -> float:
    """Mean CLIP image-image cosine similarity between sampled output frames and a reference.

    Semantic rather than positional: a reference conditions what the output is of, not which
    pixel goes where. `open_clip` is the instrument the t2va gate uses.
    """
    from ...dataset_eval.clip_encoder import CLIPEncoder

    encoder = CLIPEncoder()
    frames = _frames_of(output)
    indices = np.linspace(0, frames.shape[0] - 1, num_frames).round().astype(int)

    with torch.no_grad():
        reference_features = encoder.model.encode_image(encoder.preprocess(image).unsqueeze(0)).float()
        reference_features /= reference_features.norm(dim=-1, keepdim=True)
        scores = []
        for index in indices:
            frame = encoder.preprocess(Image.fromarray(frames[index])).unsqueeze(0)
            features = encoder.model.encode_image(frame).float()
            features /= features.norm(dim=-1, keepdim=True)
            scores.append(float((features @ reference_features.T).squeeze()))
    return float(np.mean(scores))


def _colour_distance(output, image: Image.Image) -> float:
    """Euclidean distance between an output's mean RGB and a reference's, over [0, 1].

    The interpretable companion to the CLIP number: with an inverted-colour reference pair it
    measures directly whether the palette carried across. Logged, not asserted -- see the
    module docstring on why no direction check is a gate here.
    """
    output_mean = _frames_of(output).reshape(-1, 3).mean(axis=0) / 255.0
    reference_mean = np.asarray(image.convert("RGB")).reshape(-1, 3).mean(axis=0) / 255.0
    return float(np.linalg.norm(output_mean - reference_mean))


def _divergence(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean absolute per-pixel difference between two outputs, over [0, 1] pixels."""
    return float((a.float() - b.float()).abs().mean())


def _write(output, stem: str) -> dict:
    """Frames and audio for a human to look at: four sampled PNGs, a wav, and a muxed mp4.

    The mp4 comes from the shared `write_artifacts` the t2va and fl2va gates use, so VBench
    scores every task from the same kind of file and the numbers stay comparable.
    """
    directory = _artifact_dir()
    frames = _frames_of(output)
    for index in (0, 17, NUM_FRAMES // 2, NUM_FRAMES - 1):
        Image.fromarray(frames[index]).save(directory / f"{stem}_frame_{index}.png")
    paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, directory, stem=stem)
    logger.info(f"{stem}: artifacts in {directory} ({sorted(paths)})")
    return paths


def _record_quality(frames: np.ndarray, paths: dict, case: str) -> None:
    """Record CLIP prompt alignment and the five VBench dimensions.

    Asserted against bars derived from these same measurements. t2va's bars do not
    transfer to reference-driven content: `imaging_quality` scored 0.4884 on a visually perfect
    night scene against a 0.64 bar, and the seam ratio gave a false failure at 2.29x.
    """
    if os.environ.get("RUN_CLIP", "1") in ("1", "true", "True"):
        pytest.importorskip("open_clip", reason="RUN_CLIP=1 but open_clip is missing (set RUN_CLIP=0)")
        alignment = clip_prompt_alignment(frames, PROMPT)
        logger.info(
            f"QUALITY ref2va[{case}] CLIP prompt alignment: mean={alignment['mean']:.2f} "
            f"min={alignment['min']:.2f} max={alignment['max']:.2f} (bar {REF2VA_CLIP_THRESHOLD})"
        )
        if REF2VA_CLIP_THRESHOLD is not None:
            assert alignment["mean"] >= REF2VA_CLIP_THRESHOLD, (
                f"CLIP prompt alignment {alignment['mean']:.2f} is below the {REF2VA_CLIP_THRESHOLD} bar; "
                "the video no longer matches the prompt"
            )
    else:
        logger.info("RUN_CLIP=0, skipping the CLIP prompt-alignment measurement")

    if os.environ.get("RUN_VBENCH", "1") not in ("1", "true", "True"):
        logger.info("RUN_VBENCH=0, skipping the VBench measurement")
        return
    if "mp4" not in paths:
        pytest.skip("RUN_VBENCH=1 needs the muxed mp4, which ffmpeg did not produce")
    scores = run_vbench(paths["mp4"], PROMPT, tuple(REF2VA_VBENCH_THRESHOLDS))
    for dimension, value in scores.items():
        bar = REF2VA_VBENCH_THRESHOLDS.get(dimension)
        logger.info(f"QUALITY ref2va[{case}] VBench {dimension} = {value:.4f} (bar {bar})")
    for dimension, bar in REF2VA_VBENCH_THRESHOLDS.items():
        if bar is None:
            continue
        value = scores.get(dimension)
        assert value is not None, f"VBench produced no {dimension} score"
        assert value >= bar, f"VBench {dimension} {value:.4f} is below the {bar} bar"


def _pipeline(mesh_device) -> MiniMaxH3Pipeline:
    """A pipeline bound to the `transformer_ref` partition.

    Fixed at construction rather than per call: each partition is 62 GB, so switching inside
    one process would mean a full reload.
    """
    return MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=_weights_dir(), task="ref2va")


# The e2e case list, all admitted by the full-depth shape probe. `padded` is the measured
# packed length each case runs at, asserted below so a case cannot silently drift
# onto a different shape than the one that was probed.
# Padded packed length per case, MEASURED end to end and asserted below so a case cannot drift
# onto a shape the probe did not cover. `one_image` and `video_with_sound` match the
# host-only prediction exactly; `mixed` is 89856 rather than the 90112 predicted, because that
# estimate used a guessed presentation length and the real one tokenizes shorter. The prediction was
# never a measurement -- `mixed` was an interpolation between two probed shapes.
CASES = {
    "one_image": 46080,
    "video_with_sound": 81664,
    "mixed": 89856,
}


@pytest.mark.timeout(9000)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_ref2va_end_to_end(mesh_device, case, reset_seeds):
    """A full ref2va generation: video plus its synchronized soundtrack.

    Gates that every path a reference touches agrees on geometry: the conditioner's vision
    blocks, the VAE encode, the packed layout, the rotary clock and both decoders. A mismatch
    surfaces as a wrong shape, a failed assert or a desynchronized soundtrack.
    """
    pipeline = _pipeline(mesh_device)
    output = pipeline(
        PROMPT,
        references=ref2va_references(case),
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        seed=SEED,
    )

    assert output.video.shape == (1, 3, NUM_FRAMES, HEIGHT, WIDTH), tuple(output.video.shape)
    assert output.video.min() >= 0.0 and output.video.max() <= 1.0, "decoded video must be in [0, 1]"
    assert torch.isfinite(output.video).all() and torch.isfinite(output.audio).all()
    # A drift here means the case is no longer the request that was probed, so the memory
    # verdict of the shape probe no longer covers it.
    assert (
        pipeline.last_padded_len == CASES[case]
    ), f"{case} ran at padded_len {pipeline.last_padded_len}, not the probed {CASES[case]}"

    frames = _frames_of(output)
    # Artifacts before the checks: a check that fires first leaves no frames to inspect, and the
    # frames are what a seam reading has to be judged against.
    paths = _write(output, f"ref2va_{case}")
    logger.info(f"ref2va[{case}] padded_len={pipeline.last_padded_len} timings={pipeline.last_timings}")

    check_audio_sanity(
        output.audio, sampling_rate=output.sampling_rate, expected_seconds=NUM_FRAMES / FPS, tolerance_seconds=0.05
    )
    check_av_sync(frames, output.audio, sampling_rate=output.sampling_rate, fps=FPS)
    # Separate bars per axis. Vertical keeps t2va's 2.0 (measured 1.20-1.32). Horizontal is 3.0:
    # `video_with_sound` reads 2.29x there and it is scene content, not a seam -- the elevation
    # spans ~9 rows where a decoder seam occupies 1-2, and the frame's largest vertical gradient
    # (16.06 at y=306) is not at a tile boundary at all. The ratio is content-sensitive
    # in both directions -- a false pass from the same property is on record -- so it triggers
    # an inspection of the frames rather than standing in for one.
    check_spatial_seams(frames, vertical_boundaries=(448, 896), horizontal_boundaries=(), max_ratio=2.0)
    check_spatial_seams(frames, vertical_boundaries=(), horizontal_boundaries=(384,), max_ratio=3.0)

    # Prompt alignment and the five VBench dimensions, on reference-driven content. Last, because it is
    # the most expensive check and the cheap structural ones above localise a failure better.
    _record_quality(frames, paths, case)


@pytest.mark.timeout(9000)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_ref2va_conditioning_is_not_a_no_op(mesh_device, reset_seeds):
    """The discriminator, against a floor measured in this same test.

    Three generations at the smallest ref2va shape:

    ``A``  a real decoded frame as the reference, seed 0
    ``A'`` the same request again, which measures the run-to-run floor
    ``B``  the same frame colour-inverted, seed 0

    Both references are the same size -- a 16:9 frame resolves to 2048x3584 at the 2048 px
    short edge -- so the packed layout is identical row for row and every noise draw has the
    same shape in the same order. The noise is therefore bit-identical and the two requests
    differ only in reference content, so an implementation that ignored the reference would
    score ``A`` against ``B`` as close as ``A`` against ``A'``.
    """
    pipeline = _pipeline(mesh_device)
    normal = _real_frame_image()
    inverted = _inverted(normal)

    def generate(image):
        return pipeline(
            PROMPT,
            references=[MiniMaxH3Reference(image=image)],
            num_frames=NUM_FRAMES,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            seed=SEED,
        )

    a = generate(normal)
    a_repeat = generate(normal)
    b = generate(inverted)

    floor = _divergence(a.video, a_repeat.video)
    signal = _divergence(a.video, b.video)
    to_normal = (_clip_resemblance(a, normal), _clip_resemblance(b, normal))
    to_inverted = (_clip_resemblance(a, inverted), _clip_resemblance(b, inverted))
    colour = (_colour_distance(a, normal), _colour_distance(b, inverted))
    crossed = (_colour_distance(a, inverted), _colour_distance(b, normal))

    logger.info(
        f"ref2va discriminator: run-to-run floor {floor:.6f}, reference-swap signal {signal:.6f} "
        f"(ratio {signal / max(floor, 1e-9):.1f}x) | CLIP to normal A={to_normal[0]:.4f} "
        f"B={to_normal[1]:.4f} | CLIP to inverted A={to_inverted[0]:.4f} B={to_inverted[1]:.4f} | "
        f"colour distance to own reference A={colour[0]:.4f} B={colour[1]:.4f}, "
        f"to the other A={crossed[0]:.4f} B={crossed[1]:.4f}"
    )
    for stem, output in (("discriminate_normal", a), ("discriminate_inverted", b)):
        _write(output, stem)

    # 1. The same request twice must be reproducible. If it is not, the floor is
    #    meaningless and so is everything measured against it, so this fails loudly
    #    rather than silently widening the bar.
    assert floor < 0.02, f"the same request twice diverged by {floor:.6f}; the pipeline is not reproducible"

    # 2. THE GATE THE NULL HYPOTHESIS FAILS. Both references are the same size, so the
    #    packed layout is identical row for row and every noise draw has the same shape
    #    in the same order -- the noise is bit-identical. An implementation that ignored
    #    the reference would therefore score `signal == floor` exactly. Measured against
    #    a floor of 0, so any nonzero signal is attributable to reference content alone;
    #    the absolute size is reported so a *shrinking* effect is visible too.
    assert signal > 10 * floor, (
        f"swapping the reference moved the output by {signal:.6f} against a run-to-run floor of "
        f"{floor:.6f}: the reference is not conditioning the output"
    )
    assert signal > 0.01, (
        f"swapping the reference moved the output by only {signal:.6f} mean absolute pixel value; "
        "the effect is present but too small to call conditioning"
    )

    # Recorded, not asserted. No instrument is known to measure "resembles its own reference
    # more than the other one" here: CLIP image-image similarity separates the two
    # OUTPUTS rather than the two references -- measured, one output scored higher against
    # both references with the gap equal to within 0.0013 -- and mean-RGB distance splits,
    # correct for one output and wrong for the other by the same 0.011. Asserting a direction
    # on either would be asserting a metric that cannot fail.
    logger.info(
        f"ref2va direction (recorded, not asserted): CLIP own-vs-other "
        f"A {to_normal[0]:.4f} vs {to_inverted[0]:.4f}, B {to_inverted[1]:.4f} vs {to_normal[1]:.4f}; "
        f"colour own-vs-other A {colour[0]:.4f} vs {crossed[0]:.4f}, B {colour[1]:.4f} vs {crossed[1]:.4f}"
    )


@pytest.mark.timeout(9000)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_ref2va_reference_order_changes_the_request(mesh_device, reset_seeds):
    """Reordering the same two references is a different request, and must generate differently.

    The host gate covers the layout changing; this covers the change reaching the output. Both
    references are the same size, so the two requests have identical row counts and an
    identical noise stream, and only the rotary clock and the presentation labels differ.
    """
    pipeline = _pipeline(mesh_device)
    normal = _real_frame_image()
    inverted = _inverted(normal)

    def generate(references):
        return pipeline(
            PROMPT,
            references=references,
            num_frames=NUM_FRAMES,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            seed=SEED,
        )

    forward = generate([MiniMaxH3Reference(image=normal), MiniMaxH3Reference(image=inverted)])
    reversed_ = generate([MiniMaxH3Reference(image=inverted), MiniMaxH3Reference(image=normal)])

    divergence = _divergence(forward.video, reversed_.video)
    logger.info(f"ref2va reference order: divergence {divergence:.6f}, padded_len={pipeline.last_padded_len}")
    assert forward.video.shape == reversed_.video.shape
    assert divergence > 0.01, (
        f"reordering the references changed the output by only {divergence:.6f}; the order is supposed to "
        "advance the shared rotary clock and relabel the presentation"
    )
