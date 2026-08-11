# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 `fl2va` end to end: a prompt plus a first and/or last keyframe in, video and audio out.

Three cases, the three tasks the reference's `_workflow_map` names:

    first            `image=`                   -> fl2va
    last             `last_image=`              -> fl2va_last_frame
    first_and_last   both                       -> fl2va with two anchors

A separate file from `test_pipeline_minimax_h3.py` rather than more cases in it, so the
two e2e gates default to separate processes. One process holding the 50-block DiT's programs *and*
its CCL persistent buffers at several different padded sequence lengths (37888 for t2va, 39424 and
40448 for the two fl2va layouts) is a memory risk nobody has measured.

What this gates beyond the t2va tiers
-------------------------------------
The reference-free A/V checks are shared with t2va. Two things are specific to `fl2va` and are the
reason this file exists:

- **the keyframe anchor actually anchors.** Decoded frame 0 (or -1) must correlate with the keyframe.
  At `noise_aug = 0.999` the anchor rows are essentially the clean VAE latent of the keyframe, so this
  is a strong signal, and it is the only one that would notice conditioning rows placed at the wrong
  sequence position.
- **the anchors survive the loop.** Nothing re-imposes them; they persist only because the loop never
  writes them. `t2va` cannot test this, having no conditioning rows at all.

Tier 6 and the keyframe's content
---------------------------------
The gated prompt and the tier-6 thresholds are a **matched pair**, and
`imaging_quality` is a no-reference IQA metric that moves with content, not just with correctness.
A keyframe *forces* the content, so an arbitrary photograph would invalidate the t2va calibration
outright.

So the gated keyframe is **frame 0 of the calibrated t2va generation** -- the fox scene those
thresholds were calibrated against. The content distribution is then the one they were measured on,
the same prompt still applies, and the anchor check is maximally meaningful because the keyframe is
in-distribution for the model. It also means this file needs the t2va gate to have run; it skips
rather than inventing content when the artifact is absent.

Tier 6 was recorded before it was gated. CLIP measures 36.63 / 37.30 / 37.00 across the three cases
against t2va's 37.37, so the t2va bar of 33.0 transfers by measurement rather than by assumption.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.minimax_h3.packing import align_num_frames, prepare_keyframe_image
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ..wan2_2.common import check_output_sanity
from .common import create_fractal_image
from .common_av import (
    check_audio_sanity,
    check_av_sync,
    check_keyframe_anchor,
    check_spatial_seams,
    check_tile_boundary_gradient,
    clip_prompt_alignment,
    decoded_frames,
    log_spectral_flatness,
    probe_streams,
    temporal_seam_score,
    write_artifacts,
)

# The t2va working point, unchanged, so the two are comparable. One anchor adds 1008 conditioning rows.
HEIGHT, WIDTH = 768, 1344
NUM_FRAMES = 124
NUM_INFERENCE_STEPS = 50
SEED = 0

# The prompt the t2va tier-6 thresholds were calibrated against, kept verbatim for the reason in the
# module docstring.
PROMPT = (
    "A red fox trots across a snowy field at dawn, its breath visible in the cold air. "
    "The low sun throws long blue shadows behind it, and loose snow lifts from each footfall."
)

WEIGHTS_ENV = "MINIMAX_H3_DIFFUSERS_DIR"
DEFAULT_WEIGHTS = "/data/cglagovich/MiniMax-H3-diffusers"
ARTIFACT_ENV = "MINIMAX_H3_ARTIFACT_DIR"
# Where the *calibrated t2va* artifact lives, which is where the gated keyframe comes from. Separate
# from ARTIFACT_ENV: that one says where fl2va *writes*, and pointing it at a fresh
# directory must not silently turn the keyframe source into a missing file.
T2VA_ARTIFACT_ENV = "MINIMAX_H3_T2VA_ARTIFACT_DIR"

# Ring collectives, so the fabric must be FABRIC_1D_RING -- same as the t2va gate.
MESH_4X8 = [
    pytest.param(
        (4, 8),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "require_exact_physical_num_devices": True,
            "l1_small_size": 65536,
        },
        id="4x8",
    )
]

# Set from measurement, not inherited. The wan2_2 analogue ships a provisional 0.3 with a note to
# tighten it once real values are observed; these are the observed values, all three cases:
#
#   first            frame  0 vs keyframe   0.9971
#   last             frame -1 vs keyframe   0.9943
#   first_and_last   frame  0 / frame -1    0.9971 / 0.9946
#
# 0.95 keeps ~9x margin on `1 - PCC` against the worst of those, which is the same margin convention
# the conditioner bar uses. Not tighter: the anchors go through a VAE round trip
# and 49 denoising steps of a *neighbouring* frame's influence, so some spread is expected, and a
# genuinely broken conditioning path scores near zero rather than 0.9.
ANCHOR_PCC_FLOOR = 0.95

# t2va's bar, applied here because the first fl2va run measured 36.63 / 37.30 / 37.00 against t2va's
# 37.37 -- i.e. it transfers. It transfers *because* the gated keyframe comes from the calibrated t2va
# generation; an arbitrary keyframe would need its own calibration.
CLIP_THRESHOLD = 33.0


def _weights_dir() -> Path:
    directory = Path(os.environ.get(WEIGHTS_ENV, DEFAULT_WEIGHTS))
    if not directory.is_dir():
        pytest.skip(f"no MiniMax-H3 snapshot at {directory}; set {WEIGHTS_ENV}")
    return directory


def _artifact_dir() -> Path:
    directory = Path(os.environ.get(ARTIFACT_ENV) or Path.home() / "h3_t2va_artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _gated_keyframe() -> Image.Image:
    """Frame 0 of the calibrated t2va generation, at the production canvas.

    **This keyframe cannot on its own show that conditioning works, and the anchor correlation it
    produces is confounded.** It is taken from t2va's *own output*, so a pipeline that ignored the
    keyframe entirely and merely re-ran t2va would also score ~0.997 against it. That is what
    `test_fl2va_follows_the_keyframe` exists to rule out, with a keyframe the model would never
    produce. What this one is for is tier 6: the CLIP and VBench bars were calibrated on this content,
    so it is the only keyframe those bars legitimately apply to.

    Read from the t2va gate's artifact rather than generated here: a second generation would cost
    another 90 s of Galaxy time to produce content this file only needs to *read*, and reusing the
    calibrated one is what makes the thresholds applicable (see the module docstring on tier 6).
    """
    source_dir = Path(os.environ.get(T2VA_ARTIFACT_ENV) or Path.home() / "h3_t2va_artifacts")
    artifact = source_dir / "t2va.mp4"
    if not artifact.is_file():
        pytest.skip(
            f"no calibrated t2va artifact at {artifact}; run test_pipeline_minimax_h3.py first so the "
            "fl2va keyframe comes from content the tier-6 thresholds were calibrated on"
        )
    frame = _first_frame(artifact)
    assert frame.size == (WIDTH, HEIGHT), f"t2va artifact is {frame.size}, expected {(WIDTH, HEIGHT)}"
    return frame


def _first_frame(path: Path) -> Image.Image:
    """Decode frame 0 of an mp4 with imageio, which the tier-5 checks already depend on."""
    import imageio.v3 as iio

    return Image.fromarray(np.asarray(iio.imread(path, index=0, plugin="pyav"))).convert("RGB")


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize(
    ("anchors", "case"),
    [
        pytest.param(("first",), "first", id="first"),
        # A lone `last_image` is the geometry anchor, so it is *stretched* despite being the "last"
        # anchor. That looks like a bug in `prepare_keyframe_image` until this case passes.
        pytest.param(("last",), "last", id="last"),
        # The only case with two conditioning blocks: 2016 rows, a two-run vision scatter in the
        # conditioner, and the cover-crop path for the follower keyframe.
        pytest.param(("first", "last"), "first_and_last", id="first_and_last"),
    ],
)
def test_fl2va_end_to_end(mesh_device, anchors, case, reset_seeds):
    keyframe = _gated_keyframe()
    image = keyframe if "first" in anchors else None
    last_image = keyframe if "last" in anchors else None

    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=_weights_dir())
    output = pipeline(
        PROMPT,
        image=image,
        last_image=last_image,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    expected_frames = align_num_frames(NUM_FRAMES)
    logger.info(
        f"fl2va[{case}] anchors={anchors} padded_len={pipeline.last_padded_len} "
        f"video={tuple(output.video.shape)} audio={tuple(output.audio.shape)}"
    )
    assert output.video.shape[2] == expected_frames, f"{output.video.shape[2]} frames, expected {expected_frames}"

    frames = (output.video[0].permute(1, 2, 3, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()

    # ---- tier 4: reference-free sanity, shared with t2va ----
    check_output_sanity(frames, num_frames=expected_frames, height=HEIGHT, width=WIDTH)
    check_audio_sanity(output.audio, sampling_rate=output.sampling_rate, expected_seconds=expected_frames / output.fps)
    check_av_sync(frames, output.audio, sampling_rate=output.sampling_rate, fps=output.fps)
    log_spectral_flatness(output.audio, sampling_rate=output.sampling_rate)

    # Boundaries from the VAE's own tile grid rather than re-derived: an earlier re-derivation used an
    # overlap of 32 against the real 64 and checked positions that were not boundaries at all. Note the
    # return shape -- `((y_starts, lengths, overlaps), (x_starts, ...))`, rows first -- and that
    # vertical seams come from the *x* starts.
    ratio = pipeline.vae_config.spatial_compression_ratio
    (y_starts, _, _), (x_starts, _, _) = pipeline.vae.decode_tile_grid(HEIGHT // ratio, WIDTH // ratio)
    check_spatial_seams(frames, vertical_boundaries=x_starts[1:], horizontal_boundaries=y_starts[1:])
    # The sensitive complement: block-mean seam ratios cannot see a one-pixel discontinuity, and the
    # tiled VAE decode leaves one at ~0.12 % of full scale. Sub-visible, and gated so it stays that way.
    check_tile_boundary_gradient(frames, vertical_boundaries=x_starts[1:], horizontal_boundaries=y_starts[1:])

    # ---- fl2va-specific: the anchors ----
    # `stretch` follows the pipeline's rule: the FIRST keyframe given is the geometry anchor.
    if image is not None:
        check_keyframe_anchor(
            frames, image, index=0, stretch=True, width=WIDTH, height=HEIGHT, pcc_floor=ANCHOR_PCC_FLOOR
        )
    if last_image is not None:
        check_keyframe_anchor(
            frames,
            last_image,
            index=-1,
            stretch=image is None,
            width=WIDTH,
            height=HEIGHT,
            pcc_floor=ANCHOR_PCC_FLOOR,
        )

    # ---- tier 5: the written file, not just the tensor ----
    artifacts = _artifact_dir()
    paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts, stem=f"fl2va_{case}")
    if "mp4" in paths:
        streams = probe_streams(paths["mp4"])
        if streams:
            assert "video" in streams and "audio" in streams, f"muxed file is missing a stream: {list(streams)}"
            durations = {k: float(v["duration"]) for k, v in streams.items() if v.get("duration")}
            if {"video", "audio"} <= set(durations):
                skew = durations["audio"] - durations["video"]
                # AAC pads to a frame boundary, so a little more slack than the tensor-level check.
                assert abs(skew) < 0.15, f"muxed A/V skew {skew:+.3f} s"
                logger.info(f"muxed A/V skew: {skew:+.4f} s")

        decoded = decoded_frames(paths["mp4"], count=1)
        if decoded.size:
            assert (
                decoded.shape[0] >= expected_frames - 1
            ), f"the written mp4 decodes to {decoded.shape[0]} frames, expected ~{expected_frames}"
            seam = temporal_seam_score(decoded, period=17)
            logger.info(f"temporal seam score at the 17-frame chunk period: {seam:.3f} (1.0 = no seam)")
            if np.isfinite(seam):
                assert seam < 3.0, (
                    f"inter-frame delta at chunk boundaries is {seam:.2f}x the delta elsewhere; "
                    "suspect temporal stitching (see the artifact rubric)"
                )

        # Frames for the artifact rubric. Statistics average away exactly the two defects that matter
        # -- a seam and a flicker -- so the reminder below is not rhetorical.
        for index in (0, 17, 62, expected_frames - 1):
            Image.fromarray(frames[index]).save(artifacts / f"fl2va_{case}_frame_{index}.png")

    # ---- tier 6: generative quality ----
    # The bar is t2va's, and it is applied because it was *measured* to transfer rather than assumed to.
    # First-run values, all three cases: mean 36.63 / 37.30 / 37.00 against t2va's 37.37, min 34.86.
    # That the keyframe is frame 0 of the calibrated t2va generation is what makes this legitimate --
    # the recorded counterexample is a night scene that scored imaging_quality 0.4884 against a
    # 0.64 bar while being visually perfect, purely because the content moved.
    if os.environ.get("RUN_CLIP", "1") == "1":
        alignment = clip_prompt_alignment(frames, PROMPT)
        logger.info(
            f"fl2va[{case}] CLIP prompt alignment: mean={alignment['mean']:.2f} "
            f"min={alignment['min']:.2f} max={alignment['max']:.2f} (bar {CLIP_THRESHOLD})"
        )
        assert alignment["mean"] > CLIP_THRESHOLD, (
            f"CLIP prompt alignment {alignment['mean']:.2f} is below {CLIP_THRESHOLD}; the video may not "
            "follow the prompt"
        )

    logger.info(
        "REMINDER: read the artifact rubric against these frames -- seams and flicker are what every "
        "whole-tensor metric averages away, and both are parallelism bugs"
    )


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_fl2va_follows_the_keyframe(mesh_device, reset_seeds):
    """The keyframe *drives* the generation -- it is not merely consistent with it.

    This is the test that makes the anchor numbers mean something, and it exists because the gated
    keyframe cannot do that job. That keyframe is frame 0 of the calibrated t2va generation, so a
    pipeline that ignored keyframes altogether and just re-ran t2va would score ~0.997 against it. The
    correlation is real but confounded, and reporting it alone overstates what has been shown.

    Here the keyframe is a **Mandelbrot fractal** -- content the model will never produce for a prompt
    about a fox in snow. Three claims, and only the three together are conclusive:

    1. decoded frame 0 resembles the fractal. Impossible to satisfy by ignoring the keyframe.
    2. decoded frame 0 resembles the fractal *much better* than it resembles t2va's own frame 0, which
       is what it would resemble if conditioning were a no-op. This is the discriminating comparison.
    3. the tail of the clip has left the fractal behind. A pipeline that pinned every frame rather than
       just the anchor would also pass 1 and 2, and would not be generating a video at all.

    The prompt still describes the fox, so the keyframe and the prompt disagree, which is the
    sharpest possible test of whether the conditioning rows are actually reaching the DiT: a no-op
    keyframe leaves the prompt to win outright.
    """
    fractal = create_fractal_image(WIDTH, HEIGHT)
    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=_weights_dir())
    output = pipeline(
        PROMPT,
        image=fractal,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )
    frames = (output.video[0].permute(1, 2, 3, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()

    def pcc(a, b):
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        return float(np.corrcoef(a, b)[0, 1])

    prepared = np.asarray(prepare_keyframe_image(fractal, HEIGHT, WIDTH, True))
    to_keyframe = pcc(frames[0], prepared)
    to_t2va = pcc(frames[0], np.asarray(_gated_keyframe()))
    tail = pcc(frames[-1], prepared)

    logger.info(
        f"fl2va keyframe-drives-generation: frame 0 vs fractal keyframe {to_keyframe:.4f}, "
        f"frame 0 vs t2va's own frame 0 {to_t2va:.4f}, frame -1 vs fractal keyframe {tail:.4f}"
    )
    for index in (0, 17, 62, align_num_frames(NUM_FRAMES) - 1):
        Image.fromarray(frames[index]).save(_artifact_dir() / f"fl2va_fractal_frame_{index}.png")

    # (1) the anchor followed the supplied keyframe.
    assert to_keyframe > ANCHOR_PCC_FLOOR, (
        f"decoded frame 0 correlates only {to_keyframe:.3f} with the fractal keyframe it was "
        "conditioned on; the keyframe is not reaching the DiT"
    )
    # (2) and it followed *that* keyframe rather than reproducing t2va. The margin is what rules out
    # the confound; a no-op pipeline inverts this comparison.
    assert to_keyframe > to_t2va + 0.30, (
        f"frame 0 resembles the fractal keyframe ({to_keyframe:.3f}) barely more than it resembles "
        f"t2va's own frame 0 ({to_t2va:.3f}); conditioning may be a no-op"
    )
    # (3) it is still a video, not 124 copies of the keyframe.
    assert tail < to_keyframe - 0.20, (
        f"the last frame still correlates {tail:.3f} with the keyframe against frame 0's "
        f"{to_keyframe:.3f}; the clip may be pinned throughout rather than anchored at one end"
    )
