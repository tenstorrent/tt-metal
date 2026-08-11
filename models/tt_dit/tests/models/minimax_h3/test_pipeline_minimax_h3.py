# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end `t2va`: a prompt in, an mp4 with a soundtrack out, at the production working point.

One test carries both the fully-warm latency measurement and the quality gates, folded into a
single weight load: build the pipeline once, run one full warmup generation at the target shape
(`MiniMaxH3Pipeline.warmup`, the analogue of `LTXPipeline.warmup_buffers`), then run the timed
generation and gate *its* output. The timing method is `pipelines/ltx`'s, so the numbers stay
comparable: prepares and export excluded, `Total (compute)` = the sum of `pipeline.last_timings`
rows, and only the fully-warm second call is quoted. There is no tuned perf target yet -- the
bringup directive was "current perf, no tuning" -- so `EXPECTED_TOTAL_S` is a loose
did-something-collapse bar, not a performance target. Duration scaling (10 s / 15 s) is covered by
the block-level test in `test_performance_minimax_h3.py`, not here.

There is no torch reference for a whole generation -- 50 layers over 38222 rows for 49 steps is not
a CPU computation -- so correctness here is established by the tiers that do not need one, in the
order they catch things:

  tier 4  reference-free sanity: geometry, finiteness, range, spatial variance, inter-frame motion
          (`tests/models/wan2_2/common.py`), plus the audio analogue and an A/V sync check
          (`common_av.py`)
  tier 5  the written *file* re-decoded with ffmpeg -- so container, pixel format and frame count
          are checked, not just the tensor -- plus a temporal-seam score and a log-spectrum shape
  tier 6  CLIP prompt alignment in-process, and VBench dimensions in a separate interpreter --
          VBench pins numpy < 2 and transformers 4.33, so it cannot share this environment. Both
          env-gated and defaulting **on**.

Per-component numerics are gated elsewhere and are not repeated here: the conditioner in
`test_text_encoder_minimax_h3.py`, the DiT in `test_transformer_minimax_h3.py`, both VAEs in
`test_vae_*` and `test_audio_minimax_h3.py`, all at this same 768P/5s working point.

Artifacts are written to a stable path so the output can be *looked at*. Every numeric gate below
can pass on video that is visibly wrong, and the two failure modes whole-tensor statistics hide best
are both live here: seams at the video VAE's 4x7 = 28 spatial tiles and its 17-frame temporal chunks,
and temporal flicker from per-frame GroupNorm statistics or chunk-boundary stitching. Reading the artifact rubric
against the real frames is part of the gate, not an optional follow-up.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....utils.test import ring_params_req_exact_devices
from ..wan2_2.common import check_output_sanity
from .common_av import (
    check_audio_sanity,
    check_av_sync,
    check_spatial_seams,
    clip_prompt_alignment,
    decoded_frames,
    log_spectral_flatness,
    probe_streams,
    run_vbench,
    temporal_seam_score,
    write_artifacts,
)

# The production working point, and the only one gated: 1344x768 at 16:9, 124 frames @ 24 fps
# (5.17 s), 50 scheduler steps -> 49 forwards.
HEIGHT, WIDTH = 768, 1344
NUM_FRAMES = 124
NUM_INFERENCE_STEPS = 50
SEED = 0

# Dense: a moving camera, a reflective wet surface, several independent light sources at
# different colour temperatures, volumetric haze, and foreground/background motion at different depths.
# Those are the things a video model is most likely to get wrong, and they are also what the artifact
# rubric reads best -- banding shows in the haze gradients, seams show in the reflections, and flicker
# shows in the neon.
# The gated prompt and the tier-6 thresholds are a **matched pair**. Both were calibrated together on
# this prompt; swapping one without recalibrating the other breaks the gate. Measured
# here: CLIP 37.37, VBench imaging_quality 0.6896.
#
# `imaging_quality` in particular is prompt-dependent, not just model-dependent: it is a no-reference
# IQA model that rewards sharp, well-lit frames. A dark rain-at-night scene scored **0.4884** against
# this same 0.64 bar while looking entirely correct. So a showcase prompt belongs in a
# manual run, not in this constant.
PROMPT = (
    "A red fox trots across a snowy field at dawn, its breath visible in the cold air. "
    "The low sun throws long blue shadows behind it, and loose snow lifts from each footfall."
)


WEIGHTS_ENV = "MINIMAX_H3_DIFFUSERS_DIR"
DEFAULT_WEIGHTS = "/data/cglagovich/MiniMax-H3-diffusers"
ARTIFACT_ENV = "MINIMAX_H3_ARTIFACT_DIR"

# Generous: a regression bar, not a target. Measured fully-warm total is well inside this, and the
# point is to notice a collapse (a lost cache, a fallback kernel) rather than to police seconds.
EXPECTED_TOTAL_S = 400.0

# The pipeline's CCLManager runs ring collectives, so the fabric must be FABRIC_1D_RING. Taken from
# the shared helper rather than hand-written: on a plain FABRIC_1D (line) fabric a ring collective
# cannot resolve a forwarding direction and fails as
# `TT_FATAL fabric.cpp:174 forwarding_direction.has_value()`, which reads like a CCL bug.
MESH_4X8 = [
    pytest.param(
        (4, 8),
        {**ring_params_req_exact_devices, "l1_small_size": 65536},
        id="4x8",
    )
]


def _weights_dir():
    base = os.environ.get(WEIGHTS_ENV, DEFAULT_WEIGHTS)
    missing = [p for p in ("transformer", "text_encoder", "vae", "audio_vae") if not os.path.isdir(f"{base}/{p}")]
    if missing:
        pytest.skip(f"MiniMax-H3 snapshot at {base} is missing {missing}; set {WEIGHTS_ENV}")
    return base


def _artifact_dir():
    directory = Path(os.environ.get(ARTIFACT_ENV) or Path.home() / "h3_t2va_artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _to_uint8_frames(video: torch.Tensor) -> np.ndarray:
    """`(1, 3, F, H, W)` in [0, 1] -> `(F, H, W, 3)` uint8, which is what the checkers take."""
    assert video.ndim == 5 and video.shape[0] == 1, f"unexpected video shape {tuple(video.shape)}"
    frames = video[0].permute(1, 2, 3, 0).clamp(0, 1).mul(255).round().to(torch.uint8)
    return frames.cpu().numpy()


# VBench dimensions and the bars, **calibrated on this model at this working point** (2026-08-04,
# seed 0, the fox prompt) rather than copied from LTX's 1088p set:
#
#   subject_consistency 0.9820   background_consistency 0.9831   motion_smoothness 0.9905
#   dynamic_degree 1.0           imaging_quality 0.6896
#
# For reference, LTX's calibrated 1088p bars are 0.92 / 0.93 / 0.955 / 1.0 / 0.645, which H3 clears on
# every dimension, so those values would gate nothing here.
#
# The margins below are generous because this is a **single-sample** calibration: one
# prompt, one seed. They are set to catch a broken pipeline, not to certify quality. `dynamic_degree`
# stays at 1.0 because over one video it is effectively binary -- either the clip has real motion or
# it does not, and "it does not" is the frozen-video failure.
VBENCH_THRESHOLDS = {
    "subject_consistency": 0.95,
    "background_consistency": 0.95,
    "motion_smoothness": 0.97,
    "dynamic_degree": 1.0,
    "imaging_quality": 0.64,
}
# Measured mean 37.05 over 8 evenly-spaced frames (min 36.16), which sits at wan2.2's ~37 baseline
# rather than LTX's ~31.3. 33.0 leaves ~4 points, matching the headroom LTX allows for run-to-run
# variance.
CLIP_THRESHOLD = 33.0


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_t2va_end_to_end(mesh_device, reset_seeds):
    weights_dir = _weights_dir()
    artifacts = _artifact_dir()

    vbench_enabled = os.environ.get("RUN_VBENCH", "1") in ("1", "true", "True")
    run_clip = os.environ.get("RUN_CLIP", "1") in ("1", "true", "True")

    # `MINIMAX_H3_PROMPT` overrides the gated prompt for a manual showcase run, which is where the
    # constant's own note says a showcase prompt belongs. The tier-6 thresholds are calibrated
    # *against that constant*, so an override forces CLIP and VBench off rather than reporting a
    # failure that says nothing about the model -- `imaging_quality` alone has swung 0.6896 -> 0.4884
    # between two correct-looking scenes. Tiers 4 and 5 are prompt-independent and still run.
    prompt = os.environ.get("MINIMAX_H3_PROMPT") or PROMPT
    if prompt is not PROMPT:
        logger.info("MINIMAX_H3_PROMPT override in use; disabling the prompt-calibrated tier-6 gates")
        vbench_enabled = run_clip = False
    # A missing dependency must report SKIPPED, never a silent pass: a quality gate that no-ops
    # reads green, which is worse than not having it. VBench is checked as an *interpreter* rather
    # than an import, because it does not live in this environment (see `common_av.run_vbench`);
    # CLIP needs only `open_clip`, which is already here, and this test's own ffmpeg frames.
    if run_clip:
        pytest.importorskip("open_clip", reason="RUN_CLIP=1 but open_clip is not installed (set RUN_CLIP=0)")

    if not os.environ.get("TT_DIT_CACHE_DIR"):
        logger.warning(
            "TT_DIT_CACHE_DIR is unset, so every weight load reads safetensors. Prepares are excluded "
            "from the total either way, but the run will take far longer than the reported compute."
        )

    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=weights_dir)

    # ---- fully-warm latency, `pipelines/ltx`'s method ----
    # Warm every program and buffer this shape touches, at the real prompt. Not timed: this is the
    # warm-window method, not the measurement.
    pipeline.warmup(
        prompt=prompt, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, num_inference_steps=NUM_INFERENCE_STEPS
    )
    warm_padded_len = pipeline.last_padded_len
    # Prime the disk cache so the Encoder row is the cache-hit row every reported number quotes;
    # `warmup` runs with `use_prompt_cache=False`, so it compiles the conditioner but writes nothing.
    pipeline.encode_prompt(prompt, use_cache=True)

    output = pipeline(
        prompt,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    assert pipeline.last_padded_len == warm_padded_len, (
        f"warmup ran at padded_len {warm_padded_len} but the measured call ran at "
        f"{pipeline.last_padded_len}; this number is not warm"
    )

    expected_frames = align_num_frames(NUM_FRAMES)
    logger.info(
        f"generated {output.num_frames} frames ({output.video_seconds:.3f} s) and "
        f"{output.audio_seconds:.3f} s of audio at {output.sampling_rate} Hz"
    )

    rows = pipeline.last_timings
    total = sum(seconds for _, seconds in rows)
    num_forwards = NUM_INFERENCE_STEPS - 1
    logger.info(
        f"MEASUREMENT t2va fully warm | mesh 4x8 Blackhole, TP=4 axis 0 / SP=8 axis 1, ring, 2 links "
        f"| {WIDTH}x{HEIGHT}, {expected_frames} frames @ {MINIMAX_H3_FPS} fps "
        f"({expected_frames / MINIMAX_H3_FPS:.2f} s), {num_forwards} forwards, padded_len {warm_padded_len} "
        f"| warm window: one full warmup generation at this shape, prepares and export excluded"
    )
    for label, seconds in rows:
        logger.info(f"  {label:<18} {seconds:8.1f} s  ({100 * seconds / total:4.1f} %)")
    logger.info(f"  {'Total (compute)':<18} {total:8.1f} s")
    denoise = dict(rows).get("Denoise")
    if denoise:
        logger.info(
            f"  per forward        {denoise / num_forwards * 1000:8.1f} ms  "
            f"({num_forwards} forwards over {denoise:.1f} s)"
        )
    logger.info(f"  realtime factor    {total / (expected_frames / MINIMAX_H3_FPS):8.1f} x  (compute / video seconds)")
    # No tuned target yet (bringup at current perf); this is a loose did-something-collapse bar.
    assert total < EXPECTED_TOTAL_S, f"fully-warm total {total:.1f} s exceeds the {EXPECTED_TOTAL_S:.0f} s floor bar"

    frames = _to_uint8_frames(output.video)

    # ---- tier 4: reference-free sanity, video then audio then the pairing ----
    check_output_sanity(frames, num_frames=expected_frames, height=HEIGHT, width=WIDTH)
    check_audio_sanity(
        output.audio,
        sampling_rate=output.sampling_rate,
        expected_seconds=expected_frames / MINIMAX_H3_FPS,
    )
    sync = check_av_sync(
        frames,
        output.audio,
        sampling_rate=output.sampling_rate,
        fps=MINIMAX_H3_FPS,
    )
    log_spectral_flatness(output.audio, sampling_rate=output.sampling_rate)

    # Spatial seams, at the tile geometry the VAE **actually used**. Asked of the VAE object rather
    # than re-derived: an earlier version of this recomputed `split_tiles(HEIGHT, 256, 32, ratio)`
    # with a hardcoded overlap of 32 against the real `DEFAULT_TILE_OVERLAP = 64`, which produced a
    # 4x6 grid instead of 4x7 and checked columns that are not tile boundaries at all. The gate
    # passed while measuring nothing. Duplicating a derivation is how that happens; asking the
    # object that owns it is how it does not.
    ratio = pipeline.vae_config.spatial_compression_ratio
    (y_starts, _, _), (x_starts, _, _) = pipeline.vae.decode_tile_grid(HEIGHT // ratio, WIDTH // ratio)
    check_spatial_seams(frames, vertical_boundaries=x_starts[1:], horizontal_boundaries=y_starts[1:])

    # ---- tier 5: the written file, not just the tensor ----
    paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts)
    if "mp4" in paths:
        streams = probe_streams(paths["mp4"])
        if streams:
            logger.info(f"container streams: { {k: v.get('duration') for k, v in streams.items()} }")
            assert "video" in streams and "audio" in streams, f"muxed file is missing a stream: {list(streams)}"
            durations = {k: float(v["duration"]) for k, v in streams.items() if v.get("duration")}
            if {"video", "audio"} <= set(durations):
                skew = durations["audio"] - durations["video"]
                # AAC pads to a frame boundary, so allow a little more than the tensor-level check.
                assert abs(skew) < 0.15, f"muxed A/V skew {skew:+.3f} s"
                logger.info(f"muxed A/V skew: {skew:+.4f} s")

        decoded = decoded_frames(paths["mp4"], count=1)
        if decoded.size:
            assert (
                decoded.shape[0] >= expected_frames - 1
            ), f"the written mp4 decodes to {decoded.shape[0]} frames, expected ~{expected_frames}"
            # The VAE's temporal chunk covers clip_length (17) pixel frames.
            seam = temporal_seam_score(decoded, period=17)
            logger.info(f"temporal seam score at the 17-frame chunk period: {seam:.3f} (1.0 = no seam)")
            if np.isfinite(seam):
                assert seam < 3.0, (
                    f"inter-frame delta at chunk boundaries is {seam:.2f}x the delta elsewhere; "
                    "suspect temporal stitching (see the artifact rubric)"
                )

    # ---- tier 6: generative quality ----
    if run_clip:
        alignment = clip_prompt_alignment(frames, prompt)
        logger.info(
            f"CLIP prompt alignment: mean={alignment['mean']:.2f} "
            f"min={alignment['min']:.2f} max={alignment['max']:.2f} (bar {CLIP_THRESHOLD})"
        )
        assert (
            alignment["mean"] >= CLIP_THRESHOLD
        ), f"CLIP mean {alignment['mean']:.2f} < {CLIP_THRESHOLD}; the video does not match the prompt"
    else:
        logger.info("RUN_CLIP=0, skipping the CLIP prompt-alignment gate")

    if vbench_enabled:
        assert "mp4" in paths, "RUN_VBENCH=1 needs the muxed mp4, which ffmpeg did not produce"
        scores = run_vbench(paths["mp4"], prompt, VBENCH_THRESHOLDS.keys())
        for dimension, bar in VBENCH_THRESHOLDS.items():
            # A requested dimension with no returned score is an *ungated* dimension, not a pass.
            assert dimension in scores, f"VBench returned no score for {dimension}"
            logger.info(f"VBench {dimension} = {scores[dimension]:.4f} (bar {bar})")
        failures = [f"{d} = {scores[d]:.4f} < {bar:.4f}" for d, bar in VBENCH_THRESHOLDS.items() if scores[d] < bar]
        assert not failures, "VBench below threshold: " + "; ".join(failures)
    else:
        logger.info("RUN_VBENCH=0, skipping the VBench gate")

    logger.info(f"artifacts in {artifacts}: {sorted(p.name for p in artifacts.iterdir())}")
    logger.info(
        "REMINDER: read the artifact rubric against these frames -- the seam and flicker scores above "
        "are statistics, and only looking at the output catches what they average away"
    )
    assert sync["video_seconds"] > 0
