# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end `t2va`: a prompt in, an mp4 with a soundtrack out, at the production working point.

One test carries both the fully-warm latency measurement and the quality gates on a single weight
load: build the pipeline once, run one full warmup generation at the target shape
(`MiniMaxH3Pipeline.warmup`, the analogue of `LTXPipeline.warmup_buffers`), then run the timed
generation and gate *its* output. The timing method is `pipelines/ltx`'s, so the numbers stay
comparable: prepares and export excluded, `Total (compute)` = the sum of `pipeline.last_timings`
rows, and only the fully-warm second call is quoted. There is no tuned perf target yet -- the
bringup scope is "current perf, no tuning" -- so `EXPECTED_TOTAL_S` is a loose
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

import pytest
from loguru import logger

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....utils.test import ring_params_req_exact_devices
from ..wan2_2.common import check_output_sanity
from .common_av import (
    CALIBRATED_FOX_PROMPT,
    artifact_dir,
    check_audio_sanity,
    check_av_sync,
    check_spatial_seams,
    check_written_file,
    clip_gate_enabled,
    gate_clip,
    gate_vbench,
    log_spectral_flatness,
    log_timing_table,
    run_warm_generation,
    to_uint8_frames,
    vbench_gate_enabled,
    weights_dir,
    write_artifacts,
)

# The production working point, and the only one gated: 1344x768 at 16:9, 124 frames @ 24 fps
# (5.17 s), 50 scheduler steps -> 49 forwards.
HEIGHT, WIDTH = 768, 1344
NUM_FRAMES = 124
NUM_INFERENCE_STEPS = 50
SEED = 0

# The calibrated fox prompt: the tier-6 thresholds below and this prompt are a **matched pair**
# (the full calibration note lives on the constant in `common_av.py`, which also exports it to the
# fl2va gate so the two stay identical by import).
PROMPT = CALIBRATED_FOX_PROMPT

ARTIFACT_ENV = "MINIMAX_H3_ARTIFACT_DIR"

# Generous: a regression bar, not a target. Measured fully-warm total is well inside this even with
# the ~2.8 s encoder forward inside the timed window (there is no prompt-embedding cache), and
# the point is to notice a collapse (a lost cache, a fallback kernel) rather than to police seconds.
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
    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    artifacts = artifact_dir(ARTIFACT_ENV, "h3_t2va_artifacts")

    vbench_enabled = vbench_gate_enabled()
    run_clip = clip_gate_enabled()

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

    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=weights)

    # ---- fully-warm latency, `pipelines/ltx`'s method ----
    # The warmup inside `run_warm_generation` warms every program and buffer this shape touches, at
    # the real prompt, and is not timed. The timed run pays the real encoder forward: there is no
    # prompt-embedding cache, so the Encoder row is a genuine measurement (~2.8 s for t2va with the
    # encoder co-resident), and the warmup already compiled the conditioner's kernels at this
    # padded length.
    output = run_warm_generation(
        pipeline,
        prompt,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    expected_frames = align_num_frames(NUM_FRAMES)
    logger.info(
        f"generated {output.num_frames} frames ({output.video_seconds:.3f} s) and "
        f"{output.audio_seconds:.3f} s of audio at {output.sampling_rate} Hz"
    )

    num_forwards = NUM_INFERENCE_STEPS - 1
    video_seconds = expected_frames / MINIMAX_H3_FPS
    # No tuned target yet (bringup at current perf); EXPECTED_TOTAL_S is a loose
    # did-something-collapse bar.
    log_timing_table(
        pipeline,
        "t2va",
        num_forwards=num_forwards,
        video_seconds=video_seconds,
        expected_total_s=EXPECTED_TOTAL_S,
        extra=(
            f" | {WIDTH}x{HEIGHT}, {expected_frames} frames @ {MINIMAX_H3_FPS} fps "
            f"({video_seconds:.2f} s), {num_forwards} forwards, padded_len {pipeline.last_padded_len}"
        ),
    )

    frames = to_uint8_frames(output)

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
    # than re-derived: recomputing `split_tiles(HEIGHT, 256, 32, ratio)` with a hardcoded overlap of
    # 32 against the real `DEFAULT_TILE_OVERLAP = 64` yields a 4x6 grid instead of 4x7 and checks
    # columns that are not tile boundaries at all -- a gate that passes while measuring nothing.
    # Asking the object that owns the derivation rules that out.
    ratio = pipeline.vae_config.spatial_compression_ratio
    (y_starts, _, _), (x_starts, _, _) = pipeline.vae.decode_tile_grid(HEIGHT // ratio, WIDTH // ratio)
    check_spatial_seams(frames, vertical_boundaries=x_starts[1:], horizontal_boundaries=y_starts[1:])

    # ---- tier 5: the written file, not just the tensor ----
    paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts)
    check_written_file(paths, expected_frames)

    # ---- tier 6: generative quality ----
    gate_clip(frames, prompt, CLIP_THRESHOLD, "t2va", enabled=run_clip)
    gate_vbench(paths, prompt, VBENCH_THRESHOLDS, "t2va", enabled=vbench_enabled)

    logger.info(f"artifacts in {artifacts}: {sorted(p.name for p in artifacts.iterdir())}")
    logger.info(
        "REMINDER: read the artifact rubric against these frames -- the seam and flicker scores above "
        "are statistics, and only looking at the output catches what they average away"
    )
    assert sync["video_seconds"] > 0
