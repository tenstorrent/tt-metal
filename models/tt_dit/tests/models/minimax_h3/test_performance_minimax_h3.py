# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Fully-warm end-to-end `t2va` latency, measured the way `pipelines/ltx` measures it.

The method is copied from `pipeline_ltx_distilled.py`, not invented here, so the two models'
numbers are directly comparable:

* **Warmup first.** One full generation at the target shape (`MiniMaxH3Pipeline.warmup`, the
  analogue of `LTXPipeline.warmup_buffers`) compiles every program and allocates every persistent
  buffer this working point touches. Only the *second* call is measured.
* **Prepares and export excluded.** Weight upload is one-time construction cost and the measurement
  contract never counts it. Each stage's `_prepare_*` runs outside its timed row, and writing the
  mp4 is not timed at all.
* **`Total (compute)` is the sum of the stage rows**, which is what LTX reports and what should be
  quoted.

A number without its mesh shape, input shape and warm-window method is not a measurement, so all
three are logged with every result.

This file measures and reports; it does not gate. Correctness lives in
`test_pipeline_minimax_h3.py`, and there is no tuned target to regress against yet -- the bringup
directive was explicitly "current perf, no tuning". `EXPECTED_TOTAL_S` exists as a loose
did-something-collapse bar, not a performance target.
"""

from __future__ import annotations

import os

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM, MiniMaxH3RotaryPosEmbed
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock as TorchMiniMaxH3Block
from diffusers.modular_pipelines.minimax_h3.packing import (
    MINIMAX_H3_FPS,
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)
from loguru import logger
from tracy import signpost

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....utils.tensor import bf16_tensor_2dshard, from_torch
from ....utils.test import skip_if_unsupported_num_links
from .common import dit_mesh_params, mesh_params

# The production working point, identical to the correctness gate's.
HEIGHT, WIDTH = 768, 1344
NUM_INFERENCE_STEPS = 50
SEED = 0

# Frame counts for the three target durations at 24 fps, under the model's 17n+5 alignment rule:
# `align_num_frames(round(duration * MINIMAX_H3_FPS))` gives 124 / 243 / 362, i.e. n = 7 / 14 / 21.
# 124 is the original single working point and stays the default-looking first case.
DURATIONS = [
    pytest.param(124, id="5s"),
    pytest.param(243, id="10s"),
    pytest.param(362, id="15s"),
]

# Dialogue-heavy: t2va generates a soundtrack, so a prompt with a spoken line exercises the
# audio path rather than leaving it to ambience.
PROMPT = (
    "Jerry, George, Elaine and Kramer are crowded into a red vinyl booth at a bright New York diner, "
    "coffee cups and menus on the table. George leans in and says, 'These video generation models can "
    "conjure a whole city out of nothing, and they still can't count fingers. The machine watched every "
    "movie ever made and concluded people have, what, nine? Eleven on a good day?' Elaine laughs into "
    "her coffee, Jerry shrugs with both palms up, and Kramer bursts through the door behind them."
)

# ref2va runs at the 5 s working point only: the correctness gate covers its three shapes there,
# and a perf
# row at a duration no correctness gate covers would be a number about an unverified request.
NUM_FRAMES_REF2VA = 124

WEIGHTS_ENV = "MINIMAX_H3_DIFFUSERS_DIR"
DEFAULT_WEIGHTS = "/data/cglagovich/MiniMax-H3-diffusers"

# Generous: a regression bar, not a target. Measured fully-warm total is well inside this, and the
# point is to notice a collapse (a lost cache, a fallback kernel) rather than to police seconds.
EXPECTED_TOTAL_S = 400.0

MESHES = mesh_params()


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("num_frames", DURATIONS)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESHES, indirect=["mesh_device", "device_params"])
def test_t2va_warm_latency(mesh_device, reset_seeds, num_frames):
    base = os.environ.get(WEIGHTS_ENV, DEFAULT_WEIGHTS)
    missing = [p for p in ("transformer", "text_encoder", "vae", "audio_vae") if not os.path.isdir(f"{base}/{p}")]
    if missing:
        pytest.skip(f"MiniMax-H3 snapshot at {base} is missing {missing}; set {WEIGHTS_ENV}")
    if not os.environ.get("TT_DIT_CACHE_DIR"):
        logger.warning(
            "TT_DIT_CACHE_DIR is unset, so every weight load reads safetensors. Prepares are excluded "
            "from the total either way, but the run will take far longer than the reported compute."
        )

    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=base)

    # Warm every program and buffer this shape touches. Not timed: this is the warm-window method,
    # not the measurement.
    # `prompt=PROMPT`, matching the fl2va and ref2va warmups. The default is the literal string
    # "warmup", which tokenizes to a different length, and the text stream's matmuls are keyed on it:
    # `token_refiner(context_embedder(prompt_1BLP))` has M = the prompt's token count. Warming at a
    # different prompt therefore leaves the text path's programs uncompiled at the measured shape,
    # so the "fully warm" run was still paying for them -- and on the quad it is fatal rather than
    # merely slow, because a trace capture cannot compile ("Cannot load new binaries during trace
    # capture. This program is not yet in program cache").
    pipeline.warmup(
        prompt=PROMPT,
        num_frames=num_frames,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )

    output = pipeline(
        PROMPT,
        num_frames=num_frames,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    rows = pipeline.last_timings
    total = sum(seconds for _, seconds in rows)
    num_forwards = NUM_INFERENCE_STEPS - 1
    aligned_frames = align_num_frames(num_frames)

    # Read off the pipeline, not written out: "a number without its mesh shape is not a measurement",
    # and a hardcoded 4x8 in this line would mislabel every 4x32 row rather than fail.
    shape = tuple(mesh_device.shape)
    logger.info(
        f"MEASUREMENT t2va fully warm | mesh {shape[0]}x{shape[1]} Blackhole, "
        f"TP={pipeline.tp_factor} axis {pipeline.tp_axis} / SP={pipeline.sp_factor} axis {pipeline.sp_axis}, "
        f"{pipeline.ccl_manager.topology}, {pipeline.ccl_manager.num_links} links "
        f"| {WIDTH}x{HEIGHT}, {aligned_frames} frames @ {MINIMAX_H3_FPS} fps "
        f"({aligned_frames / MINIMAX_H3_FPS:.2f} s), {num_forwards} forwards, "
        f"packed {pipeline.last_padded_len} padded ({pipeline.last_padded_len // pipeline.sp_factor}/device) "
        f"| warm window: one full warmup generation, prepares and export excluded"
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
    logger.info(f"  realtime factor    {total / (aligned_frames / MINIMAX_H3_FPS):8.1f} x  (compute / video seconds)")

    assert output.num_frames == aligned_frames
    assert total < EXPECTED_TOTAL_S, f"fully-warm total {total:.1f} s exceeds the {EXPECTED_TOTAL_S:.0f} s floor bar"


# The fl2va keyframe comes from the calibrated t2va artifact, same as the fl2va correctness gate --
# see that file's module docstring for why the content is not arbitrary.
T2VA_ARTIFACT_ENV = "MINIMAX_H3_T2VA_ARTIFACT_DIR"


def _fl2va_keyframe():
    from pathlib import Path

    import imageio.v3 as iio
    import numpy as np
    from PIL import Image

    source = Path(os.environ.get(T2VA_ARTIFACT_ENV) or Path.home() / "h3_t2va_artifacts") / "t2va.mp4"
    if not source.is_file():
        pytest.skip(f"no t2va artifact at {source} to take the fl2va keyframe from")
    return Image.fromarray(np.asarray(iio.imread(source, index=0, plugin="pyav"))).convert("RGB")


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESHES, indirect=["mesh_device", "device_params"])
def test_fl2va_warm_latency(mesh_device, reset_seeds):
    """Fully-warm `fl2va` latency, by the same method as the t2va row so the two are comparable.

    Three things have to be right or the number means nothing, and each has bitten a measurement in
    this model's bringup or its sibling's:

    1. **The warmup must be fl2va-shaped, keyframe included.** `padded_len` goes 37888 (t2va) to 39936
       (one anchor), and every program in the 50-block stack is keyed on it, so a t2va warmup warms
       nothing for fl2va.
    2. **The warmup must run at the same prompt length**, and that is *asserted* rather than assumed.
       t2va got away with a one-token `"warmup"` prompt purely by luck -- 1 and 39 tokens both round up
       to 37888 -- and with a ~1010-row vision block that luck is gone. The assertion below is one line
       and covers a whole class of silently-cold measurements.
    3. **The embedding cache must be populated.** `warmup` runs with `use_prompt_cache=False`, so it
       compiles the conditioner and writes nothing; without the priming call the measured run would pay
       a full device conditioner encode -- now including the vision tower -- inside the timed Encoder
       row, which is exactly what amendment 81's `Encoder (cache) 0.0 s` row does not include.
    """
    base = os.environ.get(WEIGHTS_ENV, DEFAULT_WEIGHTS)
    missing = [p for p in ("transformer", "text_encoder", "vae", "audio_vae") if not os.path.isdir(f"{base}/{p}")]
    if missing:
        pytest.skip(f"MiniMax-H3 snapshot at {base} is missing {missing}; set {WEIGHTS_ENV}")

    keyframe = _fl2va_keyframe()
    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=base)

    # (1) and (2): warm at the real shape, with the keyframe and the real prompt.
    pipeline.warmup(
        prompt=PROMPT,
        image=keyframe,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )
    warm_padded_len = pipeline.last_padded_len

    # (3): prime the disk cache so the Encoder row means what it means for t2va.
    pipeline.encode_prompt(PROMPT, keyframes=[keyframe], use_cache=True)

    output = pipeline(
        PROMPT,
        image=keyframe,
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

    rows = pipeline.last_timings
    total = sum(seconds for _, seconds in rows)
    num_forwards = NUM_INFERENCE_STEPS - 1
    aligned_frames = align_num_frames(NUM_FRAMES)

    logger.info(
        f"MEASUREMENT fl2va fully warm | mesh 4x8 Blackhole, TP=4 axis 0 / SP=8 axis 1, ring, 2 links "
        f"| {WIDTH}x{HEIGHT}, {aligned_frames} frames @ {MINIMAX_H3_FPS} fps "
        f"({aligned_frames / MINIMAX_H3_FPS:.2f} s), {num_forwards} forwards, one 'first' anchor, "
        f"padded_len {warm_padded_len} "
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
    logger.info(f"  realtime factor    {total / (aligned_frames / MINIMAX_H3_FPS):8.1f} x  (compute / video seconds)")

    assert output.num_frames == aligned_frames
    assert total < EXPECTED_TOTAL_S, f"fully-warm total {total:.1f} s exceeds the {EXPECTED_TOTAL_S:.0f} s floor bar"


# --------------------------------------------------------------------------- ref2va

# `l1_small_size` 16384, not the 65536 the t2va and fl2va rows use. Measured, not chosen: a video
# reference goes through the video VAE's taps=3 encoder, whose static circular buffers clash with L1
# above 16384 (am. 124/126). The mesh and every other device parameter are unchanged, so the
# ref2va row stays comparable to the other two on everything that affects the denoise loop.
REF2VA_MESHES = mesh_params(l1_small_size=16384)

# The three shapes gated end to end, by their measured padded packed length. `padded_len`
# is what every program in the 50-block stack is keyed on, so it is the identity of a perf row here.
REF2VA_CASES = [
    pytest.param("one_image", 46080, id="one_image_s46080"),
    pytest.param("video_with_sound", 81664, id="video_with_sound_s81664"),
    pytest.param("mixed", 89856, id="mixed_s89856"),
]

REF2VA_MEDIA_ENV = "MINIMAX_H3_REFERENCE_MEDIA"


def _ref2va_references(case: str):
    """The correctness gate's own reference sets, imported rather than restated.

    A perf row measured on a different request than the one that was gated is a number about
    nothing.
    """
    from .test_pipeline_ref2va_minimax_h3 import _references

    return _references(case)


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(("mesh_device", "device_params"), REF2VA_MESHES, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize(("case", "expected_padded_len"), REF2VA_CASES)
def test_ref2va_warm_latency(mesh_device, case, expected_padded_len, reset_seeds):
    """Fully-warm `ref2va` latency, by the same method as the t2va and fl2va rows.

    A cold total says almost nothing about the loop: at 81664 padded rows the shape probe measured
    a cold forward of 114 s against a warm 3.26 s (am. 123), the difference being kernel
    compilation.

    The same three conditions as for fl2va apply:

    1. **The warmup must be ref2va-shaped, with the same references.** `padded_len` runs 46080 to
       89856 across the three cases against t2va's 37888, and depends on the number and resolution
       of the references, so warming with different references warms nothing. Asserted per case.
    2. **The warmup must run at the same prompt length**, which for ref2va means the same
       presentation: a 2048 px image reference contributes ~4096 vision tokens and a video
       reference ~1008 per merged frame pair.
    3. **The embedding cache must be populated.** `warmup` runs with `use_prompt_cache=False`,
       so the priming call below is what makes the Encoder row comparable to t2va's. For ref2va
       that row is the conditioner plus the vision tower over up to 7168 patches per reference.
    """
    base = os.environ.get(WEIGHTS_ENV, DEFAULT_WEIGHTS)
    missing = [p for p in ("transformer_ref", "text_encoder", "vae", "audio_vae") if not os.path.isdir(f"{base}/{p}")]
    if missing:
        pytest.skip(f"MiniMax-H3 snapshot at {base} is missing {missing}; set {WEIGHTS_ENV}")

    from pathlib import Path

    media = Path(os.environ.get(REF2VA_MEDIA_ENV) or Path.home() / "h3_fl2va_artifacts" / "fl2va_first.mp4")
    if case != "one_image" and not media.is_file():
        pytest.skip(f"no reference video at {media}; set {REF2VA_MEDIA_ENV}")

    references = _ref2va_references(case)
    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=base, task="ref2va")

    # (1) and (2): warm at the real shape, with the real references and the real prompt.
    pipeline.warmup(
        prompt=PROMPT,
        references=references,
        num_frames=NUM_FRAMES_REF2VA,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
    )
    warm_padded_len = pipeline.last_padded_len

    # (3): prime the disk cache so the Encoder row means what it means for the other two rows. The
    # references have to be *prepared* first, because that is what the cache key digests.
    from ....pipelines.minimax_h3.references import prepare_references

    prepared, _ = prepare_references(references, NUM_FRAMES_REF2VA, pipeline.audio_sampling_rate)
    pipeline.encode_prompt(PROMPT, references=prepared, use_cache=True)

    output = pipeline(
        PROMPT,
        references=references,
        num_frames=NUM_FRAMES_REF2VA,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    assert pipeline.last_padded_len == warm_padded_len, (
        f"warmup ran at padded_len {warm_padded_len} but the measured call ran at "
        f"{pipeline.last_padded_len}; this number is not warm"
    )
    assert warm_padded_len == expected_padded_len, (
        f"{case} ran at padded_len {warm_padded_len}, not the gated {expected_padded_len}; this is a "
        "different request than the one the correctness gate covers"
    )

    rows = pipeline.last_timings
    total = sum(seconds for _, seconds in rows)
    num_forwards = NUM_INFERENCE_STEPS - 1
    aligned_frames = align_num_frames(NUM_FRAMES_REF2VA)

    logger.info(
        f"MEASUREMENT ref2va[{case}] fully warm | mesh 4x8 Blackhole, TP=4 axis 0 / SP=8 axis 1, ring, "
        f"2 links, l1_small_size 16384 | {WIDTH}x{HEIGHT}, {aligned_frames} frames @ {MINIMAX_H3_FPS} fps "
        f"({aligned_frames / MINIMAX_H3_FPS:.2f} s), {num_forwards} forwards, padded_len {warm_padded_len} "
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
    logger.info(f"  realtime factor    {total / (aligned_frames / MINIMAX_H3_FPS):8.1f} x  (compute / video seconds)")

    assert output.num_frames == aligned_frames


# -------------------------------------------------------------------- one transformer block, device time
#
# Device performance of one MiniMax-H3 transformer block at realistic 768P sequence lengths.
#
# Run under the Tracy device profiler, which emits the per-op CSV this test exists to produce:
#
#     scripts/run_safe_pytest.sh --profile \
#         models/tt_dit/tests/models/minimax_h3/test_performance_minimax_h3.py -k transformer_block
#
# The block runs twice per parameter: once to compile and populate the program cache, then once warm.
# The warm iteration is bracketed by `signpost("start")` / `signpost("stop")`, the convention the rest of
# tt_dit uses (see the LTX block test), so the report tool can isolate exactly it:
#
#     tt-perf-report <csv> --start-signpost start --end-signpost stop
#
# Both signposts matter. Without the closing one the analysed region runs to the end of the file and
# folds the output readback into the measurement.
#
# IMPORTANT: one profiled run yields one parameter's worth of ops. Running all three durations under a
# single `--profile` invocation produced a CSV containing only the first (verified against the recorded
# tensor shapes), so profile one duration at a time with `-k`:
#
#     for d in 5s_768p 10s_768p 15s_768p; do
#         scripts/run_safe_pytest.sh --profile <this file> -k $d
#     done
#
# No torch reference is run and no PCC is checked -- correctness lives in
# the transformer-block section of `test_transformer_minimax_h3.py`. This test only asserts the output is the right shape and
# finite, so that a broken run fails instead of quietly producing a CSV of nonsense.
#
# NOTE: `--profile` makes the tracy wrapper mask pytest's exit code, so the run reports PASS as long as
# profiling completed. Check the logged shapes and the CSV, not just the exit status.


# Real MiniMax-H3 block config.
HIDDEN_SIZE = 5376
NUM_HEADS = 56
HEAD_DIM = 128
FFN_DIM = 14336
TIME_EMBED_DIM = 2688
NORM_EPS = 1e-5
QK_NORM_EPS = 1e-5
ROPE_FREQ_DIM = 16
ROPE_THETA = 10000.0

PATCH_SIZE = (1, 2, 2)
# prod(spatial_downsample_factors) from the video VAE config: [2, 2, 2, 2, 1, 1].
VAE_SPATIAL_DOWNSAMPLE = 16
# Representative Qwen3-VL prompt length; the real value is prompt-dependent.
NUM_TEXT_TOKENS = 512
# 768P at 16:9. resolve_canvas_size caps the area at 768 * 1344, so this is the widest 768P canvas.
ASPECT = (16, 9)

TAG_VIDEO, TAG_TEXT, TAG_AUDIO = 0, 1, 2


def _packed_sizes(duration_s: float) -> dict:
    """Token counts for `duration_s` seconds of 768P video, from the pipeline's own helpers.

    Derived rather than hardcoded: frame alignment (`17n + 5`), the VAE's `5n + 2` latent
    frame count, the 40 Hz audio latent grid and the canvas area cap all come from `packing.py`, so
    these stay correct if the pipeline's constants change.
    """
    height, width = resolve_canvas_size(*ASPECT)
    tokens_per_latent_frame = (height // VAE_SPATIAL_DOWNSAMPLE // PATCH_SIZE[1]) * (
        width // VAE_SPATIAL_DOWNSAMPLE // PATCH_SIZE[2]
    )
    num_frames = align_num_frames(int(duration_s * MINIMAX_H3_FPS))
    latent_frames = video_latent_num_frames(num_frames)
    # Audio latents are *per channel*: `build_packed_sequence` lays the two channels out as
    # `num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS` consecutive rows, so the latent count is half
    # the row count. Getting this wrong does not fail anything here -- it just makes every packed
    # length in this file a few hundred rows short of the pipeline's, which silently moves the
    # per-device sequence length the SDPA chunk table is keyed on and stops the table from ever
    # hitting in production.
    num_audio = audio_latent_num_frames(num_frames)
    num_audio_rows = num_audio * MINIMAX_H3_AUDIO_CHANNELS
    num_video = latent_frames * tokens_per_latent_frame
    return {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "latent_frames": latent_frames,
        "grid_h": height // VAE_SPATIAL_DOWNSAMPLE // PATCH_SIZE[1],
        "grid_w": width // VAE_SPATIAL_DOWNSAMPLE // PATCH_SIZE[2],
        "num_video": num_video,
        "num_audio": num_audio,
        "num_audio_rows": num_audio_rows,
        "num_text": NUM_TEXT_TOKENS,
        "seq_len": NUM_TEXT_TOKENS + num_audio_rows + num_video,
    }


def _packed_metadata(sizes: dict, padded_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(position_ids, token_tags, timestep_indices)` for the padded packed sequence.

    Same layout as the correctness test: text, then audio, then video, with the first video frame at
    timestep 0 (clean conditioning) and everything else at timestep 1. The values do not affect
    device timing, but keeping them realistic avoids degenerate index patterns.
    """
    # Rows, not latents: the two audio channels each occupy their own row.
    n_text, n_audio, n_video = sizes["num_text"], sizes["num_audio_rows"], sizes["num_video"]
    frame = sizes["grid_h"] * sizes["grid_w"]

    def clock(n: int) -> torch.Tensor:
        return torch.stack([torch.arange(n), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)], -1)

    vt, vh, vw = torch.meshgrid(
        torch.arange(sizes["latent_frames"]),
        torch.arange(sizes["grid_h"]),
        torch.arange(sizes["grid_w"]),
        indexing="ij",
    )
    position_ids = torch.cat(
        [clock(n_text), clock(n_audio), torch.stack([vt.reshape(-1), vh.reshape(-1), vw.reshape(-1)], -1)]
    )
    tags = torch.cat(
        [
            torch.full((n_text,), TAG_TEXT, dtype=torch.long),
            torch.full((n_audio,), TAG_AUDIO, dtype=torch.long),
            torch.full((n_video,), TAG_VIDEO, dtype=torch.long),
        ]
    )
    timestep_indices = torch.cat(
        [
            torch.zeros(n_text, dtype=torch.long),
            torch.ones(n_audio, dtype=torch.long),
            torch.zeros(frame, dtype=torch.long),
            torch.ones(n_video - frame, dtype=torch.long),
        ]
    )

    pad = padded_len - position_ids.shape[0]
    if pad:
        position_ids = torch.cat([position_ids, torch.zeros((pad, 3), dtype=position_ids.dtype)])
        tags = torch.cat([tags, torch.zeros(pad, dtype=tags.dtype)])
        timestep_indices = torch.cat([timestep_indices, torch.zeros(pad, dtype=timestep_indices.dtype)])
    return position_ids, tags, timestep_indices


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    dit_mesh_params(),
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "duration_s",
    [
        pytest.param(5.0, id="5s_768p"),
        pytest.param(10.0, id="10s_768p"),
        pytest.param(15.0, id="15s_768p"),
    ],
)
# Which AdaLN path the profiled window exercises. `cachedadaln` is what the pipeline runs
# (`precompute_adaln` defaults on): the six modulation tables are built once and handed to every
# forward, so `_modulation_tables`' SiLU and projection matmul never appear in the step. `projadaln`
# projects `temb` per block, which is what a profile of this test measured before this axis existed.
#
# The projection is sequence-independent -- it is keyed on `num_timesteps` -- so it is a fixed cost
# per block that shrinks as a share of a longer sequence. Both are measured rather than argued.
#
# The tables here come from the block's own `_modulation_tables` run once outside the window rather
# than from a host-built checkpoint table. For a *timing* A/B that is equivalent: the op mix inside
# the window is identical to production's. It does leave `adaln_proj` resident, so this axis says
# nothing about the 26 GB/device the real cache exists to free.
@pytest.mark.parametrize(
    "adaln_cached",
    [pytest.param(False, id="projadaln"), pytest.param(True, id="cachedadaln")],
)
def test_minimax_h3_transformer_block_perf(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    duration_s: float,
    adaln_cached: bool,
    is_fsdp: bool,
    topology: ttnn.Topology,
    reset_seeds,
) -> None:
    skip_if_unsupported_num_links(mesh_device, num_links)

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    sizes = _packed_sizes(duration_s)
    seq_len = sizes["seq_len"]
    alignment = sp_factor * ttnn.TILE_SIZE
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    logger.info(
        f"{duration_s:g}s @ {sizes['height']}x{sizes['width']}: {sizes['num_frames']} frames -> "
        f"{sizes['latent_frames']} latent frames x {sizes['grid_h']}x{sizes['grid_w']} patches = "
        f"{sizes['num_video']} video + {sizes['num_audio']} audio + {sizes['num_text']} text "
        f"= seq_len {seq_len} (padded {padded_len}, {padded_len // sp_factor} rows/device)"
    )

    num_timesteps = 2
    position_ids, tags, timestep_indices = _packed_metadata(sizes, padded_len)
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + tags.clamp(min=0)

    # The reference block is built only to source a correctly-keyed random state dict; its forward is
    # never called. Weight *values* do not affect device timing.
    torch_block = TorchMiniMaxH3Block(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=HEAD_DIM,
        ffn_dim=FFN_DIM,
        time_embed_dim=TIME_EMBED_DIM,
        norm_eps=NORM_EPS,
        qk_norm_eps=QK_NORM_EPS,
    ).to(torch.float32)

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    with torch.no_grad():
        rope_cos, rope_sin = rope(position_ids)
    # The fused RoPE consumes head_dim-wide tables in the interleaved layout.
    rope_cos, rope_sin = prepare_rope_tables(rope_cos, rope_sin, HEAD_DIM)
    rotary_dim = rope_cos.shape[-1]

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    tt_block = MiniMaxH3TransformerBlock(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=2 * 3 * ROPE_FREQ_DIM,
        ffn_dim=FFN_DIM,
        time_embed_dim=TIME_EMBED_DIM,
        norm_eps=NORM_EPS,
        qk_norm_eps=QK_NORM_EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    tt_block.load_torch_state_dict(torch_block.state_dict())
    del torch_block

    tt_spatial = bf16_tensor_2dshard(
        torch.randn(1, 1, padded_len, HIDDEN_SIZE),
        device=mesh_device,
        shard_mapping={sp_axis: 2, tp_axis: 3},
    )
    tt_temb = from_torch(torch.randn(1, 1, num_timesteps, TIME_EMBED_DIM), device=mesh_device, dtype=ttnn.float32)
    tt_adaln = from_torch(
        adaln_indices.to(torch.int32).reshape(1, 1, 1, padded_len),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, sp_axis],
    )
    tt_rope_cos = from_torch(
        rope_cos.reshape(1, 1, padded_len, rotary_dim),
        device=mesh_device,
        dtype=ttnn.float32,
        mesh_axes=[..., sp_axis, None],
    )
    tt_rope_sin = from_torch(
        rope_sin.reshape(1, 1, padded_len, rotary_dim),
        device=mesh_device,
        dtype=ttnn.float32,
        mesh_axes=[..., sp_axis, None],
    )

    # Built once, OUTSIDE the profiled window, standing in for the host-built cache the pipeline
    # uploads once per request. `None` leaves `forward` on its own projection path.
    modulation_tables = tt_block._modulation_tables(tt_temb) if adaln_cached else None
    if adaln_cached:
        ttnn.synchronize_device(mesh_device)

    def run_block() -> ttnn.Tensor:
        out = tt_block(
            tt_spatial,
            # The true (unpadded) length: ring attention masks the pad tail via logical_n.
            N=seq_len,
            temb=tt_temb,
            adaln_indices=tt_adaln,
            rope_cos=tt_rope_cos,
            rope_sin=tt_rope_sin,
            modulation_tables=modulation_tables,
        )
        ttnn.synchronize_device(mesh_device)
        return out

    logger.info(
        f"AdaLN path: {'precomputed tables (as the pipeline runs)' if adaln_cached else 'per-block projection'}"
    )
    logger.info("iteration 1: compiling kernels and populating the program cache")
    run_block()

    logger.info("iteration 2: warm run (the profiled region)")
    signpost("start")
    tt_out = run_block()
    signpost("stop")

    assert tuple(tt_out.shape) == (
        1,
        1,
        padded_len // sp_factor,
        HIDDEN_SIZE // tp_factor,
    ), f"unexpected output shape {tuple(tt_out.shape)}"
    # Cheap guard that the profiled run actually computed something, without a reference.
    local = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
    assert torch.isfinite(local).all(), "block output contains NaN or Inf"
    logger.info(f"output {tuple(tt_out.shape)}, local shard std={local.std().item():.4f}")
