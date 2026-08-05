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
from loguru import logger

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....utils.test import ring_params_req_exact_devices

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

# Dialogue-heavy on purpose: t2va generates a soundtrack, so a prompt with a spoken line exercises the
# audio path rather than leaving it to ambience.
PROMPT = (
    "Jerry, George, Elaine and Kramer are crowded into a red vinyl booth at a bright New York diner, "
    "coffee cups and menus on the table. George leans in and says, 'These video generation models can "
    "conjure a whole city out of nothing, and they still can't count fingers. The machine watched every "
    "movie ever made and concluded people have, what, nine? Eleven on a good day?' Elaine laughs into "
    "her coffee, Jerry shrugs with both palms up, and Kramer bursts through the door behind them."
)

WEIGHTS_ENV = "MINIMAX_H3_DIFFUSERS_DIR"
DEFAULT_WEIGHTS = "/data/cglagovich/MiniMax-H3-diffusers"

# Generous: a regression bar, not a target. Measured fully-warm total is well inside this, and the
# point is to notice a collapse (a lost cache, a fallback kernel) rather than to police seconds.
EXPECTED_TOTAL_S = 400.0

MESH_4X8 = [
    pytest.param(
        (4, 8),
        {**ring_params_req_exact_devices, "l1_small_size": 65536},
        id="4x8",
    )
]


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("num_frames", DURATIONS)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
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
    pipeline.warmup(num_frames=num_frames, height=HEIGHT, width=WIDTH, num_inference_steps=NUM_INFERENCE_STEPS)

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

    logger.info(
        f"MEASUREMENT t2va fully warm | mesh 4x8 Blackhole, TP=4 axis 0 / SP=8 axis 1, ring, 2 links "
        f"| {WIDTH}x{HEIGHT}, {aligned_frames} frames @ {MINIMAX_H3_FPS} fps "
        f"({aligned_frames / MINIMAX_H3_FPS:.2f} s), {num_forwards} forwards "
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
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_fl2va_warm_latency(mesh_device, reset_seeds):
    """Fully-warm `fl2va` latency, by the same method as the t2va row so the two are comparable.

    Three things have to be right or the number means nothing, and each has bitten a measurement in
    this campaign or its sibling:

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
