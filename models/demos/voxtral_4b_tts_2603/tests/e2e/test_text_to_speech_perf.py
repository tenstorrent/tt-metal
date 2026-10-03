# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PERFORMANCE test for the `text_to_speech` pipeline of `mistralai/Voxtral-4B-TTS-2603` (no PCC).

Each of the four stages in `PIPELINE_STAGES` (prefill / decode / acoustic / vocode) is timed through the
pipeline's own trace contract: `<stage>_trace_inputs()` -> `<stage>_trace_setup(inputs)` stages real
inputs in persistent device buffers, then `<stage>_trace_step()` -- one fixed-shape, host-op-free step --
is warmed up, captured as a trace and replayed back to back on one command queue. The per-stage time is
the average replay.

What one step covers, at batch B = 32:
  prefill   the voiced speech request (voice block + text + controls) through the 26-layer text stack
  decode    ONE decode step of the text stack for all B rows (one audio frame)
  acoustic  ONE frame of the acoustic sampler (7 Euler steps with CFG) for all B rows
  vocode    the codec decoder over a `VOXTRAL_TRACE_VOCODE_C`-frame chunk (default 32) for all B rows

An audio frame is 80 ms of audio (12.5 frames/s) and costs one decode step plus one acoustic step, so
`ms per frame = decode + acoustic`; prefill runs once per request and vocode once per utterance.
"""
from __future__ import annotations

import os
import time

import pytest

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common
from models.demos.voxtral_4b_tts_2603.tt.pipeline import PIPELINE_STAGES, build_pipeline

pytestmark = pytest.mark.timeout(3600)

WARMUP_STEPS = 3
REPLAYS = int(os.environ.get("TT_PERF_REPLAYS", "16"))
DEVICE_ID = int(os.environ.get("TT_PERF_DEVICE_ID", os.environ.get("VOXTRAL_DEVICE_ID", "0")))
# The demo's own device configuration (demo_text_to_speech.py / device_session.py).
L1_SMALL_SIZE = 24576
TRACE_REGION_SIZE = int(os.environ.get("TT_PERF_TRACE_REGION", str(200 * 1024 * 1024)))
REAL_TIME_FRAMES_PER_S = 12.5
# The README's expected numbers (one Blackhole chip, batch 32, trace + 1 CQ), in ms per traced step. A
# run fails when any stage, or the per-frame cost, is more than PERF_MARGIN slower than this.
EXPECTED_MS = {"prefill": 85.6, "decode": 37.2, "acoustic": 25.8, "vocode": 75.3}
PERF_MARGIN = float(os.environ.get("TT_PERF_MARGIN", "0.10"))


def _time_stage(device, pipe, stage: str) -> float:
    """Average ms of one traced `<stage>_trace_step`, replayed REPLAYS times on cq 0."""
    getattr(pipe, f"{stage}_trace_setup")(getattr(pipe, f"{stage}_trace_inputs")())
    step = getattr(pipe, f"{stage}_trace_step")
    for _ in range(WARMUP_STEPS):
        step()
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    step()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    try:
        start = time.perf_counter()
        for _ in range(REPLAYS):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        return (time.perf_counter() - start) / REPLAYS * 1000.0
    finally:
        ttnn.release_trace(device, trace_id)


def test_text_to_speech_perf():
    device = ttnn.open_device(
        device_id=DEVICE_ID,
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=TRACE_REGION_SIZE,
        num_command_queues=1,
    )
    try:
        pipe = build_pipeline(device, model=common.load_reference_model())
        batch = pipe.batch
        ms = {stage: _time_stage(device, pipe, stage) for stage in PIPELINE_STAGES}
    finally:
        ttnn.close_device(device)

    frame_ms = ms["decode"] + ms["acoustic"]
    per_user = 1000.0 / frame_ms
    print(f"\nVoxtral-4B-TTS perf, batch {batch}, trace + 1 CQ, {REPLAYS} replays per stage")
    for stage in PIPELINE_STAGES:
        print(f"  {stage:9s} {ms[stage]:8.2f} ms")
    print(f"  ms per audio frame (decode + acoustic): {frame_ms:.2f}")
    print(
        f"  frames/s per user: {per_user:.2f} (real time = {REAL_TIME_FRAMES_PER_S}); "
        f"frames/s total: {per_user * batch:.1f}; {per_user / REAL_TIME_FRAMES_PER_S:.2f}x real time"
    )
    assert set(ms) == set(PIPELINE_STAGES) and all(v > 0 for v in ms.values())
    limits = {stage: EXPECTED_MS[stage] * (1 + PERF_MARGIN) for stage in PIPELINE_STAGES}
    slow = {stage: round(ms[stage], 2) for stage in PIPELINE_STAGES if ms[stage] > limits[stage]}
    assert not slow, f"stages slower than expected +{PERF_MARGIN:.0%}: {slow} (limits {limits})"
    frame_limit = (EXPECTED_MS["decode"] + EXPECTED_MS["acoustic"]) * (1 + PERF_MARGIN)
    assert frame_ms <= frame_limit, f"{frame_ms:.2f} ms per audio frame exceeds {frame_limit:.2f}"
