# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end serving numbers per configuration, measured the way a server would see them: one
call that synthesizes B requests (same medium sentence, seeds 0..B-1) and returns B waveforms.

  B=1   TtVoxtralPipeline.synthesize (the single-user path)
  B>1   TtVoxtralBatchedPipeline.synthesize_batch

Reports per configuration: wall per call, audio seconds produced, aggregate real-time factor
(audio / wall), per-user real-time factor, frames per second, prefill share, codec share, and the
single-request latency for a ~10 s clip at B=1. Jerry's README numbers are printed alongside.

Env: PM_BATCHES ("8,16,32"), PM_REPEATS (2, best of), PM_RESULTS (json path), VOXTRAL_DEVICE_ID (0).
"""

import json
import os
import time

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.tests.reference_helpers import needs_checkpoint  # noqa: E402
from models.experimental.voxtral_tts.tests.sentence_corpus import wer_band  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_batched import TtVoxtralBatchedPipeline  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline, open_device  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600), needs_checkpoint]

BATCHES = [int(b) for b in os.environ.get("PM_BATCHES", "8,16,32").split(",") if b]
REPEATS = int(os.environ.get("PM_REPEATS", "2"))
DEVICE_ID = int(os.environ.get("VOXTRAL_DEVICE_ID", "0"))
RESULTS_PATH = os.environ.get("PM_RESULTS", "")
VOICE = "neutral_male"
SR = 24000
# PR #58516 README (one p150 chip, batch 32, trace + 1 CQ, tip 2fa38a243a1): 63.0 ms per frame for 32
# users, 508 frames/s. It is the SUM of two separately traced stages (decode 37.2 + acoustic 25.8), not
# a measured frame loop; our ms per frame is one measured trace replay per frame.
JERRY = {"ms_per_frame_32": 63.0, "frames_per_s_32": 508.0, "rtf_per_user_32": 1000.0 / 12.5 / 63.0}


def _row(name, wall, audio_s, frames, prefill_s, codec_s, users):
    agg = audio_s / wall
    return {
        "config": name,
        "users": users,
        "wall_s": wall,
        "audio_s": audio_s,
        "rtf_aggregate": agg,
        "rtf_per_user": agg / users,
        "frames_per_s": frames / wall,
        "prefill_s": prefill_s,
        "codec_s": codec_s,
    }


def _print(rows):
    print(
        f"\n{'config':14s} {'users':>5s} {'wall s':>7s} {'audio s':>8s} {'RTF agg':>8s} {'RTF/user':>9s} {'frames/s':>9s} {'prefill s':>9s} {'codec s':>8s}"
    )
    for r in rows:
        print(
            f"{r['config']:14s} {r['users']:5d} {r['wall_s']:7.2f} {r['audio_s']:8.1f} {r['rtf_aggregate']:8.2f} {r['rtf_per_user']:9.2f} {r['frames_per_s']:9.0f} {r['prefill_s']:9.2f} {r['codec_s']:8.2f}"
        )
    print(
        f"{'Jerry PR, 32':14s} {32:5d} {'':>7s} {'':>8s} {32 * JERRY['rtf_per_user_32']:8.2f} {JERRY['rtf_per_user_32']:9.2f} {JERRY['frames_per_s_32']:9.0f}   (README: 63.0 ms per 32-user frame = decode 37.2 + acoustic 25.8, stage sum)"
    )


# Every configuration opens its own device and builds exactly ONE pipeline, closed with the device before
# the next starts. Two pipelines on one chip in one process (a second one warming up its codec while the
# first still held device memory -- pipeline.close() frees only the trace) hung the chip on 2026-10-02.
CONFIGS = ["single"] + BATCHES
ROWS = []


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    if ROWS:
        _print(ROWS)
        if RESULTS_PATH:
            json.dump({"rows": ROWS, "jerry": JERRY}, open(RESULTS_PATH, "w"), indent=2)


def _single(dev, text, long_text):
    single = TtVoxtralPipeline(dev, max_seq_len=1024)
    single.warmup()
    best = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        wav = single.synthesize(text, VOICE, seed=0)
        wall = time.perf_counter() - t0
        t = single.last_timings
        if best is None or wall < best[0]:
            best = (wall, wav.shape[-1] / SR, t["frames"], t["prefill_s"], t.get("codec_s", 0.0))
    rows = [_row("single B=1", *best, users=1)]
    t0 = time.perf_counter()
    wav = single.synthesize(long_text, VOICE, seed=0)
    long_wall = time.perf_counter() - t0
    t = single.last_timings
    rows.append(
        _row("single long", long_wall, wav.shape[-1] / SR, t["frames"], t["prefill_s"], t.get("codec_s", 0.0), users=1)
    )
    single.close()
    return rows


def _batched(dev, B, text):
    pipe = TtVoxtralBatchedPipeline(dev, max_batch=B, max_seq_len=1024)
    pipe.warmup()
    reqs = [(text, VOICE, s) for s in range(B)]
    best = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        wavs = pipe.synthesize_batch(reqs)
        wall = time.perf_counter() - t0
        t = pipe.last_timings
        audio = sum(w.shape[-1] for w in wavs) / SR
        if best is None or wall < best[0]:
            best = (
                wall,
                audio,
                sum(t["frames"]),
                t["prefill_s"],
                t.get("codec_s", 0.0),
                t["decode_ms_per_frame"],
                t["steps"],
            )
    wall, audio, frames, pre, codec, ms_frame, steps = best
    r = _row(f"batched B={B}", wall, audio, frames, pre, codec, users=B)
    r["decode_ms_per_frame"] = ms_frame
    r["steps"] = steps
    print(
        f"[pm] B={B}: {ms_frame:.1f} ms per frame for all users ({(1000 / 12.5) / ms_frame:.2f}x real time each), {steps} steps, trace capture included in wall",
        flush=True,
    )
    pipe.close()
    return [r]


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"B{c}" if c != "single" else "single")
def test_perf_matrix(config):
    text = wer_band("en", "medium")[0]
    long_text = " ".join(wer_band("en", "long")[:2])
    dev = open_device(device_id=DEVICE_ID)
    try:
        rows = _single(dev, text, long_text) if config == "single" else _batched(dev, int(config), text)
        ROWS.extend(rows)
        assert rows
    finally:
        ttnn.close_device(dev)  # frees every buffer this configuration allocated
