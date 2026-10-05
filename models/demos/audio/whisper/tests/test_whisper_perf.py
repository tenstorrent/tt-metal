# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""whisper-large-v3 latency on one device at batch 1, for short, medium and long speech.

RTF is the end-to-end wall time (waveform in, text out) divided by the audio's duration. Each clip is
measured after a warm-up pass that compiles every program and captures the traces, best of REPEATS.
The medium and long clips join the demo's own recordings with short gaps; all stay under the 30 s
window, since longer audio is cut there.
"""

import os
import time

import numpy as np
import pytest
from loguru import logger
from scipy.io import wavfile

from models.demos.audio.whisper.demo.demo import create_functional_whisper_for_conditional_generation_inference_pipeline
from models.demos.audio.whisper.tt.ttnn_optimized_functional_whisper import (
    WHISPER_L1_SMALL_SIZE,
    WHISPER_TRACE_REGION_SIZE,
)

MODEL_NAME = "openai/whisper-large-v3"
DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo", "dataset", "conditional_generation"
)
SAMPLING_RATE = 16000
GAP = np.zeros(int(0.3 * SAMPLING_RATE), dtype=np.float32)
CLIPS = (
    ("short", ["11150113890463037787.wav"]),
    ("medium", ["11150113890463037787.wav", "1298409023920250606.wav", "17566024285835266239.wav"]),
    (
        "long",
        [
            "17646385371758249908.wav",
            "17659141715436566244.wav",
            "17928171511082320095.wav",
            "17938133003986293739.wav",
        ],
    ),
)
REPEATS = 2  # best of, so one noisy run on a shared card does not decide the result

# Ceilings per clip: RTF (lower is faster) and time to first token, and a floor on decode speed.
MAX_RTF = {"short": 0.15, "medium": 0.11, "long": 0.09}
MAX_TTFT_S = 0.15
MIN_TOKENS_PER_S = 33.0


def join_recordings(names):
    parts = []
    for name in names:
        sr, data = wavfile.read(os.path.join(DATA, name))
        assert sr == SAMPLING_RATE and data.dtype == np.float32 and data.ndim == 1, (name, sr, data.dtype, data.shape)
        parts += [data, GAP]
    return np.concatenate(parts[:-1])


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": WHISPER_L1_SMALL_SIZE, "trace_region_size": WHISPER_TRACE_REGION_SIZE, "num_command_queues": 2}],
    indirect=True,
)
def test_whisper_large_v3_perf(mesh_device):
    pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
        mesh_device, MODEL_NAME, language="en", task="transcribe", batch_size_per_device=1
    )
    clips = [(label, join_recordings(names)) for label, names in CLIPS]
    for label, audio in clips:
        assert len(audio) / SAMPLING_RATE < 30.0, f"{label}: longer than the 30 s window"

    for _, audio in clips:  # warm-up: compile every program and capture the traces
        pipeline([(SAMPLING_RATE, audio)], return_perf_metrics=True)

    rows = []
    for label, audio in clips:
        best = None
        for _ in range(REPEATS):
            start = time.perf_counter()
            text, _, _, perf = pipeline([(SAMPLING_RATE, audio)], return_perf_metrics=True)
            wall = time.perf_counter() - start
            if best is None or wall < best[0]:
                best = (wall, text[0] if isinstance(text, list) else text, perf)
        wall, text, perf = best
        rows.append((label, len(audio) / SAMPLING_RATE, wall, text, perf))

    logger.info(f"{MODEL_NAME}, batch 1, one device, best of {REPEATS}")
    logger.info(f"{'clip':>7} {'audio':>6} {'encoder':>8} {'TTFT':>7} {'t/s/u':>6} {'total':>7} {'RTF':>6}")
    for label, seconds, wall, _, perf in rows:
        logger.info(
            f"{label:>7} {seconds:5.1f}s {perf.encoder_s * 1e3:6.0f}ms {perf.ttft * 1e3:5.0f}ms "
            f"{perf.decode_throughput:6.1f} {wall:6.3f}s {wall / seconds:6.3f}"
        )
    for label, seconds, wall, text, perf in rows:
        assert text.strip(), f"{label}: empty transcript"
        assert wall / seconds <= MAX_RTF[label], f"{label}: RTF {wall / seconds:.3f} above {MAX_RTF[label]}"
        assert perf.ttft <= MAX_TTFT_S, f"{label}: time to first token {perf.ttft:.3f} s above {MAX_TTFT_S} s"
        assert (
            perf.decode_throughput >= MIN_TOKENS_PER_S
        ), f"{label}: {perf.decode_throughput:.1f} tokens/s below {MIN_TOKENS_PER_S}"
