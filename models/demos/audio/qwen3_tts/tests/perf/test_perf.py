# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Where an utterance spends its time, block by block, warm.

Warm is the second utterance onward, so the first compiles kernels and captures traces,
and each case runs twice off one seed because the codec compiles per frame count.

The frame loop is charged per block. `profile=True` syncs the device at every split so
each block is charged for its own work, which costs a few percent: `decode_s` is the
honest total. The last column is audio over wall clock, so above 1 is faster than real
time. Ceilings are loose because the card is shared; the table's job is to say which
block moved.

Run:
    pytest -svv models/demos/audio/qwen3_tts/tests/perf/test_perf.py
"""

import pytest

from models.demos.audio.qwen3_tts.tests.checkpoints import use_release
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import Qwen3TTSPipeline

DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 90_000_000}]

SPEAKER = "ryan"
LANGUAGE = "English"
WARMUP = "Warming the kernels up."
CASES = (
    ("short", "The kettle is on."),
    ("medium", "The kettle is on, and the rain has not let up since yesterday morning."),
    (
        "long",
        "The kettle is on, and the rain has not let up since yesterday morning. "
        "Nobody has come down the lane all day, which is unusual for a Tuesday. "
        "I have put the radio on for the company of it.",
    ),
)
BLOCKS = ("codec_head", "sample", "predictor", "predictor_sample", "embed", "talker")
SEED = 0

MAX_MS_PER_FRAME = 60.0
MAX_PREFILL_S = 4.0
# Per frame decoded, padding included: a short utterance throws most of its bucket away.
MAX_CODEC_MS_PER_FRAME = 8.0


@pytest.fixture(scope="module", autouse=True)
def custom_voice_checkpoint():
    """CustomVoice at the ambient size, so the measurement needs no reference clip."""
    yield from use_release("custom_voice")


def _row(name, timings, seconds):
    frames = timings["frames"]
    blocks = timings["blocks"]
    per_frame = [1000 * blocks.get(key, 0.0) / max(frames, 1) for key in BLOCKS]
    return (
        f"  {name:8s} {seconds:6.1f}s {frames:6d} {timings['prefill_s']:7.2f}s {timings['capture_s']:7.2f}s "
        + " ".join(f"{value:7.2f}" for value in per_frame)
        + f" {timings['ms_per_frame']:8.2f} {timings['codec_s']:7.2f}s {timings['total_s']:7.2f}s "
        f"{seconds / timings['total_s']:8.2f}x"
    )


@pytest.mark.slow
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_perf(device):
    """One warmup utterance, then a table of the warm ones.

    Each case runs twice off the same seed: same frames, same bucket, so the second run pays
    no kernel build.
    """
    pipeline = Qwen3TTSPipeline(device, max_frames=400, seed=SEED, profile=True)
    pipeline.generate(WARMUP, speaker=SPEAKER, language=LANGUAGE)

    rows, failed = [], []
    for name, text in CASES:
        for _ in range(2):
            pipeline.reseed(SEED)
            waveform, _ = pipeline.generate(text, speaker=SPEAKER, language=LANGUAGE)
        seconds = waveform.shape[1] / 24000
        timings = dict(pipeline.last_timings)
        timings["total_s"] = timings["prefill_s"] + timings["capture_s"] + timings["decode_s"] + timings["codec_s"]
        rows.append(_row(name, timings, seconds))

        if timings["ms_per_frame"] > MAX_MS_PER_FRAME:
            failed.append(f"{name}: {timings['ms_per_frame']:.1f} ms/frame")
        if timings["prefill_s"] > MAX_PREFILL_S:
            failed.append(f"{name}: prefill {timings['prefill_s']:.2f} s")
        codec_ms = 1000 * timings["codec_s"] / max(timings["codec_padded_frames"], 1)
        if codec_ms > MAX_CODEC_MS_PER_FRAME:
            failed.append(f"{name}: codec {codec_ms:.1f} ms per decoded frame")

    header = " ".join(f"{key:>7s}" for key in BLOCKS)
    print(
        f"\n  {'':8s} {'audio':>7s} {'frames':>6s} {'prefill':>8s} {'capture':>8s} {header} {'ms/frame':>8s} "
        f"{'codec':>8s} {'total':>8s} {'faster':>9s}"
    )
    print("  " + "-" * 118)
    print("\n".join(rows))
    print("\n  per-frame block columns are milliseconds, averaged over the utterance's frames")
    assert not failed, "; ".join(failed)
