# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Where an utterance spends its time, block by block, warm.

Warm means the second utterance onward: the first compiles kernels at whatever shapes it
sees and captures both traces, and those costs land wherever they happen to fall. So this
runs one utterance to warm the process and reports the next ones.

The frame loop is charged per block, and the blocks are the ones a change can move:

| block            | what it covers                                                      |
|---|---|
| `codec_head`     | the [2048, 3072] projection, plus the read that waits for the talker |
| `sample`         | host sampling for codebook 0, with the repetition penalty            |
| `predictor`      | the predictor's 15 traced steps and their 15 output heads            |
| `predictor_sample` | host sampling for codebooks 1 to 15                               |
| `embed`          | summing 16 codebook embeddings on host into the next prompt position  |
| `talker`         | the talker's 28-layer traced step                                   |

The last column is audio over wall clock, so above 1 is faster than real time. The demos
print it the same way round.

`profile=True` syncs the device at every split, so each block is charged for its own device
work rather than for whatever the next read waited on. The syncs cost a few percent of the
frame, which is the price of knowing where the time goes; `decode_s` is the honest total.

Ceilings are per block and deliberately loose. This runs on a shared card, so anything
tight enough to catch a few percent would fail on load instead. What the table is for is
the shape of the split: if one block moves, it says which.

Run:
    pytest -svv models/demos/audio/qwen3_tts/tests/perf/test_perf.py
"""

import os

import pytest

from models.demos.audio.qwen3_tts import frontend, weights
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import Qwen3TTSPipeline

CUSTOM_VOICE_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
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


def _clear_caches():
    for cached in (
        weights.checkpoint_dir,
        weights._model_config_json,
        weights.codec_dir,
        weights._codec_config_json,
        frontend.tokenizer,
        frontend.special_tokens,
        frontend.language_ids,
    ):
        cached.cache_clear()


@pytest.fixture(scope="module", autouse=True)
def custom_voice_checkpoint():
    """CustomVoice, so the measurement needs no reference clip to get started."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(CUSTOM_VOICE_REPO)
    previous = os.environ.get("QWEN3_TTS_CKPT")
    os.environ["QWEN3_TTS_CKPT"] = path
    _clear_caches()
    yield path
    if previous is None:
        os.environ.pop("QWEN3_TTS_CKPT", None)
    else:
        os.environ["QWEN3_TTS_CKPT"] = previous
    _clear_caches()


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

    Every utterance runs twice off the same seed and the second one is reported. The
    codec decoder compiles a program per frame-count bucket and holds it for the life of
    the device, so a length nothing has decoded yet pays a build of tens of seconds that
    lands in `codec`. Same seed, same frames, same bucket: the second run is the warm one.
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
