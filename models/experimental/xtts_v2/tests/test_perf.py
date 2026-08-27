# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Where a request spends its time, per stage, across utterance lengths.

Warm figures: warmup and compute_voice are one-time and excluded, so this is what a caller waits for
the second request onward. The three stages do not sum to the total -- the gap is tokenizing,
assembling the prompt, and the host-side sampling between decode steps -- and seeing that gap is
half the point of splitting the columns.

The stage columns exist so a regression localises itself. decode/token is the one to watch: it
should barely move with utterance length, since each step does the same work over a slightly longer
cache. RTF rises with length because the vocoder is a fixed cost per bucket, so short utterances
carry proportionally more of it.

Ceilings are per stage and deliberately loose. This runs on a shared card, so anything tight enough
to catch a few percent would fail on load instead.

Run:
    pytest -svv models/experimental/xtts_v2/tests/test_perf.py
"""
import time

import pytest

from models.experimental.xtts_v2.tests.language_corpus import WER_SENTENCES
from models.experimental.xtts_v2.tests.test_wer import _representative, _speakers
from models.experimental.xtts_v2.tt.ttnn_xtts_model import OUTPUT_SR, XttsV2

EN = WER_SENTENCES["en"]
CASES = (("short", EN[0]), ("medium", EN[2]), ("long", " ".join(EN[:3])))
REPEATS = 3  # best of, so one noisy run on a shared card does not decide the result

# Loose enough that only a real regression trips them, from the measured spread.
MAX_PREFILL_MS = 30
MAX_DECODE_MS_PER_TOKEN = 10
MIN_RTF = 4.0


def _row(tts, voice, text):
    """Best of REPEATS -> the timings of the fastest run."""
    best = None
    for _ in range(REPEATS):
        t0 = time.time()
        wav = tts.generate(text, voice, seed=0)
        wall = time.time() - t0
        if best is None or wall < best[0]:
            best = (wall, wav.shape[-1] / OUTPUT_SR, dict(tts.last_timings))
    return best


@pytest.mark.slow
def test_perf():
    tts = XttsV2()
    tts.warmup()
    voice = next(iter(_representative(_speakers(tts.ckpt_path), 1).values()))
    rows, failed = [], []
    try:
        for name, text in CASES:
            wall, secs, t = _row(tts, voice, text)
            rows.append(
                f"  {name + f' ({secs:.1f} s)':16s} {t['codes']:5d} {t['prefill_s']:8.2f}s "
                f"{t['decode_s']:8.2f}s {t['decode_ms_per_token']:10.2f} ms {t['vocoder_s']:8.2f}s "
                f"{wall:8.2f}s {secs / wall:7.2f}x"
            )
            if t["prefill_s"] * 1000 > MAX_PREFILL_MS:
                failed.append(f"{name}: prefill {t['prefill_s'] * 1000:.0f} ms")
            if t["decode_ms_per_token"] > MAX_DECODE_MS_PER_TOKEN:
                failed.append(f"{name}: decode {t['decode_ms_per_token']:.2f} ms/token")
            if secs / wall < MIN_RTF:
                failed.append(f"{name}: RTF {secs / wall:.2f}x")
    finally:
        tts.close()
    print(
        f"\n  {'utterance':16s} {'codes':>5s} {'prefill':>9s} {'decode':>9s} {'decode/token':>13s} "
        f"{'vocoder':>9s} {'total':>9s} {'RTF':>8s}"
    )
    print("\n".join(rows))
    assert not failed, "; ".join(failed)


if __name__ == "__main__":
    test_perf()
