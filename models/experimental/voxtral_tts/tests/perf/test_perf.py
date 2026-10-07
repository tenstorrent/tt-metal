# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where a request spends its time, per stage, across utterance lengths.

Warm figures (warmup excluded). Ceilings are loose smoke checks; the regression detector is
`tools/quality_report.py --compare` (bring-up tooling, see the README).

Run:
    pytest -svv models/experimental/voxtral_tts/tests/perf/test_perf.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.tests.reference_helpers import fixture_embeds, needs_checkpoint  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import (  # noqa: E402
    FRAME_RATE,
    TtVoxtralPipeline,
    open_device,
)

pytestmark = needs_checkpoint

CASES = ((14, "short"), (2, "long"))
REPEATS = 2  # best of, so one noisy run on a shared card does not decide the result
MAX_FRAMES = 520

MAX_PREFILL_S = 2.0
MAX_DECODE_MS_PER_FRAME_LONG = 40.0  # the long case amortises the capture
MAX_DECODE_MS_PER_FRAME_ANY = 70.0  # the short case carries the whole capture
MAX_CODEC_S = 2.0
MIN_RTF = 1.2  # short utterances pay prefill+capture over little audio


@pytest.fixture(scope="module")
def pipe():
    d = open_device()
    p = TtVoxtralPipeline(d)
    p.warmup()
    yield p
    p.close()
    ttnn.close_device(d)


def _best_of(pipe, embeds, max_frames=MAX_FRAMES):
    """-> the timings of the fastest of REPEATS runs."""
    best = None
    for _ in range(REPEATS):
        pipe.backbone.reset()
        frames, _, _ = pipe.generate(embeds, max_frames=max_frames, seed=0, verbose=False)
        pipe.decode(frames)
        t = dict(pipe.last_timings)
        total = t["prefill_s"] + t["decode_s"] + t.get("codec_s", 0.0)
        if best is None or total < best[0]:
            best = (total, t)
    return best


@pytest.mark.slow
@pytest.mark.timeout(1800)  # warm-up compiles every kernel on an empty kernel cache
def test_perf(pipe):
    rows, failed = [], []
    for ci, name in CASES:
        embeds, case = fixture_embeds(ci, pipe.wb)
        total, t = _best_of(pipe, embeds)
        n = t["frames"]
        audio_s = n / FRAME_RATE
        rtf = audio_s / max(total, 1e-9)
        rows.append(
            (
                name,
                n,
                audio_s,
                t["prefill_s"],
                t["decode_s"],
                t["decode_ms_per_frame"],
                t.get("codec_s", 0.0),
                total,
                rtf,
                t["traced"],
            )
        )
        if t["prefill_s"] > MAX_PREFILL_S:
            failed.append(f"{name}: prefill {t['prefill_s']:.2f}s > {MAX_PREFILL_S}s")
        ceil = MAX_DECODE_MS_PER_FRAME_LONG if name == "long" else MAX_DECODE_MS_PER_FRAME_ANY
        if t["decode_ms_per_frame"] > ceil:
            failed.append(f"{name}: decode {t['decode_ms_per_frame']:.2f} ms/frame > {ceil}")
        if t.get("codec_s", 0.0) > MAX_CODEC_S:
            failed.append(f"{name}: codec {t['codec_s']:.2f}s > {MAX_CODEC_S}s")
        if rtf < MIN_RTF:
            failed.append(f"{name}: RTF {rtf:.2f}x < {MIN_RTF}x")

    print(f"\n  best of {REPEATS}, warm; decode_s includes the one-time trace capture")
    print(
        f"  {'utterance':>9} {'frames':>7} {'audio':>7} {'prefill':>8} {'decode':>8} "
        f"{'ms/frame':>9} {'codec':>7} {'total':>7} {'RTF':>7} {'traced':>7}"
    )
    for r in rows:
        print(
            f"  {r[0]:>9} {r[1]:>7} {r[2]:>6.1f}s {r[3]:>7.2f}s {r[4]:>7.2f}s "
            f"{r[5]:>9.2f} {r[6]:>6.2f}s {r[7]:>6.2f}s {r[8]:>6.2f}x {str(r[9]):>7}"
        )
    mspf = [r[5] for r in rows]
    print(
        f"\n  ms/frame spread across lengths: {min(mspf):.2f} .. {max(mspf):.2f} "
        f"({max(mspf) - min(mspf):.2f} ms) -- should be nearly flat"
    )
    assert not failed, "per-stage ceilings exceeded:\n    " + "\n    ".join(failed)
