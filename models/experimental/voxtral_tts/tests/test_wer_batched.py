# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Intelligibility of BATCHED output: every voice, two medium-band sentences in its own language,
generated B users at a time by TtVoxtralBatchedPipeline, transcribed back with Whisper-large-v3 on
the CPU and scored per language against the single-user suite's ceilings (test_wer_languages.py).
Same transcriber, same scorer, same sentences, same ceilings; only the generator differs.

Env: WERB_BATCH (8), WERB_SENTENCES (2), WERB_RESULTS (json path), VOXTRAL_DEVICE_ID (0).
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.tests.reference_helpers import all_voices, needs_checkpoint  # noqa: E402
from models.experimental.voxtral_tts.tests.sentence_corpus import lang_of, wer_band  # noqa: E402
from models.experimental.voxtral_tts.tests.test_wer_languages import (  # noqa: E402
    CEILINGS,
    COLLAPSE,
    MAX_DEGENERATE,
    MAX_DEGENERATE_CELL,
    Asr,
    wer,
)
from models.experimental.voxtral_tts.tt.ttnn_voxtral_batched import TtVoxtralBatchedPipeline  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200), needs_checkpoint]

B = int(os.environ.get("WERB_BATCH", "8"))
N_SENT = int(os.environ.get("WERB_SENTENCES", "2"))
DEVICE_ID = int(os.environ.get("VOXTRAL_DEVICE_ID", "0"))
RESULTS_PATH = os.environ.get("WERB_RESULTS", "")


def test_wer_batched_every_voice():
    dev = open_device(device_id=DEVICE_ID)
    pipe = None
    try:
        pipe = TtVoxtralBatchedPipeline(dev, max_batch=B, max_seq_len=1024)
        pipe.warmup()
        voices = tuple(all_voices())
        jobs = []  # (text, voice, seed, lang)
        for v in voices:
            lang = lang_of(v)
            for text in wer_band(lang, "medium")[:N_SENT]:
                jobs.append((text, v, 0, lang))
        wavs = []
        for i in range(0, len(jobs), B):
            chunk = jobs[i : i + B]
            wavs.extend(pipe.synthesize_batch([(t, v, s) for t, v, s, _ in chunk]))
            print(
                f"[werb] batch {i // B + 1}: {len(chunk)} clips, {pipe.last_timings['decode_ms_per_frame']:.1f} ms/frame, "
                f"frames {pipe.last_timings['frames']}",
                flush=True,
            )
        asr = Asr()
        scores = {}
        for (text, v, s, lang), wav in zip(jobs, wavs):
            hyp = asr(wav.reshape(-1), lang)
            w = wer(text, hyp)
            scores[(v, text)] = (lang, w, hyp)
            print(f"[werb] {lang} {v:16s} WER {w:.3f}  | {hyp[:80]!r}", flush=True)
        by_lang = {}
        for (v, text), (lang, w, hyp) in scores.items():
            by_lang.setdefault(lang, []).append(w)
        summary, failures = {}, []
        for lang, ws in sorted(by_lang.items()):
            mean, worst = sum(ws) / len(ws), max(ws)
            degenerate = sum(1 for w in ws if w >= COLLAPSE)
            ceiling = CEILINGS[(lang, "medium")]
            ok = mean <= ceiling and degenerate <= MAX_DEGENERATE_CELL.get((lang, "medium"), MAX_DEGENERATE)
            summary[lang] = {
                "n": len(ws),
                "mean": mean,
                "worst": worst,
                "degenerate": degenerate,
                "ceiling": ceiling,
                "ok": ok,
            }
            print(
                f"[werb] {lang}: {len(ws)} clips, mean WER {mean:.4f} (ceiling {ceiling}), worst {worst:.3f}, degenerate {degenerate} -> {'OK' if ok else 'FAIL'}",
                flush=True,
            )
            if not ok:
                failures.append(lang)
        allw = [w for ws in by_lang.values() for w in ws]
        print(
            f"[werb] B={B}: {len(allw)} clips over {len(voices)} voices, overall mean WER {sum(allw) / len(allw):.4f}, perfect {sum(1 for w in allw if w == 0)}/{len(allw)}",
            flush=True,
        )
        if RESULTS_PATH:
            json.dump(
                {
                    "batch": B,
                    "summary": summary,
                    "clips": [
                        {"voice": v, "lang": l, "wer": w, "text": t, "hyp": h} for (v, t), (l, w, h) in scores.items()
                    ],
                },
                open(RESULTS_PATH, "w"),
                indent=2,
                ensure_ascii=False,
            )
        assert not failures, f"languages above their medium-band ceiling at B={B}: {failures}"
    finally:
        if pipe is not None:
            pipe.close()
        ttnn.close_device(dev)
