# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase A gates for TtVoxtralBatchedPipeline (B users per step).

  A1 independence/determinism: the same request in two different slots gives identical codes, and
     the whole batch is identical when run twice.
  A2 equivalence: for a few requests, the batched row and TtVoxtralPipeline (batch 1, same seed,
     so the same x0 noise sequence) agree frame for frame until a first divergence; the fraction of
     identical frames is reported and gated loosely (argmax over 8194 logits may flip on a 1e-4
     difference and the rows then legitimately part ways).
  A3 coverage: one batch of B rows covering all 20 voices in their own languages; every row stops
     on [END_AUDIO] before its cap and yields finite, non-trivial audio.
  A4 perf: traced ms per frame for B users against the 80 ms real-time budget.

Run (environment: /home/ttuser/nkira/voxtral_spike/run_spike.sh sets it):
  pytest -svv models/experimental/voxtral_tts/tests/test_batched_pipeline.py
Env: PHASEA_BATCH (32), PHASEA_MAX_SEQ (1024), PHASEA_RESULTS (json path), VOXTRAL_DEVICE_ID (0).
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts import frontend  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import FRAME_RATE  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import needs_checkpoint  # noqa: E402
from models.experimental.voxtral_tts.tests.sentence_corpus import first_sentence_for  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_batched import TtVoxtralBatchedPipeline  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline, open_device  # noqa: E402

pytestmark = [pytest.mark.slow, needs_checkpoint]

B = int(os.environ.get("PHASEA_BATCH", "32"))
MAX_SEQ = int(os.environ.get("PHASEA_MAX_SEQ", "1024"))
DEVICE_ID = int(os.environ.get("VOXTRAL_DEVICE_ID", "0"))
RESULTS_PATH = os.environ.get("PHASEA_RESULTS", "")
REAL_TIME_MS = 1000.0 / FRAME_RATE  # 80 ms per frame
MIN_IDENTICAL_FRACTION = 0.5  # loose: see A2 in the module docstring
RESULTS = {"batch": B, "max_seq": MAX_SEQ, "device_id": DEVICE_ID}


def _record(**kv):
    RESULTS.update(kv)
    if RESULTS_PATH:
        with open(RESULTS_PATH, "w") as fh:
            json.dump(RESULTS, fh, indent=2, default=str)


@pytest.fixture(scope="module")
def dev():
    d = open_device(device_id=DEVICE_ID)
    yield d
    ttnn.close_device(d)


@pytest.fixture(scope="module")
def pipe(dev):
    p = TtVoxtralBatchedPipeline(dev, max_batch=B, max_seq_len=MAX_SEQ)
    p.warmup(verbose=True)
    _record(warmup=p.warmed)
    yield p
    p.close()


@pytest.fixture(scope="module")
def single(dev, pipe):
    """Luka's batch-1 pipeline on the same device, for the equivalence check."""
    s = TtVoxtralPipeline(dev, max_seq_len=MAX_SEQ)
    s.warmup()
    yield s
    s.close()


def _requests_all_voices(model_dir):
    """B requests: every voice preset once with a sentence in its language, then repeats."""
    voices = frontend.voices(model_dir)
    reqs = [(first_sentence_for(v), v, i) for i, v in enumerate(voices)]
    while len(reqs) < B:
        t, v, s = reqs[len(reqs) % len(voices)]
        reqs.append((t, v, s + 100))
    return reqs[:B]


def test_a1_slots_are_independent_and_deterministic(pipe):
    reqs = _requests_all_voices(pipe.model_dir)
    # the same request in slot 0 and slot B//2
    reqs[B // 2] = reqs[0]
    f1 = pipe.generate_batch(reqs)
    t1 = dict(pipe.last_timings)
    f2 = pipe.generate_batch(reqs)
    same_slot = f1[0].shape == f1[B // 2].shape and bool(torch.equal(f1[0], f1[B // 2]))
    same_run = all(a.shape == b.shape and bool(torch.equal(a, b)) for a, b in zip(f1, f2))
    _record(a1_same_request_two_slots_identical=same_slot, a1_two_runs_identical=same_run, a1_timings=t1)
    print(f"\n[phaseA] A1 same request in two slots identical: {same_slot}; two runs identical: {same_run}")
    print(f"[phaseA] A1 frames per row: {t1['frames']}")
    assert same_slot, "the same request produced different codes in different slots"
    assert same_run, "the same batch produced different codes on a second run"


def test_a2_rows_match_the_single_user_pipeline(pipe, single):
    reqs = _requests_all_voices(pipe.model_dir)
    probe = [0, 1, 5, 12]  # four rows to compare against batch 1
    batched = pipe.generate_batch(reqs)
    rows = {}
    worst = 1.0
    for b in probe:
        text, voice, seed = reqs[b]
        embeds = frontend.build_prompt_embeds(text, voice, single.wb, model_dir=single.model_dir)
        single.backbone.reset()
        ref, _, _ = single.generate(embeds, max_frames=int(batched[b].shape[0]) + 50, seed=seed, verbose=False)
        n = min(ref.shape[0], batched[b].shape[0])
        eq = (ref[:n] == batched[b][:n]).all(dim=1)
        first_div = int((~eq).nonzero()[0]) if bool((~eq).any()) else n
        frac = float(eq.float().mean()) if n else 0.0
        rows[b] = {
            "voice": voice,
            "batched_frames": int(batched[b].shape[0]),
            "single_frames": int(ref.shape[0]),
            "first_divergence": first_div,
            "identical_fraction": frac,
        }
        worst = min(worst, frac)
        print(
            f"[phaseA] A2 row {b:2d} {voice:16s}: batched {batched[b].shape[0]:3d} frames, single "
            f"{ref.shape[0]:3d}, identical until frame {first_div}, identical fraction {frac:.2f}"
        )
    _record(a2_rows=rows, a2_worst_identical_fraction=worst)
    assert worst >= MIN_IDENTICAL_FRACTION, f"a batched row diverged early from the single-user pipeline: {rows}"


def test_a3_all_voices_in_one_batch_stop_naturally(pipe):
    reqs = _requests_all_voices(pipe.model_dir)
    wavs = pipe.synthesize_batch(reqs, verbose=True)
    t = dict(pipe.last_timings)
    bad = []
    for b, (w, (text, voice, _)) in enumerate(zip(wavs, reqs)):
        secs = w.shape[-1] / 24000.0
        ok = bool(torch.isfinite(w).all()) and secs > 0.5 and float(w.abs().max()) > 1e-3 and t["stopped_naturally"][b]
        if not ok:
            bad.append((b, voice, secs, t["stopped_naturally"][b]))
    _record(a3_frames=t["frames"], a3_audio_s=t["audio_s"], a3_stopped_naturally=t["stopped_naturally"], a3_bad=bad)
    print(f"\n[phaseA] A3 frames per row: {t['frames']}")
    print(f"[phaseA] A3 audio seconds per row: {[round(s, 1) for s in t['audio_s']]}")
    assert not bad, f"rows without natural stop or with bad audio: {bad}"


def test_a4_ms_per_frame_at_b_users(pipe):
    reqs = _requests_all_voices(pipe.model_dir)
    pipe.generate_batch(reqs, verbose=True)
    t = dict(pipe.last_timings)
    ms = t["decode_ms_per_frame"]
    dev_ms = pipe.frame_graph_ms()
    _record(a4_timings=t, a4_ms_per_frame=ms, a4_rtf_per_user=ms / REAL_TIME_MS, a4_device_only_ms_per_frame=dev_ms)
    print(
        f"\n[phaseA] A4 B={B}: prefill {t['prefill_s']:.2f}s for {B} users, {t['steps']} traced steps, "
        f"{ms:.1f} ms/frame for all users ({REAL_TIME_MS / ms:.2f}x real time per user), traced={t['traced']}"
    )
    print(
        f"[phaseA] A4 device-only frame graph: {dev_ms:.1f} ms/frame; host work per frame: {ms - dev_ms:.1f} ms "
        f"(what Phase B removes)"
    )
    assert ms < REAL_TIME_MS, f"{ms:.1f} ms/frame is slower than real time ({REAL_TIME_MS} ms)"
