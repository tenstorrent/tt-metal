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
    """Luka's batch-1 pipeline on the same device, for the equivalence check. A2 compares codes
    only, so this pipeline never decodes audio: it borrows the batched pipeline's codec (its own is
    dropped) and skips the codec warmup. A second codec instance warming up on a chip that already
    holds the batched pipeline hung the device (2026-10-01, BH Galaxy chip 03:00.0: stuck in the
    512-frame bucket after 128..384 completed; the same buckets never hang with one codec per chip,
    which is also what production runs)."""
    s = TtVoxtralPipeline(dev, max_seq_len=MAX_SEQ)
    s.codec = pipe.codec
    s.warmup(codec=False)
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
    # The same request in several slots, including adjacent ones and ones that are not multiples of 8:
    # slots 0 and B//2 alone happened to agree while rows b != 0 mod 8 did not (2026-10-01, the device
    # loop's next-frame embedding grouped each row's sum differently).
    copies = sorted({0, min(3, B - 1), B // 2, B - 1})
    for s in copies[1:]:
        reqs[s] = reqs[0]
    f1 = pipe.generate_batch(reqs)
    t1 = dict(pipe.last_timings)
    f2 = pipe.generate_batch(reqs)
    same_slot = all(f1[0].shape == f1[s].shape and bool(torch.equal(f1[0], f1[s])) for s in copies[1:])
    same_run = all(a.shape == b.shape and bool(torch.equal(a, b)) for a, b in zip(f1, f2))
    _record(a1_same_request_two_slots_identical=same_slot, a1_two_runs_identical=same_run, a1_timings=t1)
    print(f"\n[phaseA] A1 same request in slots {copies} identical: {same_slot}; two runs identical: {same_run}")
    print(f"[phaseA] A1 frames per row: {t1['frames']}")
    assert same_slot, "the same request produced different codes in different slots"
    assert same_run, "the same batch produced different codes on a second run"


def test_a2_rows_match_the_single_user_pipeline(pipe, single):
    """Teacher-forced: both pipelines get the SAME code history and the SAME noise at every frame,
    and we compare what each produces for that frame (semantic code, 36 acoustic codes). A
    free-running comparison is the wrong gate: codes are 36 values rounded onto 21 levels 0.1
    apart, so a 1e-3 difference in the hidden state flips a code near a boundary in a fair share
    of frames, after which the two trajectories are legitimately different utterances. Frame 0
    (prefill hidden -> flow) must match exactly: same prefill, same noise."""
    from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref

    reqs = _requests_all_voices(pipe.model_dir)
    probe = [p for p in (0, 1, 5, 12) if p < B]  # rows that exist at this batch (B=1: row 0 only)
    K = 20
    # The history both sides are fed: the single-user pipeline's own free run for each probe row.
    history, embeds, lens = {}, {}, {}
    for b in probe:
        text, voice, seed = reqs[b]
        e = frontend.build_prompt_embeds(text, voice, single.wb, model_dir=single.model_dir)
        single.backbone.reset()
        fr, _, _ = single.generate(e, max_frames=K + 10, seed=seed, verbose=False)
        history[b], embeds[b], lens[b] = fr, e, e.shape[1]
    K = min(K, min(int(h.shape[0]) for h in history.values()))

    # Batched side: prefill the probe rows (other slots get row 0's prompt), then step with the history.
    bp = pipe.backbone
    h0 = {}
    for slot in range(B):
        b = slot if slot in history else probe[0]
        h0[slot] = bp.prefill(embeds[b], last_only=True, user=slot)[0, 0]
    rows_out = {}
    frame0_ok = True
    # noise per slot: probe rows use their own seed stream, other slots reuse row probe[0]'s
    noise = {s: pipe.noise_for(reqs[s if s in history else probe[0]][2], K) for s in range(B)}
    X0 = lambda t: torch.stack([noise[s][t] for s in range(B)], dim=0)  # [B,36]
    H0 = torch.stack([h0[s] for s in range(B)], dim=0)  # [B,3072]
    # frame 0: the batched flow at B rows (what generate_batch runs) vs the single-user flow at 1 row
    cB_all = pipe.flow(H0, x_0=X0(0))
    for b in probe:
        text, voice, seed = reqs[b]
        c1 = single.flow(h0[b].reshape(1, -1), x_0=noise[b][0:1])
        same = bool(torch.equal(c1[0], cB_all[b]))
        frame0_codes_eq = float((c1[0] == cB_all[b]).float().mean())
        if not torch.equal(c1[0], history[b][0]):
            frame0_ok = False  # the single-user side itself must reproduce its own run
        rows_out[b] = {"voice": voice, "frame0_identical": same, "frame0_codes_equal_fraction": frame0_codes_eq}
    # Teacher-forced steps. Single-user side FIRST, one probe row at a time to completion: Luka's
    # backbone has one cache row and one position counter, so rows cannot be interleaved.
    codes1 = {}
    for b in probe:
        single.backbone.reset()
        single.backbone.prefill_last(embeds[b])
        out = []
        for t in range(1, K):
            h1 = single.backbone.step(bref.embed_frame(single.wb, history[b][t - 1]))[0]  # [1,3072]
            out.append(single.flow(h1, x_0=noise[b][t : t + 1])[0])  # [37]
        codes1[b] = out
    # Batched side: every slot fed its history frame t-1 at its own position; compare frame t.
    pos = torch.tensor([lens[s] if s in history else lens[probe[0]] for s in range(B)], dtype=torch.int32)
    per_row = {b: {"sem_eq": 0, "ac_eq": 0, "n": 0} for b in probe}
    for t in range(1, K):
        x = torch.cat(
            [bref.embed_frame(pipe.wb, history[s if s in history else probe[0]][t - 1]) for s in range(B)], dim=1
        )
        hB = bp.step_batched(x, pos + (t - 1))[0]  # [B,3072]
        codesB = pipe.flow(hB, x_0=X0(t))  # the batched flow at B rows: [B,37]
        for b in probe:
            c1 = codes1[b][t - 1]
            per_row[b]["sem_eq"] += int(codesB[b, 0] == c1[0])
            per_row[b]["ac_eq"] += int((codesB[b, 1:] == c1[1:]).sum())
            per_row[b]["n"] += 1
    worst_sem = worst_ac = 1.0
    for b in probe:
        r = per_row[b]
        rows_out[b].update(
            sem_agree=r["sem_eq"] / r["n"], acoustic_code_agree=r["ac_eq"] / (r["n"] * 36), frames=r["n"]
        )
        worst_sem, worst_ac = min(worst_sem, rows_out[b]["sem_agree"]), min(
            worst_ac, rows_out[b]["acoustic_code_agree"]
        )
        print(
            f"[phaseA] A2 row {b:2d} {rows_out[b]['voice']:16s}: frame0 identical {rows_out[b]['frame0_identical']}, "
            f"teacher-forced over {r['n']} frames: semantic agree {rows_out[b]['sem_agree']:.2f}, "
            f"acoustic codes agree {rows_out[b]['acoustic_code_agree']:.3f}"
        )
    _record(a2_rows=rows_out, a2_frame0_identical=frame0_ok, a2_worst_sem_agree=worst_sem, a2_worst_ac_agree=worst_ac)
    assert frame0_ok, "the single-user pipeline did not reproduce its own frame 0 (noise stream mismatch)"
    # Measured 2026-10-01 on a BH Galaxy chip, B=32, four rows x 19 frames: semantic 1.00/1.00/0.95/1.00,
    # acoustic 0.958..0.968, frame 0 identical on three rows and 35/37 codes on the fourth. A ~1e-3
    # hidden-state difference (spike: PCC 0.9993 vs the fp32 reference) flips ~3.5% of the 21-level
    # acoustic codes. Whether that is audible is the WER gate's call (Phase C), not this test's.
    # Device-vs-device only. With the token-major flow (default for B > 1) acoustic agreement with
    # the single-user path reads 0.944..0.972 (2026-10-01): the two flows differ, and against the
    # fp32 reference the token-major one is the closer (tests/perf/test_flow_tm.py), so a lower
    # agreement here is not a regression. The quality call stays with the WER gate.
    assert worst_sem >= 0.90, f"semantic code agreement under teacher forcing too low: {rows_out}"
    assert worst_ac >= 0.93, f"acoustic code agreement under teacher forcing too low: {rows_out}"


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
    bb_ms = pipe.backbone_graph_ms()
    _record(
        a4_timings=t,
        a4_ms_per_frame=ms,
        a4_rtf_per_user=ms / REAL_TIME_MS,
        a4_device_only_ms_per_frame=dev_ms,
        a4_backbone_only_ms_per_frame=bb_ms,
        a4_flow_share_ms_per_frame=dev_ms - bb_ms,
        a4_flow_prg=os.environ.get("VOXTRAL_FLOW_PRG", "mcast1d"),
    )
    print(
        f"[phaseA] A4 backbone-only traced: {bb_ms:.1f} ms/frame; flow + semantic head share: {dev_ms - bb_ms:.1f} ms/frame "
        f"(flow configs: {os.environ.get('VOXTRAL_FLOW_PRG', 'mcast1d')})"
    )
    print(
        f"\n[phaseA] A4 B={B}: prefill {t['prefill_s']:.2f}s for {B} users, {t['steps']} traced steps, "
        f"{ms:.1f} ms/frame for all users ({REAL_TIME_MS / ms:.2f}x real time per user), traced={t['traced']}"
    )
    print(
        f"[phaseA] A4 device-only frame graph: {dev_ms:.1f} ms/frame; host work per frame: {ms - dev_ms:.1f} ms "
        f"(what Phase B removes)"
    )
    assert ms < REAL_TIME_MS, f"{ms:.1f} ms/frame is slower than real time ({REAL_TIME_MS} ms)"
