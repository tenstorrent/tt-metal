# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SPIKE: does the backbone decode step cost about the same at B rows as at 1, and does every row
stay numerically equal to a single-row run?

The batching plan rests on one claim: decode is bound by weight streaming, so B <= 32 users in one
tile of rows should cost close to one user. This test measures it and records the per-row
equivalence against the unchanged batch-1 path (`TtVoxtralGPT(max_batch=1).step`).

Gates (env-tunable):
  per-row PCC of the B-row step against the batch-1 step        >= SPIKE_PCC_MIN   (0.999)
  traced ms/step ratio  time(B) / time(1)                        <= SPIKE_RATIO_MAX (2.0; measured 1.76 at B=32 on a BH Galaxy chip)
  eager ratio is reported, not gated: eager carries per-op host dispatch on both sides.

Run (environment and checkpoint: see the README, "Checkpoint" and "Tests"):
  SPIKE_LAYERS=4 pytest -svv models/experimental/voxtral_tts/tests/perf/test_decode_batch_spike.py   # smoke
  pytest -svv models/experimental/voxtral_tts/tests/perf/test_decode_batch_spike.py                  # full
Env: SPIKE_BATCH (32), SPIKE_LAYERS (26), SPIKE_STEPS (20), SPIKE_REPLAYS (32), SPIKE_RESULTS (json path),
     VOXTRAL_DEVICE_ID (0).
"""

import json
import os
import time

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import DIM, N_LAYERS, pcc  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import (  # noqa: E402
    backbone_state,
    fixture_cases,
    fixture_embeds,
    needs_checkpoint,
    real_frames,
)
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import TILE, TtVoxtralGPT  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

pytestmark = [pytest.mark.slow, needs_checkpoint]

B = int(os.environ.get("SPIKE_BATCH", "32"))
LAYERS = int(os.environ.get("SPIKE_LAYERS", str(N_LAYERS)))
STEPS = int(os.environ.get("SPIKE_STEPS", "20"))
REPLAYS = int(os.environ.get("SPIKE_REPLAYS", "32"))
WARM = 3
MAX_SEQ = 1024  # sdpa_decode wants a multiple of its 512 k-chunk
DEVICE_ID = int(os.environ.get("VOXTRAL_DEVICE_ID", "0"))
PCC_MIN = float(os.environ.get("SPIKE_PCC_MIN", "0.999"))
# Against the fp32 reference (the arbiter): the base port's horizon gate, and the batched path may not sit
# further from the reference than the batch-1 path by more than this.
PCC_REF_MIN = float(os.environ.get("SPIKE_PCC_REF_MIN", "0.997"))
PCC_REF_SLACK = float(os.environ.get("SPIKE_PCC_REF_SLACK", "0.001"))
RATIO_MAX = float(os.environ.get("SPIKE_RATIO_MAX", "2.0"))  # measured 1.76 on BH; the real bar is absolute ms/frame
RESULTS_PATH = os.environ.get("SPIKE_RESULTS", "")

RESULTS = {"batch": B, "layers": LAYERS, "steps": STEPS, "device_id": DEVICE_ID, "max_seq": MAX_SEQ}


def _record(**kv):
    RESULTS.update(kv)
    if RESULTS_PATH:
        with open(RESULTS_PATH, "w") as fh:
            json.dump(RESULTS, fh, indent=2)


def _frames():
    """Real teacher-forced frames [T, 37] from the fixture: on-manifold decode inputs."""
    fr = real_frames()
    if isinstance(fr, dict):
        fr = next(iter(fr.values()))
    if isinstance(fr, (list, tuple)):
        fr = fr[0]
    fr = torch.as_tensor(fr)
    assert fr.ndim == 2 and fr.shape[-1] == 37, f"unexpected frames shape {tuple(fr.shape)}"
    return fr.long()


def _frame_embed(w, codes):
    """one frame's 37 codes -> [1, 1, 3072] backbone input."""
    return torch.as_tensor(bref.embed_frame(w, codes)).float().reshape(1, 1, DIM)


@pytest.fixture(scope="module")
def dev():
    d = open_device(device_id=DEVICE_ID)
    yield d
    ttnn.close_device(d)


@pytest.fixture(scope="module")
def w():
    return backbone_state()


@pytest.fixture(scope="module")
def g1(dev, w):
    return TtVoxtralGPT(dev, n_layers=LAYERS, state=w, max_seq_len=MAX_SEQ, max_batch=1)


@pytest.fixture(scope="module")
def gB(dev, w):
    return TtVoxtralGPT(dev, n_layers=LAYERS, state=w, max_seq_len=MAX_SEQ, max_batch=B)


def _sync(dev):
    ttnn.synchronize_device(dev)


def test_shapes_and_slots(gB, g1):
    """Caches carry one row per user; the batch-1 object is untouched by the batch code."""
    k, v = gB.caches[0]
    assert tuple(k.shape) == (B, 8, MAX_SEQ, 128) and tuple(v.shape) == (B, 8, MAX_SEQ, 128)
    assert tuple(g1.caches[0][0].shape) == (1, 8, MAX_SEQ, 128)
    assert gB.max_batch == B and g1.max_batch == 1


def test_same_prompt_every_row_matches_batch1(dev, w, g1, gB):
    """Every one of the B rows, fed the same prompt and frames as the batch-1 object, must agree
    with it. Also the eager timing of both loops."""
    embeds, _ = fixture_embeds(0, w)
    P = embeds.shape[1]
    frames = _frames()
    n = WARM + STEPS
    assert P + n < MAX_SEQ and frames.shape[0] >= n, "fixture too short for this many steps"

    g1.reset()
    g1.prefill_last(embeds)
    for u in range(B):
        gB.prefill(embeds, last_only=True, user=u)

    worst = 1.0
    worst_at = None
    t1 = tB = 0.0
    for t in range(n):
        e = _frame_embed(w, frames[t])
        pos = P + t
        _sync(dev)
        s = time.perf_counter()
        h1 = g1.step(e)
        _sync(dev)
        d1 = time.perf_counter() - s
        s = time.perf_counter()
        hB = gB.step_batched(e.expand(1, B, DIM).contiguous(), torch.full((B,), pos, dtype=torch.int32))
        _sync(dev)
        dB = time.perf_counter() - s
        if t >= WARM:
            t1 += d1
            tB += dB
        for b in range(B):
            p = pcc(hB[0, b], h1[0, 0])
            if p < worst:
                worst, worst_at = p, (t, b)
    ms1, msB = t1 / STEPS * 1e3, tB / STEPS * 1e3
    _record(
        eager_ms_b1=ms1,
        eager_ms_bB=msB,
        eager_ratio=msB / ms1,
        same_prompt_worst_pcc=worst,
        same_prompt_worst_at=worst_at,
    )
    print(f"\n[spike] eager  B=1 {ms1:7.2f} ms/step   B={B} {msB:7.2f} ms/step   ratio {msB / ms1:.2f}")
    print(f"[spike] same-prompt worst per-row PCC vs batch-1: {worst:.6f} at (step, row)={worst_at}")
    assert worst >= PCC_MIN, f"row diverged from the batch-1 path: PCC {worst:.5f} at {worst_at}"


def test_distinct_prompts_and_positions_per_row(dev, w, g1, gB):
    """Each row gets its own prompt and its own length, so positions differ per user; each row must
    match its own batch-1 run. This is the shape a server batch actually has."""
    cases = fixture_cases()
    n_cases = len(cases)
    K = min(5, STEPS)
    frames = _frames()
    prompts = []
    for b in range(B):
        e, _ = fixture_embeds(b % n_cases, w)
        cut = max(16, e.shape[1] - (b % 7) * 3)  # vary the length a little per row
        prompts.append(e[:, :cut].contiguous())
    lens = torch.tensor([p.shape[1] for p in prompts], dtype=torch.int32)
    assert int(lens.max()) + K < MAX_SEQ

    # Reference per row: the batch-1 object, one row at a time.
    ref = torch.zeros(K, B, DIM)
    for b in range(B):
        g1.reset()
        g1.prefill_last(prompts[b])
        for t in range(K):
            ref[t, b] = g1.step(_frame_embed(w, frames[t]))[0, 0]

    for b in range(B):
        gB.prefill(prompts[b], last_only=True, user=b)
    worst, worst_at = 1.0, None
    outB = torch.zeros(K, B, DIM)
    for t in range(K):
        e = _frame_embed(w, frames[t]).expand(1, B, DIM).contiguous()
        hB = gB.step_batched(e, lens + t)
        outB[t] = hB[0]
        for b in range(B):
            p = pcc(hB[0, b], ref[t, b])
            if p < worst:
                worst, worst_at = p, (t, b)
    print(f"\n[spike] distinct-prompt worst per-row PCC vs own batch-1 run: {worst:.6f} at (step, row)={worst_at}")

    # The arbiter is the fp32 reference (the base port's decode gate: 0.999, 0.997 over a horizon), not the
    # batch-1 device path: sdpa_decode reduces in a different order per batch size and grid, so the
    # two device paths may legitimately differ a little. Check the worst row, row 0, and the
    # shortest and longest prompts against the reference, for BOTH device paths.
    rows = sorted({worst_at[1], 0, int(lens.argmin()), int(lens.argmax())})
    ref_pcc = {}
    worst_ref_B, worst_ref_1 = 1.0, 1.0
    for b in rows:
        inc = bref.IncrementalBackbone(w, n_layers=LAYERS)
        inc.prefill(prompts[b])
        pB, p1 = [], []
        for t in range(K):
            h_ref = inc.step(_frame_embed(w, frames[t]))[0, 0]
            pB.append(pcc(outB[t, b], h_ref))
            p1.append(pcc(ref[t, b], h_ref))
        ref_pcc[b] = {"len": int(lens[b]), "b32_vs_ref": pB, "b1_vs_ref": p1}
        worst_ref_B, worst_ref_1 = min(worst_ref_B, min(pB)), min(worst_ref_1, min(p1))
        print(
            f"[spike] row {b:2d} len {int(lens[b]):3d}: B={B} vs fp32 ref min {min(pB):.6f}   "
            f"B=1 vs fp32 ref min {min(p1):.6f}"
        )
    _record(
        distinct_worst_pcc=worst,
        distinct_worst_at=worst_at,
        distinct_lens=lens.tolist(),
        distinct_ref_rows=ref_pcc,
        distinct_worst_b32_vs_ref=worst_ref_B,
        distinct_worst_b1_vs_ref=worst_ref_1,
    )
    print(f"[spike] distinct-prompt vs fp32 reference: B={B} worst {worst_ref_B:.6f}, B=1 worst {worst_ref_1:.6f}")
    assert worst_ref_B >= PCC_REF_MIN, f"batched row diverged from the fp32 reference: {worst_ref_B:.5f}"
    assert worst_ref_B >= worst_ref_1 - PCC_REF_SLACK, (
        f"batched path is further from the reference than batch-1 by more than {PCC_REF_SLACK}: "
        f"{worst_ref_B:.5f} vs {worst_ref_1:.5f}"
    )


def _traced_ms(dev, g, B_rows, pos):
    """Capture one all-device step as a trace and replay it; -> average ms per step."""
    x_host = torch.randn(1, B_rows, DIM) * 0.02
    xin = ttnn.from_torch(x_host, dtype=g.dtype, layout=ttnn.TILE_LAYOUT, device=dev)
    pos_u32, pos_i32 = g.pos_tensors(torch.full((B_rows,), pos, dtype=torch.int32), dev)
    for _ in range(WARM):
        g.step_device(ttnn.clone(xin), pos_u32, pos_i32)
    _sync(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    try:
        out = g.step_device(ttnn.clone(xin), pos_u32, pos_i32)
    finally:
        ttnn.end_trace_capture(dev, tid, cq_id=0)
    _sync(dev)
    try:
        s = time.perf_counter()
        for _ in range(REPLAYS):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        _sync(dev)
        ms = (time.perf_counter() - s) / REPLAYS * 1e3
    finally:
        ttnn.release_trace(dev, tid)
    del out
    return ms


def test_traced_step_cost_ratio(dev, w, g1, gB):
    """The number the plan depends on: traced ms per decode step at B rows versus 1 row."""
    embeds, _ = fixture_embeds(0, w)
    P = embeds.shape[1]
    g1.reset()
    g1.prefill_last(embeds)
    for u in range(B):
        gB.prefill(embeds, last_only=True, user=u)
    ms1 = _traced_ms(dev, g1, 1, P)
    msB = _traced_ms(dev, gB, B, P)
    ratio = msB / ms1
    _record(traced_ms_b1=ms1, traced_ms_bB=msB, traced_ratio=ratio)
    print(f"\n[spike] traced B=1 {ms1:7.2f} ms/step   B={B} {msB:7.2f} ms/step   ratio {ratio:.2f}")
    print(f"[spike] per user-frame: B=1 {ms1:.2f} ms   B={B} {msB / B:.2f} ms   ({LAYERS} of {N_LAYERS} layers)")
    assert ratio <= RATIO_MAX, f"batched step costs {ratio:.2f}x the single-row step; plan assumed <= {RATIO_MAX}"
