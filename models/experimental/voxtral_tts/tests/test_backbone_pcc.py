# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block 1 on device against the fp32 reference: wiring, prefill, decode, and the KV cache.

  * wiring    -- one layer, which is where a rotation-convention error shows.
  * prefill   -- all 15 fixture prompts, pooled and last-position, each paired with a
    worst-sample bound because a correlation alone can hide one far-off element.
  * decode    -- teacher-forced on real frames, so each step is an independent measurement.
  * KV cache  -- every cached K and V entry, all 26 layers, at four prompt lengths, plus a check
    that decode leaves the prompt's positions untouched.

PCC floors are assertable here; worst-sample aggregate LEVELS are not, since they vary more between
prompts than between builds. Those belong in the paired comparison `scripts/quality_report.py`
performs.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_backbone_pcc.py
    pytest -svv models/experimental/voxtral_tts/tests/test_backbone_pcc.py -k "case0 or case2"
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

# Every test in this file opens a device (the one-layer wiring test included -- it was the
# 20 s outlier in the supposedly host-only subset). Module-level, so the mark cannot be
# forgotten on a new test.
pytestmark = pytest.mark.slow

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (  # noqa: E402
    DIM,
    HEAD_DIM,
    N_KV_HEADS,
    HEAD_DIM,
    N_LAYERS,
    ROPE_THETA,
    causal_bias,
    pcc,
    rope_cis,
)
from models.experimental.voxtral_tts.tests.gates import compare_hidden  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import (  # noqa: E402
    as_device_k_layout,
    backbone_state,
    case_ids,
    fixture_embeds,
    real_frames,
)
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import TtVoxtralGPT  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

PCC_PREFILL = 0.999
PCC_DECODE = 0.999
# The per-position minimum is printed, not asserted: a single position's PCC is far noisier than
# the pooled or last-position figure. What is gated is the worst-sample bound on the last position.
MAX_WORST_SAMPLE_PCT = 5.0
MAX_POOLED_WORST_SAMPLE_PCT = 8.0   # largest single-element error over all positions
DECODE_STEPS = 8


@pytest.fixture(scope="module")
def dev():
    """The model's own opener, not the repo `device` fixture: this block needs the pipeline's
    l1_small and trace-region settings."""
    d = open_device()
    yield d
    ttnn.close_device(d)


@pytest.fixture(scope="module")
def w():
    return backbone_state()


@pytest.fixture(scope="module")
def gen(dev, w):
    return TtVoxtralGPT(dev, n_layers=N_LAYERS, state=w, max_seq_len=1024)


def test_one_layer_wiring_pcc(dev):
    """One layer against the reference, where a rotation-convention error shows.

    Random inputs are fine for this test alone: it checks wiring, not accuracy."""
    S = 128
    one = TtVoxtralGPT(dev, n_layers=1)
    ws = bref.load_backbone_state()
    torch.manual_seed(0)
    x = torch.randn(1, S, DIM) * 0.02
    exp = bref._layer(x, ws, "layers.0.", rope_cis(S, HEAD_DIM, ROPE_THETA), causal_bias(S, torch.float32))
    got = one.prefill(x, apply_final_norm=False)
    got_pcc = compare_hidden(got, exp)["pcc"]
    print(f"\n  [1 layer prefill] PCC {got_pcc:.8f}  maxabs {(got - exp).abs().max():.3e}")
    assert got_pcc > 0.999, f"one-layer wiring PCC {got_pcc:.6f} -- suspect the RoPE convention"


@pytest.mark.parametrize("ci", case_ids(), ids=lambda c: f"case{c}")
def test_prefill_pcc(gen, w, ci):
    """Full 26-layer prefill on a real prompt, pooled and at the last position."""
    embeds, case = fixture_embeds(ci, w)
    P = embeds.shape[1]
    exp = bref.reference_forward(embeds, w, n_layers=N_LAYERS)
    got = gen.prefill(embeds)
    m_all = compare_hidden(got, exp)
    all_pcc = m_all["pcc"]
    m_last = compare_hidden(got[:, -1:], exp[:, -1:])
    last_pcc = m_last["pcc"]
    # The pipeline calls prefill_last, a different op sequence: slice one row then norm it, versus
    # norm every row then index. Asserted equal so the gates above cover the shipped path.
    gen.reset()
    shipped = gen.prefill(embeds, last_only=True).reshape(1, -1)
    per = [pcc(got[:, i], exp[:, i]) for i in range(P)]
    wi = min(range(P), key=lambda i: per[i])
    print(
        f"\n  case {ci} ({case['voice']}, P={P}): PCC all {all_pcc:.6f}  last {last_pcc:.6f}  "
        f"worst-sample last {m_last['worst_pct']:.2f}% pooled {m_all['worst_pct']:.2f}%  "
        f"min per-pos {per[wi]:.6f} (@{wi})"
    )
    assert last_pcc > PCC_PREFILL, f"case {ci} prefill last-position PCC {last_pcc:.6f}"
    assert all_pcc > PCC_PREFILL, f"case {ci} prefill all-positions PCC {all_pcc:.6f}"
    ws = m_last["worst_pct"]
    assert ws < MAX_WORST_SAMPLE_PCT, f"case {ci} last-position worst sample {ws:.2f}% of reference scale"
    assert m_all["worst_pct"] < MAX_POOLED_WORST_SAMPLE_PCT, (
        f"case {ci} pooled worst sample {m_all['worst_pct']:.2f}% over all {P} positions -- one "
        f"element is far off even though pooled PCC is {all_pcc:.6f}")
    assert torch.equal(shipped, got[:, -1]), (
        f"case {ci}: prefill_last (the call the pipeline makes) differs from prefill(last_only="
        f"False)[:, -1] by max {(shipped - got[:, -1]).abs().max():.3e} -- the two paths have "
        f"diverged, and only the last_only=False one is covered by the gates above")


@pytest.mark.parametrize("ci", case_ids(), ids=lambda c: f"case{c}")
def test_decode_pcc_teacher_forced(gen, w, ci):
    """Device KV cache and decode steps against the reference, teacher-forced on real frames."""
    frames = real_frames()
    embeds, case = fixture_embeds(ci, w)
    P = embeds.shape[1]
    inc = bref.IncrementalBackbone(w, n_layers=N_LAYERS)
    h_ref = inc.prefill(embeds)
    gen.reset()
    h_dev = gen.prefill(embeds, last_only=True)
    assert gen.pos == inc.pos == P, f"position mismatch after prefill: {gen.pos} vs {inc.pos}"

    pcs, wss = [], []
    for t in range(min(DECODE_STEPS, frames.shape[0])):
        emb = bref.embed_frame(w, frames[t])
        h_ref = inc.step(emb)
        h_dev = gen.step(emb)
        _m = compare_hidden(h_dev, h_ref)
        pcs.append(_m["pcc"])
        wss.append(_m["worst_pct"])
    print(
        f"\n  case {ci} ({case['voice']}, P={P}), {len(pcs)} frames: min PCC {min(pcs):.6f}  "
        f"mean worst-sample {sum(wss)/len(wss):.2f}%  max {max(wss):.2f}%"
    )
    assert min(pcs) > PCC_DECODE, f"case {ci} decode min PCC {min(pcs):.6f}"


def test_decode_is_bit_deterministic(gen, w):
    """The same config re-run must reproduce bit-identically."""
    frames = real_frames()
    embeds, _ = fixture_embeds(0, w)

    def run():
        gen.reset()
        gen.prefill(embeds, last_only=True)
        out = []
        for t in range(min(4, frames.shape[0])):
            out.append(gen.step(bref.embed_frame(w, frames[t])).clone())
        return out

    a, b = run(), run()
    for t, (x, y) in enumerate(zip(a, b)):
        assert torch.equal(x, y), f"decode step {t} not reproducible: max delta {(x - y).abs().max():.3e}"
    print(f"\n  {len(a)} decode steps reproduced bit-identically across two runs")


# ── The KV cache prefill writes ──
CACHE_CASES = (0, 2, 3, 12)         # P = 100..357
CACHE_CASE = CACHE_CASES[0]
# Cache worst-sample is not gated: it is dominated by one position per prompt, so a threshold
# loose enough to pass would assert nothing. The hidden-state worst-sample is gated above.
CACHE_PCC = 0.998
CACHE_STEPS = 4

# THE K CACHE IS STORED IN A DIFFERENT HEAD-DIM ORDER ON THE TWO SIDES, and comparing raw K without
# accounting for it reads PCC ~0.02 in all 26 layers while the model is perfectly healthy. Measured
def _reference_cache(w, embeds):
    """-> {layer_index: (k, v)} after a reference prefill, each [1, N_KV_HEADS, P, HEAD_DIM]."""
    inc = bref.IncrementalBackbone(w, n_layers=N_LAYERS)
    inc.prefill(embeds)
    return {i: inc.cache[f"layers.{i}."] for i in range(N_LAYERS)}, inc


def _device_cache(gen, P):
    """-> {layer_index: (k, v)} sliced to the prompt's P positions."""
    out = {}
    for i, (kc, vc) in enumerate(gen.caches):
        out[i] = (ttnn.to_torch(kc).float()[:, :, :P, :], ttnn.to_torch(vc).float()[:, :, :P, :])
    return out


@pytest.mark.parametrize("ci", CACHE_CASES, ids=lambda c: f"case{c}")
def test_prefill_kv_cache_matches_reference(gen, w, ci):
    """Every cached K and V entry, all 26 layers, against the reference's own cache."""
    embeds, case = fixture_embeds(ci, w)
    P = embeds.shape[1]
    ref_cache, _ = _reference_cache(w, embeds)
    gen.reset()
    gen.prefill(embeds)
    dev_cache = _device_cache(gen, P)

    rows = []
    for i in range(N_LAYERS):
        for side, j in (("K", 0), ("V", 1)):
            exp, got = ref_cache[i][j].float(), dev_cache[i][j]
            if side == "K":
                exp = as_device_k_layout(exp)       # reference_helpers explains why
            assert exp.shape == got.shape, (
                f"layer {i} {side}: reference {tuple(exp.shape)} vs device {tuple(got.shape)}")
            m = compare_hidden(got, exp)
            # worst position, so a failure names one instead of a whole layer
            per_pos = (got - exp).abs().amax(dim=(0, 1, 3))
            rows.append((i, side, m["pcc"], m["worst_pct"], int(per_pos.argmax())))

    worst_pcc = min(r[2] for r in rows)
    worst_ws = max(r[3] for r in rows)
    print(f"\n  case {ci} ({case['voice']}), P={P}, {N_LAYERS} layers x (K,V), "
          f"cache [{1}, {N_KV_HEADS}, {P}, {HEAD_DIM}]")
    for i, side, pc, ws, pos in sorted(rows, key=lambda r: r[2])[:5]:
        print(f"    weakest: layer {i:>2} {side}  PCC {pc:.6f}  worst-sample {ws:.2f}%  @pos {pos}")
    print(f"  worst PCC {worst_pcc:.6f}, worst-sample {worst_ws:.2f}% across all "
          f"{len(rows)} (layer, side) pairs")
    bad = [(i, s, pc) for i, s, pc, _, _ in rows if pc <= CACHE_PCC]
    assert not bad, "cache entries below the gate: " + ", ".join(
        f"layer {i} {s} PCC {pc:.6f}" for i, s, pc in bad)


def test_decode_does_not_disturb_the_prompt_cache(gen, w):
    """Decode must leave the prompt's positions exactly as prefill wrote them.

    Device against itself, so it isolates the write index from any numerical question."""
    embeds, _ = fixture_embeds(CACHE_CASE, w)
    P = embeds.shape[1]
    frames = real_frames()
    gen.reset()
    gen.prefill(embeds, last_only=True)
    before = _device_cache(gen, P)
    for t in range(CACHE_STEPS):
        gen.step(bref.embed_frame(w, frames[t]))
    after = _device_cache(gen, P)

    moved = []
    for i in range(N_LAYERS):
        for side, j in (("K", 0), ("V", 1)):
            if not torch.equal(before[i][j], after[i][j]):
                d = (before[i][j] - after[i][j]).abs()
                moved.append((i, side, float(d.max()), int(d.amax(dim=(0, 1, 3)).argmax())))
    if moved:
        for i, side, mx, pos in moved[:5]:
            print(f"\n    layer {i} {side} changed by {mx:.3e} at prompt position {pos}")
    assert not moved, (
        f"{len(moved)} of {N_LAYERS * 2} (layer, side) cache regions changed over {CACHE_STEPS} "
        f"decode steps -- decode is writing inside the prompt's positions [0, {P})")
    print(f"\n  {N_LAYERS} layers x (K,V) unchanged across {CACHE_STEPS} decode steps "
          f"(prompt positions [0, {P}))")
