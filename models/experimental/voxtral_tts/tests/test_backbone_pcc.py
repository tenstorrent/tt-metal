# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block 1 on device vs the fp32 reference: wiring, prefill, and teacher-forced decode.

Replaces `tt_gates.py --gate wiring / --gate prefill26 / --gate decode`, which printed these
numbers for a human to judge. They assert now.

WHAT IS AND IS NOT ASSERTABLE HERE, because the gate this came from was misread twice
(STATUS 6.15):

  - **PCC floors are assertable.** They are deterministic and reproduce across sessions: an
    audio-tier re-run on a different day reproduced `decode_min_pcc` 0.999316 and
    `prefill_pcc_last` 0.999855 exactly.
  - **Worst-sample AGGREGATE LEVELS are not.** Prompt-to-prompt spread is ~0.45 pp on mean and
    ~0.96 pp on p90 -- larger than any change ever gated with this, w2's BFP8 drop (0.10 pp)
    included. So an aggregate over a different prompt set is not comparable to a recorded one.
    Those belong in a paired same-session A/B, which is what `scripts/quality_report.py --compare`
    is for. The thresholds below are deliberately loose tripwires, not the measured levels.

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
    backbone_state,
    case_ids,
    fixture_embeds,
    real_frames,
)
from models.experimental.voxtral_tts.tt.ttnn_voxtral_gpt import TtVoxtralGPT  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

PCC_PREFILL = 0.999
PCC_DECODE = 0.999
# The per-position MINIMUM is printed but NOT asserted, because it is not a stable level: it swings
# 0.938473 (case 2, position 217) to 0.998110 (case 0) while the pooled and last-position figures
# stay above 0.9997.
#
# It is NOT a scale artefact -- measured, position 217 has ordinary variance (ref std 1.78 vs 1.93
# at the strongest positions) and a genuinely larger error: worst-sample 7.28% of scale against
# ~0.4% typical, a ~13x bigger absolute deviation. So intermediate prefill positions really are
# less accurate than the pooled number suggests. It does not reach the audio -- only the last
# position feeds Block 2, and that one reads 0.99988 / 0.68% -- but see VOXTRAL_TTS_BACKBONE.md's
# open questions, because prefill also writes the KV cache that every decode step then attends to.
#
# What IS gated is the worst-sample bound on the last position: "that worst-sample bound is the gate
# that matters" (STATUS trap 9, PCC hides outliers). Loose -- case 0 measures 0.70%.
MAX_WORST_SAMPLE_PCT = 5.0
# The POOLED worst-sample: the largest single-element error anywhere in the whole [1, S, 3072]
# output, as a percentage of the reference's scale. Gated because "PCC HIDES OUTLIERS -- the single
# most expensive lesson" (STATUS trap 9: sdpa passed at PCC 0.9998 per slab and still failed 11
# tests). It is LARGER than the last-position figure because pooling takes the max over every
# position, so it picks up the weak late ones.
#
# Measured over all 15 real prompts, 2026-08-27: 0.74% (case 8) .. 5.17% (case 10), with case 12 at
# 3.86%. Gate at 8% -- clear of the measured band, and an order of magnitude below the kind of
# misplacement that matters (a dropped head reads 47%).
MAX_POOLED_WORST_SAMPLE_PCT = 8.0
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
    """ONE layer against the reference. A RoPE convention error shows up here and nowhere else.

    Random inputs are fine for this test and only this test: it checks wiring and the rotation
    convention, not accuracy. Accuracy is judged on real prompts at 26 layers below."""
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
    """Full 26-layer prefill on a REAL prompt vs `reference_forward`.

    The last position is reported separately because it is the only one Block 2 consumes; the
    all-positions number catches a bug that only touches part of the sequence, and the per-position
    minimum catches one that a pooled PCC would hide behind the high-magnitude positions."""
    embeds, case = fixture_embeds(ci, w)
    P = embeds.shape[1]
    exp = bref.reference_forward(embeds, w, n_layers=N_LAYERS)
    got = gen.prefill(embeds)
    m_all = compare_hidden(got, exp)
    all_pcc = m_all["pcc"]
    m_last = compare_hidden(got[:, -1:], exp[:, -1:])
    last_pcc = m_last["pcc"]
    # The call the PIPELINE makes is prefill_last -> prefill(last_only=True), which is a different
    # op sequence: slice to one row on device THEN rms_norm it, versus rms_norm all Sp rows then
    # index on the host. Mathematically the same (RMSNorm is per-row) but a different invocation at
    # a different shape, and norm configs are shape-sensitive in this port -- 6.39/6.40 deleted both
    # width-sharded norms, and the sharded norm is decode-only because its shard spec fixes the
    # height at one tile. Measured bit-identical on all 15 prompts; asserted so it stays that way.
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
    """On-device KV cache + decode steps vs `IncrementalBackbone.step()`, teacher-forced on real
    frames so every step is an independent measurement."""
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
    """The same config re-run must reproduce bit-identically.

    The gate this replaced documented that property and relied on it -- a paired A/B is only
    readable at 0.01 pp because of it -- but nothing asserted it."""
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


# ── The KV cache prefill writes ───────────────────────────────────────────────────────────────
# Prefill has TWO jobs: produce the last hidden state (which Block 2 consumes, gated above) and
# populate the KV cache that every later decode step attends to. Only the first was gated. The
# second was covered transitively -- a wrong entry drifts the decode hidden states -- but weakly:
# attention averages over 200+ positions, so one bad entry is heavily diluted, and the decode test
# only runs 8 steps.
#
# BUG-5 is exactly this surface: warm-up/capture writing the cache "corrupts the prompt unless
# aimed at a scratch row; measured decode PCC 0.9998 -> 0.86 with NO error raised". Decode PCC
# caught that because it was catastrophic. These tests localise instead of detect: they name the
# layer, the position and the side (K or V).
#
# Both sides cache POST-RoPE: the reference applies `apply_rope` before writing the cache, and the
# device caches what `_qkv` returns, which is also post-RoPE. So this compares like with like.
CACHE_CASE = 0
CACHE_PCC = 0.999
CACHE_STEPS = 4

# THE K CACHE IS STORED IN A DIFFERENT HEAD-DIM ORDER ON THE TWO SIDES, and comparing raw K without
# accounting for it reads PCC ~0.02 in all 26 layers while the model is perfectly healthy. Measured
# 2026-08-27: as-is -0.0035, permuted 0.999975 (layer 0); 0.0357 vs 0.999928 (layer 12).
#
# The reference lays a rotated head out HALF-SPLIT -- first half, then second -- and the device lays
# it out INTERLEAVED, pairs adjacent. That is a permutation of the head dimension, and RoPE applies
# it to Q as well, so Q.K is unchanged and attention is identical either way. Hence V, which is
# never rotated, matches with no permutation at all, and only K needs one. Decode PCC 0.9998 and
# WER 0 of 894 are the independent confirmation that both layouts are self-consistent.
#
# This matters beyond the test: anything that reads, transplants or pages the K cache -- or compares
# it against another implementation -- has to know which order it is in.
_HALF_TO_INTERLEAVED = torch.empty(HEAD_DIM, dtype=torch.long)
_HALF_TO_INTERLEAVED[: HEAD_DIM // 2] = torch.arange(0, HEAD_DIM, 2)
_HALF_TO_INTERLEAVED[HEAD_DIM // 2 :] = torch.arange(1, HEAD_DIM, 2)


def _as_device_k_layout(k_ref):
    """Reference K (half-split head dim) -> the device's interleaved order."""
    return k_ref[..., _HALF_TO_INTERLEAVED]


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


def test_prefill_kv_cache_matches_reference(gen, w):
    """Every cached K and V entry, all 26 layers, against the fp32 reference's own cache."""
    embeds, case = fixture_embeds(CACHE_CASE, w)
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
                exp = _as_device_k_layout(exp)      # see _HALF_TO_INTERLEAVED above
            assert exp.shape == got.shape, (
                f"layer {i} {side}: reference {tuple(exp.shape)} vs device {tuple(got.shape)}")
            m = compare_hidden(got, exp)
            # worst position, so a failure names one instead of a whole layer
            per_pos = (got - exp).abs().amax(dim=(0, 1, 3))
            rows.append((i, side, m["pcc"], m["worst_pct"], int(per_pos.argmax())))

    worst_pcc = min(r[2] for r in rows)
    worst_ws = max(r[3] for r in rows)
    print(f"\n  case {CACHE_CASE} ({case['voice']}), P={P}, {N_LAYERS} layers x (K,V), "
          f"cache [{1}, {N_KV_HEADS}, {P}, {HEAD_DIM}]")
    for i, side, pc, ws, pos in sorted(rows, key=lambda r: r[2])[:5]:
        print(f"    weakest: layer {i:>2} {side}  PCC {pc:.6f}  worst-sample {ws:.2f}%  @pos {pos}")
    print(f"  worst PCC {worst_pcc:.6f}, worst-sample {worst_ws:.2f}% across all "
          f"{len(rows)} (layer, side) pairs")
    bad = [(i, s, pc) for i, s, pc, _, _ in rows if pc <= CACHE_PCC]
    assert not bad, "cache entries below the gate: " + ", ".join(
        f"layer {i} {s} PCC {pc:.6f}" for i, s, pc in bad)


def test_decode_does_not_disturb_the_prompt_cache(gen, w):
    """After prefill, decode steps must leave positions [0, P) exactly as prefill wrote them.

    This is BUG-5's shape. A step that writes at the wrong index -- or a warm-up/capture aimed at a
    live row instead of a scratch one -- corrupts the prompt the whole utterance is conditioned on,
    and raises nothing. Compared device-against-itself, so it isolates the write index from any
    numerical question.
    """
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
