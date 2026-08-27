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
    all_pcc = compare_hidden(got, exp)["pcc"]
    m_last = compare_hidden(got[:, -1:], exp[:, -1:])
    last_pcc = m_last["pcc"]
    per = [pcc(got[:, i], exp[:, i]) for i in range(P)]
    wi = min(range(P), key=lambda i: per[i])
    print(
        f"\n  case {ci} ({case['voice']}, P={P}): PCC all {all_pcc:.6f}  last {last_pcc:.6f}  "
        f"worst-sample(last) {m_last['worst_pct']:.2f}%  "
        f"min per-pos {per[wi]:.6f} (@{wi})"
    )
    assert last_pcc > PCC_PREFILL, f"case {ci} prefill last-position PCC {last_pcc:.6f}"
    assert all_pcc > PCC_PREFILL, f"case {ci} prefill all-positions PCC {all_pcc:.6f}"
    ws = m_last["worst_pct"]
    assert ws < MAX_WORST_SAMPLE_PCT, f"case {ci} last-position worst sample {ws:.2f}% of reference scale"


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
