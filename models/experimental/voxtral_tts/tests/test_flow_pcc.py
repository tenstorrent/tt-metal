# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block 2 on device against the fp32 reference.

Block 2 emits integer codes, so the frame comparison is exact-or-not and the velocity field is the
only continuous quantity worth a PCC. Synthetic inputs are acceptable for the exactness checks; the
real-hidden-state tests below drive it from the reference's own last-position hidden state, which is
what Block 1 hands it at inference.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_flow_pcc.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

# Every test in this file opens a device. Module-level so `-m "not slow"` is genuinely
# the host-only subset -- it used to still run 22 device tests from the two _ttnn_ files.
pytestmark = pytest.mark.slow

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref  # noqa: E402
from models.experimental.voxtral_tts.reference import voxtral_flow_ref as fref  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import N_LAYERS  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import (  # noqa: E402
    backbone_state,
    fixture_embeds,
)
from models.experimental.voxtral_tts.tests.gates import compare_hidden  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_flow import TtVoxtralFlow  # noqa: E402
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device  # noqa: E402

PCC_VELOCITY = 0.999


@pytest.fixture(scope="module")
def dev():
    d = open_device()
    yield d
    ttnn.close_device(d)


@pytest.fixture(scope="module")
def rig(dev):
    gen = TtVoxtralFlow(dev)
    w = fref.load_flow_state()
    h, x_0 = fref.make_synthetic_inputs(batch=2, seed=0)
    return gen, w, h, x_0


def test_velocity_pcc(rig):
    """One velocity evaluation -- the unit a trace captures."""
    gen, w, h, x_0 = rig
    t_emb = fref.time_embedding(torch.tensor(0.375).view(1, 1).repeat(2, 1), w["time_embedding.inv_freq"])
    exp = fref.predict_velocity(x_0, h, t_emb, w)
    got = gen._predict_velocity(x_0, h, t_emb)
    got_pcc = compare_hidden(got, exp)["pcc"]
    print(f"\n  [velocity] PCC {got_pcc:.8f}  maxabs {(got - exp).abs().max():.3e}")
    assert got_pcc > PCC_VELOCITY, f"velocity PCC {got_pcc:.8f}"


def test_semantic_code_is_exact(rig):
    """The semantic code is an INDEX -- a wrong one changes the audio outright, so it must match
    exactly, not closely."""
    gen, w, h, _ = rig
    exp, got = fref.semantic_code(h, w), gen.semantic_code(h)
    print(f"\n  [semantic] ref {exp.flatten().tolist()}  dev {got.flatten().tolist()}")
    assert bool((exp == got).all()), f"semantic code mismatch: ref {exp.flatten().tolist()} dev {got.flatten().tolist()}"


MAX_FRAME_CODES_DIFF = 4    # a small number of codes may differ by one FSQ level


def test_full_frame_codes_close_to_reference(rig):
    """37 integer codes x2 with a deterministic x_0, against the recorded shipped level."""
    gen, w, h, x_0 = rig
    exp = fref.reference_frame(h, w, x_0=x_0)
    got = gen(h, x_0=x_0)
    d = (exp.long() - got.long()).abs()
    n_diff = int((d != 0).sum())
    worst = int(d.max())
    print(f"\n  [full frame] {n_diff} of {exp.numel()} codes differ, max |delta| {worst} FSQ level(s)")
    if n_diff:
        print(f"      ref  {exp[0, :10].tolist()}")
        print(f"      got  {got[0, :10].tolist()}")
    assert n_diff <= MAX_FRAME_CODES_DIFF, (
        f"{n_diff} of {exp.numel()} codes differ (shipped level is 2); regression in Block 2")
    assert worst <= 1, f"a code is off by {worst} FSQ levels, not one -- that is not rounding"


# ── Real hidden states ──
# Driven from the reference's own last-position hidden state, which isolates Block 2: Block 1's
# device accuracy is test_backbone_pcc.py's job.

REAL_CASES = (0, 2, 3)
X0_SEEDS = (0, 7)


@pytest.fixture(scope="module")
def wb():
    return backbone_state()


def _real_hidden(wb, ci):
    embeds, case = fixture_embeds(ci, wb)
    h = bref.reference_forward(embeds, wb, n_layers=N_LAYERS)[:, -1]   # [1, 3072]
    return h, case


@pytest.mark.slow
@pytest.mark.parametrize("ci", REAL_CASES, ids=lambda c: f"case{c}")
def test_velocity_pcc_on_real_hidden_states(rig, wb, ci):
    gen, w, _, _ = rig
    h, case = _real_hidden(wb, ci)
    t_emb = fref.time_embedding(torch.tensor(0.375).view(1, 1), w["time_embedding.inv_freq"])
    worst = 1.0
    for seed in X0_SEEDS:
        x_0 = torch.randn(1, fref.N_ACOUSTIC_CODEBOOK,
                          generator=torch.Generator().manual_seed(seed))
        exp = fref.predict_velocity(x_0, h, t_emb, w)
        got = gen._predict_velocity(x_0, h, t_emb)
        m = compare_hidden(got, exp)
        worst = min(worst, m["pcc"])
        print(f"\n  case {ci} ({case['voice']}) x0 seed {seed}: velocity PCC {m['pcc']:.8f}  "
              f"worst-sample {m['worst_pct']:.2f}%")
    assert worst > PCC_VELOCITY, f"case {ci} velocity PCC {worst:.8f} on a real hidden state"


@pytest.mark.slow
@pytest.mark.parametrize("ci", REAL_CASES, ids=lambda c: f"case{c}")
def test_frame_codes_on_real_hidden_states(rig, wb, ci):
    """Integer codes on a real hidden state, per x_0 draw."""
    gen, w, _, _ = rig
    h, case = _real_hidden(wb, ci)
    for seed in X0_SEEDS:
        x_0 = torch.randn(1, fref.N_ACOUSTIC_CODEBOOK,
                          generator=torch.Generator().manual_seed(seed))
        exp = fref.reference_frame(h, w, x_0=x_0)
        got = gen(h, x_0=x_0)
        d = (exp.long() - got.long()).abs()
        n_diff, mx = int((d != 0).sum()), int(d.max()) if d.numel() else 0
        print(f"\n  case {ci} ({case['voice']}) x0 seed {seed}: {n_diff} of {exp.numel()} codes "
              f"differ, max |delta| {mx}")
        assert int(exp[0, 0]) == int(got[0, 0]), (
            f"case {ci} seed {seed}: SEMANTIC code differs ({int(exp[0,0])} vs {int(got[0,0])}) -- "
            f"a wrong semantic code changes the audio outright")
        assert n_diff <= MAX_FRAME_CODES_DIFF, f"case {ci} seed {seed}: {n_diff} codes differ"
        assert mx <= 1, f"case {ci} seed {seed}: a code is off by {mx} FSQ levels, not one"
