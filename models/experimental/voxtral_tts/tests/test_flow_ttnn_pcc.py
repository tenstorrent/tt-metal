# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block 2 on device vs the fp32 reference. Replaces `tt_gates.py --gate flow`.

Block 2's output is INTEGER codes, so the frame comparison is exact-or-not rather than a PCC, and
the velocity field is the only continuous quantity worth a PCC. Synthetic inputs are acceptable
here (unlike Block 1, see BUG-9): the criterion is exact code agreement given identical input, not
a weight-precision level read off off-manifold activations.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_flow_ttnn_pcc.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.reference import voxtral_flow_ref as fref  # noqa: E402
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


# Measured on the shipped build: 2 of 74 codes differ, each by one FSQ level of 21. That is the
# recorded `flow_codes_74` metric, not a defect -- so the gate is "no worse than shipped, and every
# difference still only one level", not "identical".
MAX_FRAME_CODES_DIFF = 4


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
