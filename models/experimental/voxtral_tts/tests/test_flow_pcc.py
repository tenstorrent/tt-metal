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
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (  # noqa: E402
    EMPTY_AUDIO_ID,
    END_AUDIO_ID,
    FM_INPUT_DIM,
    N_AUDIO_SPECIAL,
    N_ACOUSTIC_CODEBOOK,
    N_DECODING_STEPS,
    N_LAYERS,
)
from models.experimental.voxtral_tts.tt.ttnn_voxtral_flow import CFG_ALPHA  # noqa: E402
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


# ── Internals ─────────────────────────────────────────────────────────────────────────────────
# Block 2 is a 3-layer trunk evaluated once per Euler step, so every piece is small enough to
# compare individually: the trunk's blocks, the velocity at each scheduled timestep, the integrated
# result, the CFG input, and the schedule itself.

FM_N_LAYERS = 3
BATCH = 1


def _three_token_sequence(gen, w, x_t, t_emb, llm_h):
    """-> (device sequence [1,B*3,3072], reference sequence [B,3,3072]).

    Both sides assemble the same three projections; the device folds the batch into rows.
    """
    B = x_t.shape[0]
    ps = [ttnn.linear(gen._up(t), gen.proj[name])
          for t, name in ((x_t, "input_projection"), (t_emb, "time_projection"),
                          (llm_h, "llm_projection"))]
    dev = ttnn.reshape(ttnn.concat([ttnn.reshape(p, [B, 1, FM_INPUT_DIM]) for p in ps], dim=1),
                       [1, B * 3, FM_INPUT_DIM])
    ref = torch.cat([
        torch.nn.functional.linear(x_t, w["input_projection.weight"]).unsqueeze(1),
        torch.nn.functional.linear(t_emb, w["time_projection.weight"]).unsqueeze(1),
        torch.nn.functional.linear(llm_h, w["llm_projection.weight"]).unsqueeze(1),
    ], dim=1)
    return dev, ref


@pytest.mark.slow
def test_every_block_matches_reference(rig, wb):
    """Each of the 3 trunk blocks against the reference, cumulatively, so a failure names one."""
    gen, w, _, _ = rig
    h, _ = _real_hidden(wb, REAL_CASES[0])
    x_t = torch.randn(BATCH, N_ACOUSTIC_CODEBOOK, generator=torch.Generator().manual_seed(0))
    t_emb = fref.time_embedding(torch.tensor(0.375).view(1, 1), w["time_embedding.inv_freq"])
    dev, ref = _three_token_sequence(gen, w, x_t, t_emb, h)

    worst = 1.0
    for i in range(FM_N_LAYERS):
        dev = gen._block(dev, gen.layers[i], BATCH)
        ref = fref._block(ref, w, f"layers.{i}.")
        got = ttnn.to_torch(dev).float().reshape(BATCH, 3, FM_INPUT_DIM)
        m = compare_hidden(got, ref)
        worst = min(worst, m["pcc"])
        print(f"\n  after block {i}: PCC {m['pcc']:.8f}  worst-sample {m['worst_pct']:.2f}%")
    assert worst > PCC_VELOCITY, f"a trunk block diverged: worst PCC {worst:.8f}"


@pytest.mark.slow
def test_velocity_matches_along_the_real_trajectory(rig, wb):
    """The velocity field at each solver step, evaluated at the state the solver is actually in.

    The state has to come from the trajectory, not be held fixed while t sweeps: at a late t the
    true state is nearly converged, and pairing it with fresh noise asks for a combination the
    model never sees.
    """
    gen, w, _, _ = rig
    h, _ = _real_hidden(wb, REAL_CASES[0])
    x_0 = torch.randn(BATCH, N_ACOUSTIC_CODEBOOK, generator=torch.Generator().manual_seed(0))
    sem = fref.semantic_code(h, w)
    _, trace = fref.decode_frame(sem, h, w, x_0=x_0, return_trace=True)

    ts = torch.linspace(0, 1, N_DECODING_STEPS + 1)[:N_DECODING_STEPS]
    states = [x_0] + [trace[i] for i in range(N_DECODING_STEPS - 1)]
    worst = 1.0
    for i, t in enumerate(ts):
        t_emb = fref.time_embedding(t.view(1, 1).repeat(BATCH, 1), w["time_embedding.inv_freq"])
        m = compare_hidden(gen._predict_velocity(states[i], h, t_emb),
                           fref.predict_velocity(states[i], h, t_emb, w))
        worst = min(worst, m["pcc"])
        print(f"\n  step {i} t={float(t):.4f}: velocity PCC {m['pcc']:.8f}  "
              f"worst {m['worst_pct']:.2f}%")
    assert worst > PCC_VELOCITY, f"velocity diverged at some step: worst PCC {worst:.8f}"


@pytest.mark.slow
def test_the_solve_accumulates_correctly(rig, wb):
    """The integrated result after all 7 steps, compared before quantisation.

    Per-step device states are not exposed -- the solve is one device graph, and asking for fewer
    steps changes the schedule rather than truncating it -- so accumulation is checked at the end
    while each step's evaluation is checked by the timestep test above.
    """
    gen, w, _, _ = rig
    h, _ = _real_hidden(wb, REAL_CASES[0])
    x_0 = torch.randn(BATCH, N_ACOUSTIC_CODEBOOK, generator=torch.Generator().manual_seed(0))
    sem = fref.semantic_code(h, w)

    _, trace = fref.decode_frame(sem, h, w, x_0=x_0, return_trace=True)
    exp = trace[-1]
    h_host = gen._cfg_input(BATCH, h)
    got = ttnn.to_torch(gen._solve(
        gen._up(x_0.reshape(BATCH, 1, N_ACOUSTIC_CODEBOOK), ttnn.float32),
        gen._up(h_host), BATCH, N_DECODING_STEPS, CFG_ALPHA)).float().reshape(
            BATCH, N_ACOUSTIC_CODEBOOK)
    m = compare_hidden(got, exp)
    print(f"\n  after {N_DECODING_STEPS} steps: PCC {m['pcc']:.8f}  worst {m['worst_pct']:.2f}%")
    assert m["pcc"] > PCC_VELOCITY, f"the integrated result diverged: PCC {m['pcc']:.8f}"


def test_cfg_input_is_conditional_over_zeros(rig, wb):
    """The CFG buffer must be the conditioning on top of zeros, and refresh when reused."""
    gen, _, _, _ = rig
    h1, _ = _real_hidden(wb, REAL_CASES[0])
    h2, _ = _real_hidden(wb, REAL_CASES[1])
    buf = gen._cfg_input(BATCH, h1)
    assert buf.shape == (2 * BATCH, FM_INPUT_DIM), f"CFG buffer shaped {tuple(buf.shape)}"
    assert torch.equal(buf[:BATCH], h1), "the conditional half is not the conditioning"
    assert not buf[BATCH:].any(), "the unconditional half is not zero"
    buf2 = gen._cfg_input(BATCH, h2)
    assert torch.equal(buf2[:BATCH], h2), "the reused buffer kept the previous conditioning"
    assert not buf2[BATCH:].any(), "the unconditional half stopped being zero on reuse"


def test_schedule_matches_reference(rig):
    """Uniform timesteps summing to one, and time tokens equal to the reference's embedding."""
    gen, w, _, _ = rig
    tokens, dts = gen._schedule(BATCH, N_DECODING_STEPS)
    assert len(tokens) == len(dts) == N_DECODING_STEPS
    assert abs(sum(dts) - 1.0) < 1e-6, f"step widths sum to {sum(dts)}, not 1"
    for dt in dts:
        assert abs(dt - 1.0 / N_DECODING_STEPS) < 1e-6, f"non-uniform step width {dt}"
    ts = torch.linspace(0, 1, N_DECODING_STEPS + 1)
    worst = 1.0
    for i, tok in enumerate(tokens):
        exp = torch.nn.functional.linear(
            fref.time_embedding(ts[i].view(1, 1).repeat(BATCH, 1), w["time_embedding.inv_freq"]),
            w["time_projection.weight"]).reshape(BATCH, 1, FM_INPUT_DIM)
        worst = min(worst, compare_hidden(ttnn.to_torch(tok).float(), exp)["pcc"])
    print(f"\n  {N_DECODING_STEPS} time tokens: worst PCC {worst:.8f}")
    assert worst > PCC_VELOCITY, f"a time-conditioning token diverged: worst PCC {worst:.8f}"


@pytest.mark.slow
def test_memoised_schedules_and_buffers_do_not_go_stale(rig, wb):
    """The schedule and CFG buffers are cached per (batch, steps); a stale entry must not be served.

    Exercised by interleaving step counts and re-running the first one, whose codes must not move.
    """
    gen, w, _, _ = rig
    h, _ = _real_hidden(wb, REAL_CASES[0])
    x_0 = torch.randn(BATCH, N_ACOUSTIC_CODEBOOK, generator=torch.Generator().manual_seed(0))
    first = gen(h, x_0=x_0)
    for n in (3, 5, N_DECODING_STEPS, 2):
        gen(h, x_0=x_0, n_steps=n)
    again = gen(h, x_0=x_0)
    assert torch.equal(first, again), (
        "codes changed after other step counts ran, so a cached schedule or buffer went stale")
    print(f"\n  {tuple(sorted(gen._sched))} schedules cached; first result reproduced exactly")


def _same_codes(got, exp, label):
    """Semantic exact, acoustic within one FSQ level and no more than the shipped count."""
    d = (exp.long() - got.long()).abs()
    n_diff, mx = int((d != 0).sum()), int(d.max()) if d.numel() else 0
    print(f"\n  {label}: {n_diff} of {exp.numel()} codes differ, max |delta| {mx}")
    assert n_diff <= MAX_FRAME_CODES_DIFF, f"{label}: {n_diff} codes differ"
    assert mx <= 1, f"{label}: a code is off by {mx} FSQ levels"


@pytest.mark.slow
def test_an_end_audio_frame_is_not_decoded(rig, wb):
    """A frame whose semantic code is [END_AUDIO] must come back as empty acoustic slots.

    Otherwise the codec renders whatever the solver produced at the end of an utterance.
    """
    gen, w, _, _ = rig
    h, _ = _real_hidden(wb, REAL_CASES[0])
    x_0 = torch.randn(BATCH, N_ACOUSTIC_CODEBOOK, generator=torch.Generator().manual_seed(0))
    sem = torch.full((BATCH, 1), END_AUDIO_ID, dtype=torch.long)

    got = gen.decode_frame(sem, h, x_0=x_0)
    exp = fref.decode_frame(sem, h, w, x_0=x_0)
    expected_slot = EMPTY_AUDIO_ID + N_AUDIO_SPECIAL
    print(f"\n  [END_AUDIO] frame -> device {got[0, :4].tolist()}, reference {exp[0, :4].tolist()}")
    assert torch.equal(got, exp), "device and reference disagree on an [END_AUDIO] frame"
    assert bool((got == expected_slot).all()), (
        f"acoustic slots are {got[0, :4].tolist()}, expected all {expected_slot}")


@pytest.mark.slow
@pytest.mark.parametrize("cfg_alpha", (1.0, 2.0), ids=lambda a: f"cfg{a}")
def test_frame_codes_at_other_cfg_alphas(rig, wb, cfg_alpha):
    """CFG strengths other than the shipped default must still match the reference."""
    gen, w, _, _ = rig
    h, _ = _real_hidden(wb, REAL_CASES[0])
    x_0 = torch.randn(BATCH, N_ACOUSTIC_CODEBOOK, generator=torch.Generator().manual_seed(0))
    _same_codes(gen(h, x_0=x_0, cfg_alpha=cfg_alpha),
                fref.reference_frame(h, w, x_0=x_0, cfg_alpha=cfg_alpha), f"cfg_alpha {cfg_alpha}")


@pytest.mark.slow
@pytest.mark.parametrize("n_steps", (3, 14), ids=lambda n: f"steps{n}")
def test_frame_codes_at_other_step_counts(rig, wb, n_steps):
    """Step counts other than the shipped 7 must still match the reference, schedule included."""
    gen, w, _, _ = rig
    h, _ = _real_hidden(wb, REAL_CASES[0])
    x_0 = torch.randn(BATCH, N_ACOUSTIC_CODEBOOK, generator=torch.Generator().manual_seed(0))
    _same_codes(gen(h, x_0=x_0, n_steps=n_steps),
                fref.reference_frame(h, w, x_0=x_0, n_steps=n_steps), f"n_steps {n_steps}")
