# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The TRACED frame loop is what ships. This file is the only thing that validates it.

STATUS 6.65 captured the whole per-frame device graph and landed it on the evidence that "every
quality metric identical" -- but that was the WER/MOS quality set, which is a ~50 minute run nobody
does per commit. Every per-block PCC test drives the EAGER path (`prefill`, `step`, `flow` called
directly); every test that drives `generate()` compares device to device only. So until this file,
the shipped path had no numerical gate at all.

WHY THE COMPARISON IS TRACED-vs-EAGER AND NOT TRACED-vs-FP32. The traced loop feeds itself, so it
cannot be teacher-forced, and a free-running device-vs-reference comparison measures nothing:
after the first semantic mismatch the two are generating different sequences, so later frames
compare unrelated trajectories (the trap `gates.compare_codes` documents). Eager is the right
reference here because it is the path the per-block PCC tests already gate against fp32 -- so
traced == eager plus eager == fp32 chains to the answer we want.

The noise draws line up: the eager path draws `randn(B, N_ACOUSTIC_CODEBOOK)` with B=1 and the
traced path draws `randn(1, 1, N_ACOUSTIC_CODEBOOK)`, the same count from the same seeded
generator, and frame 0 is eager in both. So with one seed the two loops see identical x_0 and any
difference is the trace itself.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_traced_frame_loop.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.tt import ttnn_voxtral_pipeline as pipemod  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import fixture_embeds  # noqa: E402

CASES = (0, 2)
MAX_FRAMES = 24


@pytest.fixture(scope="module")
def pipe():
    d = pipemod.open_device()
    p = pipemod.TtVoxtralPipeline(d)
    yield p
    p.close()
    ttnn.close_device(d)


def _run(pipe, embeds, seed, traced, monkeypatch):
    """One generate() with tracing forced on or off.

    `generate()` decides by `TRACE_REGION_SIZE > 0`, so forcing eager is a matter of zeroing that
    module global -- the DEVICE keeps its trace region either way, which is what we want: the only
    difference between the arms is whether the frame graph is captured and replayed.
    """
    monkeypatch.setattr(pipemod, "TRACE_REGION_SIZE", 250 * 1024 * 1024 if traced else 0)
    pipe.backbone.reset()
    frames, _, _ = pipe.generate(embeds, max_frames=MAX_FRAMES, seed=seed, verbose=False)
    return frames, pipe.last_timings.get("traced")


@pytest.mark.slow
@pytest.mark.parametrize("ci", CASES, ids=lambda c: f"case{c}")
def test_traced_matches_eager(pipe, monkeypatch, ci):
    """The shipped traced loop must emit the codes the eager loop emits, on a real prompt."""
    embeds, case = fixture_embeds(ci, pipe.wb)
    eager, was_traced_e = _run(pipe, embeds, 0, traced=False, monkeypatch=monkeypatch)
    traced, was_traced_t = _run(pipe, embeds, 0, traced=True, monkeypatch=monkeypatch)

    assert was_traced_e is False, "the eager arm captured a trace anyway -- the arms are not distinct"
    assert was_traced_t is True, "the traced arm fell back to eager, so this test proved nothing"

    print(f"\n  case {ci} ({case['voice']}): eager {eager.shape[0]} frames, "
          f"traced {traced.shape[0]} frames")
    assert eager.shape == traced.shape, (
        f"frame COUNT differs: eager {eager.shape[0]} vs traced {traced.shape[0]} -- the traced "
        f"loop terminated at a different point, which changes the audio length")

    n_diff = int((eager != traced).sum())
    if n_diff:
        sem_diff = int((eager[:, 0] != traced[:, 0]).sum())
        d = (eager[:, 1:].long() - traced[:, 1:].long()).abs()
        print(f"    {n_diff} of {eager.numel()} codes differ  (semantic {sem_diff}, "
              f"acoustic max |delta| {int(d.max())})")
    assert n_diff == 0, (
        f"{n_diff} of {eager.numel()} codes differ between the traced and eager loops on identical "
        f"inputs and seed -- the trace is not replaying the same computation")


@pytest.mark.slow
def test_traced_waveform_matches_eager(pipe, monkeypatch):
    """Through the codec too, so a divergence cannot hide in a code the codec happens to ignore."""
    embeds, _ = fixture_embeds(CASES[0], pipe.wb)
    eager, _ = _run(pipe, embeds, 0, traced=False, monkeypatch=monkeypatch)
    wav_e = pipe.decode(eager)
    traced, _ = _run(pipe, embeds, 0, traced=True, monkeypatch=monkeypatch)
    wav_t = pipe.decode(traced)
    assert wav_e.shape == wav_t.shape, f"waveform length differs: {wav_e.shape} vs {wav_t.shape}"
    delta = (wav_e - wav_t).abs().max().item()
    print(f"\n  waveform max |delta| {delta:.3e} over {wav_e.shape[-1]} samples")
    assert delta == 0.0, f"traced and eager waveforms differ by {delta:.3e}"
