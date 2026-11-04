# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The traced frame loop against the eager one, on real prompts, over FULL utterances.

Eager is the reference because the per-block tests gate eager against fp32; every fixture prompt
runs to its own [END_AUDIO] at three seeds.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_traced_frame_loop.py
"""

import pytest

torch = pytest.importorskip("torch")
ttnn = pytest.importorskip("ttnn")

from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt  # noqa: E402
from models.experimental.voxtral_tts.tt import ttnn_voxtral_pipeline as pipemod  # noqa: E402
from models.experimental.voxtral_tts.tests.reference_helpers import (  # noqa: E402
    case_ids,
    fixture_embeds,
    needs_checkpoint,
)

pytestmark = needs_checkpoint

SEEDS = (0, 1, 2)
# A cap, not a budget: generation stops on [END_AUDIO], and hitting the cap is reported.
MAX_FRAMES = 1024
WAVEFORM_CASE = 3  # runs long, so the codec sees a full-length input

# What the sweep must reach to mean anything -- asserted, not assumed.
SDPA_CHUNK = gpt._SDPA_PRG.k_chunk_size  # 512: sdpa_decode walks the cache in these chunks
MIN_LONG_FRAMES = 400
MIN_LONG_UTTERANCES = 2


@pytest.fixture(scope="module")
def pipe():
    d = pipemod.open_device()
    p = pipemod.TtVoxtralPipeline(d)
    p.warmup(verbose=False)
    yield p
    p.close()
    ttnn.close_device(d)


def _run(pipe, embeds, seed, traced, monkeypatch):
    """One generate() with tracing forced on or off, by zeroing the module's trace-region size."""
    monkeypatch.setattr(pipemod, "TRACE_REGION_SIZE", 250 * 1024 * 1024 if traced else 0)
    pipe.backbone.reset()
    frames, _, _ = pipe.generate(embeds, max_frames=MAX_FRAMES, seed=seed, verbose=False)
    return frames, pipe.last_timings.get("traced")


def _diff(eager, traced):
    """-> a one-line description of how two code tensors differ, or '' if they are identical."""
    if eager.shape != traced.shape:
        return f"frame COUNT differs: eager {eager.shape[0]} vs traced {traced.shape[0]}"
    n = int((eager != traced).sum())
    if not n:
        return ""
    first = int((eager != traced).any(dim=1).nonzero()[0])
    return (
        f"{n} of {eager.numel()} codes differ, first at frame {first} "
        f"(semantic {int((eager[:, 0] != traced[:, 0]).sum())})"
    )


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_traced_matches_eager_over_full_utterances(pipe, monkeypatch):
    """Every fixture prompt x three seeds, each to its natural end: identical codes, identical length."""
    failures, lengths, max_pos, capped = [], [], 0, []
    for ci in case_ids():
        embeds, case = fixture_embeds(ci, pipe.wb)
        P = embeds.shape[1]
        for sd in SEEDS:
            eager, was_e = _run(pipe, embeds, sd, traced=False, monkeypatch=monkeypatch)
            traced, was_t = _run(pipe, embeds, sd, traced=True, monkeypatch=monkeypatch)
            assert was_e is False, "the eager arm captured a trace anyway -- the arms are not distinct"
            assert was_t is True, "the traced arm fell back to eager, so this test proved nothing"
            d = _diff(eager, traced)
            n = int(eager.shape[0])
            lengths.append(n)
            max_pos = max(max_pos, P + n)
            if n >= MAX_FRAMES:
                capped.append(f"case{ci}/seed{sd}")
            print(
                f"  case {ci:2d} seed {sd} ({case['voice']:<16}) P={P:3d}  {n:3d} frames  "
                f"-> pos {P + n:4d}  {'IDENTICAL' if not d else d}",
                flush=True,
            )
            if d:
                failures.append(f"case{ci}/seed{sd}: {d}")

    n_long = sum(1 for n in lengths if n >= MIN_LONG_FRAMES)
    print(
        f"\n  {len(lengths)} utterances, {sum(lengths)} frames, longest {max(lengths)}, "
        f"{n_long} >= {MIN_LONG_FRAMES} frames, deepest cache position {max_pos}, "
        f"capped {len(capped)}",
        flush=True,
    )

    assert not failures, (
        f"{len(failures)} of {len(lengths)} utterances differ between the traced and eager loops on "
        f"identical inputs and seed -- the trace is not replaying the same computation:\n  " + "\n  ".join(failures)
    )
    # Coverage: a sweep that silently stopped short would pass everything above.
    assert n_long >= MIN_LONG_UTTERANCES, (
        f"only {n_long} utterances reached {MIN_LONG_FRAMES} frames -- the sweep never exercised the "
        f"late-utterance replays it exists for"
    )
    assert max_pos > SDPA_CHUNK, (
        f"deepest cache position {max_pos} never crossed sdpa's {SDPA_CHUNK}-position chunk "
        f"boundary, so a chunk-dependent trace fault could not have shown"
    )
    assert len(capped) <= 1, f"{len(capped)} utterances hit the frame cap: {capped}"


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_traced_waveform_matches_eager(pipe, monkeypatch):
    """Through the codec too, on a full-length utterance, so a divergence cannot hide in a code the
    codec ignores."""
    embeds, _ = fixture_embeds(WAVEFORM_CASE, pipe.wb)
    eager, _ = _run(pipe, embeds, 0, traced=False, monkeypatch=monkeypatch)
    wav_e = pipe.decode(eager)
    traced, _ = _run(pipe, embeds, 0, traced=True, monkeypatch=monkeypatch)
    wav_t = pipe.decode(traced)
    assert (
        eager.shape[0] >= MIN_LONG_FRAMES
    ), f"case {WAVEFORM_CASE} produced only {eager.shape[0]} frames; pick a case that runs long"
    assert wav_e.shape == wav_t.shape, f"waveform length differs: {wav_e.shape} vs {wav_t.shape}"
    delta = (wav_e - wav_t).abs().max().item()
    print(f"\n  {eager.shape[0]} frames, waveform max |delta| {delta:.3e} over {wav_e.shape[-1]} samples")
    assert delta == 0.0, f"traced and eager waveforms differ by {delta:.3e}"
