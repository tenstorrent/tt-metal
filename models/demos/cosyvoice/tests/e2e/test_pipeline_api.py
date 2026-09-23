# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The three entry points, driven the way a caller drives them.

Everything else in `tests/` reaches past `CosyVoiceTTNN` and exercises a stage: the
decoder, the streaming synthesizer, the vocoder. That is the right level for
correctness, and it leaves one thing unchecked — **the wiring**. `synthesize`,
`synthesize_streaming` and `synthesize_batch` each assemble the same three stages in a
different order, and an assembly can be wrong while every part is right.

So this file drives the public API, from a `PromptContext` built by
`scripts/prepare_inputs.py`, and asks the two questions that are specific to the
assembly rather than to any stage:

1. **Does the interleaved schedule generate the same tokens as the batch one?**
   `synthesize_streaming` runs the flow decoder and the vocoder from inside the AR
   decode loop's callback. If that callback perturbed the decoder's state -- a freed
   buffer, a clobbered trace, a cache written between steps -- the tokens would drift.
   Greedy sampling makes the comparison exact.

2. **Does batched generation produce what generating alone produces?**
   `test_device_batched_decode_matches_single` answers that for one decode *step* at
   PCC. This asks it for a whole utterance, through sampling, which is the form that
   compounds: one different argmax diverges everything after it. Gated on the
   agreement rate the bring-up scope already sets for token accuracy, and the
   exact-match prefix is reported next to it.

Inputs come from `--inputs`-style `.npz` files rather than the golden corpus, because
a `PromptContext` is exactly what those files are. The tests skip, with a reason, when
they are absent -- the same contract the golden-dependent tests use.
"""
from __future__ import annotations

import glob
import os

import pytest
import torch

from models.demos.cosyvoice.tt.common import GOLDEN_DIR, pcc
from models.demos.cosyvoice.tt.weights import default_weights_path

HIFT_WEIGHTS = default_weights_path()
FLOW_WEIGHTS = HIFT_WEIGHTS.replace("hift_", "flow_")
LLM_WEIGHTS = HIFT_WEIGHTS.replace("hift_", "llm_")
# `COSYVOICE_INPUTS` names the directory `scripts/prepare_inputs.py --out-dir` wrote;
# the golden directory is the fallback so a tree that has both needs no environment.
INPUTS_DIR = os.environ.get("COSYVOICE_INPUTS", os.path.join(GOLDEN_DIR, "inputs"))


def _cases(n: int):
    """`n` distinct `.npz` prompt files, or an empty list."""
    return sorted(glob.glob(os.path.join(INPUTS_DIR, "*.npz")))[:n]


needs_weights = pytest.mark.skipif(
    not all(os.path.exists(p) for p in (HIFT_WEIGHTS, FLOW_WEIGHTS, LLM_WEIGHTS)),
    reason="export hift, flow and llm weights first",
)
needs_inputs = pytest.mark.skipif(
    len(_cases(2)) < 2,
    reason=f"no prompt .npz in {INPUTS_DIR}; run scripts/prepare_inputs.py --out-dir and set COSYVOICE_INPUTS",
)
# Trace region for the AR decode step, and the L1 the vocoder's prepared conv weights
# need. Same geometry `test_device_end_to_end_rtf` asks for.
needs_device = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 131072, "trace_region_size": 402653184}], indirect=True
)

# Long enough that the streaming path emits at least one chunk *during* generation
# (`token_hop_len + token_overlap_len` = 120 tokens), short enough to keep a full
# synthesis inside a test's time budget. A cap below the chunk size would exercise the
# wiring while never once interleaving, which is the failure mode this file exists to
# rule out.
MAX_TOKENS = 160


def _model(device):
    from models.demos.cosyvoice.tt.pipeline import CosyVoiceTTNN
    from models.demos.cosyvoice.tt.weights import WeightBag

    return CosyVoiceTTNN(
        device, WeightBag.load(LLM_WEIGHTS), WeightBag.load(FLOW_WEIGHTS), WeightBag.load(HIFT_WEIGHTS)
    )


def _agreement(a: list[int], b: list[int]) -> tuple[float, int]:
    """`(match rate over the shorter sequence, length of the agreeing prefix)`."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0
    same = [x == y for x, y in zip(a[:n], b[:n])]
    prefix = 0
    for ok in same:
        if not ok:
            break
        prefix += 1
    return sum(same) / n, prefix


# Two full syntheses of the same utterance, and the flow decoder and the vocoder
# JIT-compile per mel length -- so on a machine that has not run this geometry before,
# the compile bill alone exceeds `pytest.ini`'s 300 s default. It did on p150b, where
# the same test takes 16 s once the kernels are cached. Same marker, same reason, as
# `tests/perf/test_streaming_perf.py`.
@pytest.mark.timeout(1800)
@needs_weights
@needs_inputs
@needs_device
def test_device_streaming_generates_the_same_tokens_as_batch(device):
    """The `on_token` callback must not perturb the decode loop it hangs off.

    Greedy on both sides, so any difference is the callback's doing rather than the
    sampler's. Audio is *not* compared here: the two paths draw the CFM noise and the
    excitation phase differently by design (per chunk against once per utterance), and
    `test_device_streamed_matches_non_streamed` already gates streamed content in mel
    space where that difference does not hide a real one.
    """
    import ttnn
    from models.demos.cosyvoice.tt.pipeline import PromptContext

    ctx, meta = PromptContext.from_npz(_cases(1)[0])
    model = _model(device)

    batch_wav, batch_tokens = model.synthesize(ctx, sampler="greedy", max_tokens=MAX_TOKENS)
    n_batch = ttnn.to_torch(batch_wav).float().reshape(1, -1)
    ttnn.deallocate(batch_wav)

    res = model.synthesize_streaming(ctx, sampler="greedy", max_tokens=MAX_TOKENS, seed=1986)
    streamed = torch.cat([ttnn.to_torch(c).float().reshape(1, -1) for c in res.chunks], dim=1)
    res.free()

    rate, prefix = _agreement(res.tokens, batch_tokens)
    print(f"\n  {meta['mode']}/{meta['lang']}  {len(batch_tokens)} tokens batch, {len(res.tokens)} streaming")
    print(f"    token agreement {100 * rate:.2f} %   exact prefix {prefix}/{min(len(res.tokens), len(batch_tokens))}")
    print(f"    audio  batch {n_batch.shape[1]} samples   streaming {streamed.shape[1]} in {res.n_chunks} chunks")
    print(f"    first audio {res.first_audio_s:.3f} s of a {res.total_s:.3f} s run")

    assert res.tokens == batch_tokens, (
        "the interleaved schedule generated different tokens from the batch one under "
        "greedy sampling -- the on_token callback is perturbing the decode loop"
    )
    assert streamed.shape[1] > 0 and torch.isfinite(streamed).all(), "streamed audio is empty or not finite"

    # Interleaving must not change the signal's scale. The streamed peak is checked for
    # clipping and against the batch path's on the same prompt, because the absolute
    # peak depends on the utterance.
    peak, batch_peak = float(streamed.abs().max()), float(n_batch.abs().max())
    assert peak < 1.5, (
        f"streamed peak {peak:.4f} -- the interleaved path is clipping or corrupted. "
        f"The batch path on the same prompt peaks at {batch_peak:.4f}. A peak in the "
        "tens means the carried StreamState is being overwritten by the decode trace "
        "again; see TtStreamingSynthesizer._carry_store."
    )
    assert peak < max(20.0 * batch_peak, 0.05), (
        f"streamed peak {peak:.4f} is far above the batch path's {batch_peak:.4f} on "
        "the same prompt -- interleaving changed the signal's scale"
    )
    assert batch_peak < 1.5, (
        f"the *batch* path clips at {batch_peak:.4f} -- that path is gated elsewhere " "and should never do this"
    )

    assert res.first_audio_s is not None and res.first_audio_s <= res.total_s

    # If the utterance is long enough to chunk, a chunk must have been emitted before
    # `finish()` -- otherwise the run took the interleaved *code path* without ever
    # interleaving, and the token comparison above proved nothing about scheduling.
    from models.demos.cosyvoice.tt.streaming import StreamConfig

    if len(res.tokens) >= StreamConfig().chunk_size():
        assert res.n_chunks >= 2, (
            f"{len(res.tokens)} tokens is past the {StreamConfig().chunk_size()}-token chunk "
            f"size, but only {res.n_chunks} chunk was emitted -- nothing streamed mid-generation"
        )


# `l1_small_size` is raised for headroom: the vocoder keeps per-geometry
# `conv_transpose2d` state in L1_SMALL and never frees it, and the 131072 the rest of
# this file uses fills after a few distinct mel geometries (`docs/VALIDATION.md`).
@needs_weights
@needs_inputs
@pytest.mark.parametrize("device_params", [{"l1_small_size": 524288, "trace_region_size": 402653184}], indirect=True)
def test_device_batched_synthesis_agrees_with_one_at_a_time(device, monkeypatch):
    """Two utterances through one decode loop against the same two run alone.

    This is the whole-utterance form of `test_device_batched_decode_matches_single`,
    and it is the one that compounds: a single differing argmax diverges everything
    after it. So it is gated on the agreement *rate* the scope sets for token accuracy
    (> 95 %) rather than on exact equality, and the exact-match prefix is printed
    beside it — on Wormhole the batched decode step differs from the single-row one at
    about PCC 0.9985 (see `test_device_batched_decode_matches_single` for why), which
    is enough to move an argmax that sits near a tie.
    """
    import ttnn
    from models.demos.cosyvoice.tt.pipeline import PromptContext

    # `synthesize_batch` needs the CFM estimator's trace cache off for the whole process:
    # with it on, the estimator trace cached by an earlier utterance is live when
    # `generate_batch` captures its decode trace, and the device hangs. Releasing the
    # cached trace, or disabling the cache only around the call, hangs the same way.
    # `TtConditionalCFM` reads the variable once, in its constructor, so it is set
    # before `_model` builds the pipeline.
    monkeypatch.setenv("COSYVOICE_CFM_TRACE_CACHE", "0")

    paths = _cases(2)
    ctxs, metas = zip(*(PromptContext.from_npz(p) for p in paths))
    model = _model(device)

    alone = []
    for ctx in ctxs:
        wav, tokens = model.synthesize(ctx, sampler="greedy", max_tokens=MAX_TOKENS)
        ttnn.deallocate(wav)
        alone.append(tokens)

    together = model.synthesize_batch(list(ctxs), sampler="greedy", max_tokens=MAX_TOKENS)
    assert len(together) == len(ctxs), f"expected {len(ctxs)} results, got {len(together)}"

    worst = 1.0
    print(f"\n  batched synthesis, {len(ctxs)} utterances")
    for i, ((wav, tokens), want, meta) in enumerate(zip(together, alone, metas)):
        n = ttnn.to_torch(wav).float().reshape(1, -1)
        ttnn.deallocate(wav)
        rate, prefix = _agreement(tokens, want)
        worst = min(worst, rate)
        print(
            f"    [{i}] {meta['mode']}/{meta['lang']:<3}  {len(tokens):>3} tokens batched vs {len(want):>3} alone"
            f"   agreement {100 * rate:6.2f} %   exact prefix {prefix:>3}   {n.shape[1]} samples"
        )
        assert n.shape[1] > 0 and torch.isfinite(n).all(), f"row {i} produced empty or non-finite audio"

    assert worst > 0.95, (
        f"batched synthesis diverges from one-at-a-time: worst token agreement {100 * worst:.2f} %, "
        "below the 95 % the scope sets for token accuracy"
    )


@pytest.mark.timeout(1800)
@needs_weights
@needs_inputs
@needs_device
def test_device_consecutive_utterances_with_one_flow_length(device):
    """A second utterance with the same flow length as the one before it completes.

    The CFM solver keeps its trace across calls, and replays it when the next call has
    the same mel length. Between two utterances the vocoder creates per-length state
    and the LLM captures a trace of its own, both after the flow's capture, so a replay
    by the second utterance can overwrite what the vocoder then reads. In a five-language
    sweep that is Korean after Cantonese -- both 328 tokens -- and the device stalled at
    the first read of Korean's waveform in every run. The pipeline now releases the
    flow's trace once each utterance's mel is out (`TtMaskedDiffWithXvec.release_trace`).

    This drives that sequence through the public stages: one utterance end to end, then
    a second prompt's tokens cut to the first's count so the flow length repeats, with
    the LLM running in between as it would for a real second utterance. The second
    utterance's audio is compared with the same utterance synthesised after an explicit
    release, with the same draws.
    """
    import ttnn
    from models.demos.cosyvoice.tt.pipeline import PromptContext, RandomSources

    (ctx_a, _), (ctx_b, _) = (PromptContext.from_npz(p) for p in _cases(2))
    assert ctx_a.mel_len1 == ctx_b.mel_len1, "the two prompts must share a prompt length for the flow lengths to match"
    model = _model(device)

    def to_wav(tokens, ctx, seed):
        torch.manual_seed(seed)
        rng = RandomSources()
        mel, frames = model.tokens_to_mel(tokens, ctx, rng)
        assert model.flow.decoder._trace_id is None, "the flow's trace outlived the utterance that captured it"
        wav = model.mel_to_wav(mel, frames, rng)
        ttnn.deallocate(mel)
        out = ttnn.to_torch(wav).float().reshape(-1)
        ttnn.deallocate(wav)
        return out

    tokens_a = model.text_to_tokens(ctx_a, sampler="greedy", max_tokens=MAX_TOKENS)
    wav_a = to_wav(tokens_a, ctx_a, 1)

    tokens_b = model.text_to_tokens(ctx_b, sampler="greedy", max_tokens=MAX_TOKENS)
    tokens_b = (tokens_b * (len(tokens_a) // max(len(tokens_b), 1) + 1))[: len(tokens_a)]
    wav_b = to_wav(tokens_b, ctx_b, 2)

    model.flow.release_trace()
    wav_b_again = to_wav(tokens_b, ctx_b, 2)

    p = pcc(wav_b, wav_b_again)
    print(
        f"\n  two utterances at one flow length: {len(tokens_a)} tokens each, {wav_a.numel()} samples"
        f"\n  second utterance vs itself after an explicit release: PCC {p:.10f}"
        f"  max|d| {(wav_b - wav_b_again).abs().max():.3e}"
    )
    for name, w in (("first", wav_a), ("second", wav_b)):
        assert w.numel() > 0 and bool(torch.isfinite(w).all()), f"{name} utterance produced empty or non-finite audio"
    assert p >= 0.9999, p
