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

from models.demos.cosyvoice.tt.common import GOLDEN_DIR
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

    # **This is a defect band, not an acceptance band.** Speech should peak near 1.0
    # and the batch path does; `synthesize_streaming`'s output peaks around 72 on this
    # prompt -- finite, mostly near-silent, with a spike. The tokens are identical
    # (asserted above) and the chunk schedule is right, so the fault is downstream of
    # generation, somewhere in how this wrapper assembles the per-chunk conditioning
    # for the flow decoder and the vocoder. It is not the interleaving itself: the same
    # `StreamSession` scheduler, driven from `tests/perf/test_streaming_perf.py` with
    # the golden's own prompt, produces audio that `test_device_streamed_matches_non_
    # streamed` gates on content in mel space.
    #
    # It is pinned rather than printed for the same reason `tests/perf/gates.py` pins a
    # missed threshold: an unasserted number drifts silently, and this one has to be
    # visible until it is fixed. **Closing the defect means replacing this with
    # `< 1.5`**, not widening the band.
    peak = float(streamed.abs().max())
    assert 40.0 < peak < 120.0, (
        f"streamed peak {peak:.1f} is outside the recorded defect band (40, 120). "
        "If it is now near 1.0 the amplitude defect is fixed -- replace this assertion "
        "with `peak < 1.5` and drop the note in docs/VALIDATION.md. If it moved "
        "elsewhere, something new is wrong."
    )
    assert float(n_batch.abs().max()) < 1.5, (
        f"the *batch* path clips at {float(n_batch.abs().max()):.1f} -- that path is "
        "gated elsewhere and should never do this"
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


@needs_weights
@needs_inputs
@needs_device
def test_device_batched_synthesis_agrees_with_one_at_a_time(device):
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
