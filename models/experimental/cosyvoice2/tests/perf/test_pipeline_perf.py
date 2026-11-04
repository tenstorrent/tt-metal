# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Non-streaming RTF (Stage 1's "RTF < 1.0") and streaming's first audio and RTF (Stage 3's targets) on distinct
utterances, enforced through tests/perf/gates.py. Run each test in its own process.

The Stage 1 protocol (docs/VALIDATION.md): distinct utterances with warmed buckets.
- `CosyVoice2TTNN` in the reported configuration (bucketed) runs `warmup_buckets()` first. Its time is printed as
  the start-up cost.
- It then synthesizes the corpus's six LibriSpeech targets once each (scripts/corpus.py; prompts from
  `scripts/prepare_inputs.py`, skipped without `COSYVOICE2_INPUTS`).
- Each target is a different sentence landing in an already-warmed bucket, and none repeats an earlier request.

The gated figure is the worst per-utterance RTF; the table and the aggregate are printed too.

Streaming follows `demo.py --stream`: `warmup_streaming()` after `warmup_buckets()`, then the same six targets
streamed once each (`synthesize_stream`). Both Stage 3 targets are missed and recorded as `Misses()`: the gated
figures are the worst utterance's time to first audio (ms) and streaming RTF, each held inside its recorded band.
"""
from __future__ import annotations

import glob
import os
import time

import pytest

from models.experimental.cosyvoice2.tests.perf import gates

INPUTS_DIR = os.environ.get("COSYVOICE2_INPUTS", "")
SEED = 1986


@pytest.mark.skipif(not INPUTS_DIR, reason="set COSYVOICE2_INPUTS to scripts/prepare_inputs.py's --out-dir")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 50_000_000}], indirect=True)
# No timeout: every new sequence length compiles kernels and verifies conv geometries, minutes per utterance on a
# cold kernel cache, and a device job is never killed mid-op (pytest.ini sets 300 s).
@pytest.mark.timeout(0)
def test_device_nonstreaming_rtf_distinct_utterances(device):
    from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2TTNN
    from models.experimental.cosyvoice2.tt.prompt import PromptContext, RandomSources

    ctxs = [PromptContext.from_npz(p) for p in sorted(glob.glob(os.path.join(INPUTS_DIR, "*.npz")))]
    ctxs = [c for c in ctxs if c.meta["case"]["set"] == "librispeech"]
    assert len(ctxs) == 6
    pipe = CosyVoice2TTNN(device)
    try:
        t0 = time.perf_counter()
        pipe.warmup_buckets()
        warmup_s = time.perf_counter() - t0
        runs = [
            (c.meta["case"]["case_id"], pipe.synthesize(c, c.meta["case"]["text"], rng=RandomSources(llm_seed=SEED)))
            for c in ctxs
        ]
    finally:
        pipe.release()
    # the protocol's premise: every request ran on warmed geometries, none of which was evicted
    assert pipe.conv_cache_evictions() == 0

    print(f"\n  start-up: warmed every bucket in {warmup_s:.1f} s")
    print("  | case | audio s | tokens | wall s | RTF |")
    for cid, syn in runs:
        print(f"  | {cid} | {syn.audio_s:.2f} | {len(syn.tokens)} | {syn.wall_s:.3f} | {syn.rtf:.3f} |")
    wall, audio = sum(s.wall_s for _, s in runs), sum(s.audio_s for _, s in runs)
    worst = max(s.rtf for _, s in runs)
    line = gates.enforce("rtf_nonstreaming", worst, device, extra=f"worst of 6; aggregate {wall / audio:.3f}")
    gates.report([line], "Stage 1, non-streaming")


@pytest.mark.skipif(not INPUTS_DIR, reason="set COSYVOICE2_INPUTS to scripts/prepare_inputs.py's --out-dir")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 50_000_000}], indirect=True)
@pytest.mark.timeout(0)  # a device job is never killed mid-op (pytest.ini sets 300 s)
def test_device_streaming_first_audio_and_rtf_distinct_utterances(device):
    from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2TTNN
    from models.experimental.cosyvoice2.tt.prompt import PromptContext, RandomSources

    ctxs = [PromptContext.from_npz(p) for p in sorted(glob.glob(os.path.join(INPUTS_DIR, "*.npz")))]
    ctxs = [c for c in ctxs if c.meta["case"]["set"] == "librispeech"]
    assert len(ctxs) == 6
    pipe = CosyVoice2TTNN(device)
    try:
        t0 = time.perf_counter()
        pipe.warmup_buckets()
        pipe.warmup_streaming()
        warmup_s = time.perf_counter() - t0
        runs = [
            (
                c.meta["case"]["case_id"],
                pipe.synthesize_stream(c, c.meta["case"]["text"], rng=RandomSources(llm_seed=SEED)),
            )
            for c in ctxs
        ]
    finally:
        pipe.release()
    assert pipe.conv_cache_evictions() == 0

    print(f"\n  start-up: both warm-ups in {warmup_s:.1f} s")
    print("  | case | audio s | tokens | chunks | first audio s | wall s | RTF |")
    for cid, syn in runs:
        print(f"  | {cid} | {syn.audio_s:.2f} | {len(syn.tokens)} | {len(syn.chunks)} | {syn.first_audio_s:.3f} | "
              f"{syn.wall_s:.3f} | {syn.rtf:.3f} |")  # fmt: skip
    wall, audio = sum(s.wall_s for _, s in runs), sum(s.audio_s for _, s in runs)
    firsts = [s.first_audio_s * 1000 for _, s in runs]
    lines = [
        gates.enforce("ttfp_ms", max(firsts), device, extra=f"worst of 6; best {min(firsts):.0f} ms"),
        gates.enforce(
            "rtf_streaming", max(s.rtf for _, s in runs), device, extra=f"worst of 6; aggregate {wall / audio:.3f}"
        ),
    ]
    gates.report(lines, "Stage 3, streaming")
