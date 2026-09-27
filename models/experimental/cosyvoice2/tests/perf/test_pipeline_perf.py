# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Non-streaming RTF on distinct utterances (Stage 1's "RTF < 1.0"), enforced through tests/perf/gates.py.

The corpus's six LibriSpeech targets (scripts/corpus.py; prompts from `scripts/prepare_inputs.py`, skipped without
`COSYVOICE2_INPUTS`), each synthesized once through `CosyVoice2TTNN.synthesize` in the reported configuration,
after one warm-up call in the process. Each utterance is a different sentence with a different length, so each is
timed as a real request would be, first-sight geometries included; none is a repeat of an earlier one. The
figure gated is the worst per-utterance RTF; the table and the aggregate are printed for docs/VALIDATION.md.
"""

from __future__ import annotations

import glob
import os

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
        cold = pipe.warmup(ctxs[0])
        runs = [
            (c.meta["case"]["case_id"], pipe.synthesize(c, c.meta["case"]["text"], rng=RandomSources(llm_seed=SEED)))
            for c in ctxs
        ]
    finally:
        pipe.release()

    print(f"\n  warm-up (cold) call: audio {cold.audio_s:.2f} s, wall {cold.wall_s:.2f} s, RTF {cold.rtf:.3f}")
    print("  | case | audio s | tokens | wall s | RTF |")
    for cid, syn in runs:
        print(f"  | {cid} | {syn.audio_s:.2f} | {len(syn.tokens)} | {syn.wall_s:.3f} | {syn.rtf:.3f} |")
    wall, audio = sum(s.wall_s for _, s in runs), sum(s.audio_s for _, s in runs)
    worst = max(s.rtf for _, s in runs)
    line = gates.enforce("rtf_nonstreaming", worst, device, extra=f"worst of 6; aggregate {wall / audio:.3f}")
    gates.report([line], "Stage 1, non-streaming")
