# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wall-clock benchmark: end-to-end LM-planner text-to-music generation.

Measures the FULL hybrid path a customer runs via the LM-planner demo, all on the device:

    user query
      -> plan_song   : 5Hz LM planner (28-layer Qwen3-1.7B) Chain-of-Thought song blueprint
      -> generate_song (CFG, traced) : Qwen3 text enc + condition enc + 24-layer DiT + Oobleck VAE

Reports:
  - lm_plan_s          : LM planner blueprint wall time (autoregressive greedy decode on-device)
  - lm_plan_ms_per_tok : per-token decode cost (grows with prefix - no KV cache yet, O(n^2) prefill)
  - gen_song_traced_s  : generate_song wall time (CFG + trace, the deployment path)
  - lm_e2e_total_s     : plan + generate combined (what the demo takes end to end)

Mirrors bench_e2e_fullgen.py but as a pytest so it runs in CI alongside the other benchmarks. Warmup
excluded (first call JIT-compiles kernels). Skipped if the bundle (incl the LM planner) isn't present.
NOTE: this is a PERF gate (asserts finite, non-trivial output + a generous ceiling to catch gross
regressions), not a numerical-fidelity test - fidelity is covered by test_lm_planner + test_cfg_*.

TRACE STATUS: the generate_song (DiT+VAE) leg uses ttnn trace (use_trace=True, the deployment path).
The LM planner leg does NOT trace: greedy decode grows the prefix each step (variable shape), and the
causal-padding-to-fixed-length alternative was measured to give NO speedup (it recomputes the full
padded length every step) and drifts from the eager tokens mid-sequence. The LM decode is compute-
bound O(n^2) prefill (~85-90 ms/tok, growing) - the real speedup lever is a KV-cache decode path
(compute only the new token), a larger future task tracked in .auto/ideas.md.
"""

import time

import pytest
import torch

import ttnn

from models.experimental.acestep.reference.weight_utils import have_pipeline
from models.experimental.acestep.tt.model_config import AceStepModelConfig
from models.experimental.acestep.tt.pipeline import create_tt_pipeline
from models.experimental.acestep.tests.test_utils import require_single_device

QUERY = "Write an upbeat synthwave song about driving through a neon city at night"
PLAN_TOKENS = 80  # blueprint length for the timed run (enough for caption+bpm+key)
SECONDS = 10.24
INFER_STEPS = 30


@pytest.mark.slow
@pytest.mark.skipif(not have_pipeline(), reason="ACE-Step pipeline bundle (incl 5Hz LM planner) not downloaded")
def test_lm_planner_e2e_perf(device):
    require_single_device(device)

    t0 = time.time()
    pipe = create_tt_pipeline(AceStepModelConfig.from_hf(), device, with_lm=True)
    build_s = time.time() - t0

    # --- warmup (JIT compile all kernel shapes; excluded from timings) ---
    pipe.plan_song(QUERY, max_new_tokens=8)
    pipe.generate_song("warmup", lyrics="", seconds=SECONDS, infer_steps=INFER_STEPS, seed=0,
                       guidance_scale=7.0, use_trace=True)
    ttnn.synchronize_device(device)

    # --- 1) LM planner blueprint ---
    t1 = time.time()
    plan = pipe.plan_song(QUERY, max_new_tokens=PLAN_TOKENS)
    ttnn.synchronize_device(device)
    lm_plan_s = time.time() - t1
    n_new = max(1, len(plan["raw"]))  # chars is a proxy; use PLAN_TOKENS for per-tok
    lm_plan_ms_per_tok = lm_plan_s / PLAN_TOKENS * 1000.0

    caption = plan.get("caption") or QUERY

    # --- 2) generate_song (CFG, traced): the deployment text-to-music path ---
    t2 = time.time()
    wav = pipe.generate_song(
        caption, lyrics="", seconds=SECONDS, infer_steps=INFER_STEPS, seed=0,
        guidance_scale=7.0, use_trace=True,
        bpm=plan.get("bpm"), keyscale=plan.get("keyscale") or "", timesignature=plan.get("timesignature") or "",
    )
    ttnn.synchronize_device(device)
    gen_song_traced_s = time.time() - t2

    lm_e2e_total_s = lm_plan_s + gen_song_traced_s
    audio_s = wav.shape[-1] / pipe.SAMPLE_RATE

    print(f"LM_E2E build={build_s:.1f}s")
    print(f"METRIC lm_plan_s={lm_plan_s:.4f}")
    print(f"METRIC lm_plan_ms_per_tok={lm_plan_ms_per_tok:.2f}")
    print(f"METRIC gen_song_traced_s={gen_song_traced_s:.4f}")
    print(f"METRIC lm_e2e_total_s={lm_e2e_total_s:.4f}")
    print(f"LM_E2E audio={audio_s:.2f}s plan_tokens={PLAN_TOKENS} caption={caption[:60]!r}")

    # Perf/sanity gate: well-formed audio + a generous wall-clock ceiling to catch gross regressions
    # (not a tight fidelity bar - see test_lm_planner for PCC).
    assert wav.dim() == 3 and wav.shape[1] == 2, f"expected [1,2,S] audio, got {tuple(wav.shape)}"
    assert torch.isfinite(wav).all() and wav.abs().max() > 0, "audio non-finite or silent"
    assert lm_e2e_total_s < 120.0, f"LM e2e wall time {lm_e2e_total_s:.1f}s exceeds the 120s ceiling"
