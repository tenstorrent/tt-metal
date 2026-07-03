# DiffusionGemma — optimize-perf stage (#47465)

Per-device performance optimization of the DiffusionGemma **denoise step / per-block** path on
QB2 (`bh-qbge-06`, P150x4, mesh `(1,4)`, TP=4). The optimization unit is the **denoise step over
the 256-token canvas** (≤48 steps/block) plus the commit — *not* per-token autoregressive decode.
Precision policy (bf16 weights/activations/KV, fp32 self-cond softmax & entropy accumulation) and
the diffusion decisions (temperature 0.8→0.4, Gumbel-max, entropy-budget accept, random-token
renoise, commit = clean argmax) are preserved. No `models/demos/gemma4/` edits.

See `work_log.md` for the full topology audit, per-op tables, candidate tables, and roofline; this
README is the summary + artifact index.

## Headline result

The terminal decision path (per denoise step) was dominated by `ttnn.argmax` over the 262144 vocab,
which runs **single-core on TILE input** (1240 ms) and was called **twice per step** (Gumbel sample
+ clean commit argmax). Converting the argmax input to **ROW_MAJOR** makes it **multi-core** and
**bit-identical** (verified exact match to the TILE result), at **14.4 ms** — an **86× per-op** win.
The chain also could not be traced at all: `ttnn.full`/`ttnn.zeros_like` in the accept/renoise steps
raise `TT_FATAL: Writes are not supported during trace capture`. Preallocating those constants makes
the whole terminal path **trace-safe**.

| terminal decision step (argmax RUN-first path, `[1,1,256,262144]`) | ms/step |
|---|---|
| original (TILE argmax ×2 + per-call `ttnn.full`) | **untraceable**; eager ≈ **2494 ms** |
| optimized (ROW_MAJOR argmax + preallocated constants), **traced** warmed | **43.06 ms** |
| + share `z` across gumbel/clean argmax + entropy (`share_z`) | **42.30 ms** (kept) |

~58× faster terminal path and now trace-capturable.

**Full traced denoise step / block (real 26B, reduced-layer L=1/2/4 traced fit → 30L):**

| metric (traced, QB2 (1,4) TP=4) | value |
|---|---|
| per-layer denoise | 137.55 ms/layer |
| fixed overhead (embed + LM head + terminal sampling + final norm) | 49.24 ms |
| **full 30-layer denoise step** | **4175.7 ms** (pre-argmax-fix ≈ 6642 ms → ~37% faster) |
| commit (256 single-token decode-appends, 30L projected) | 31.5 s/block |
| **per block** (fixed 48 steps + commit) | **≈ 231.9 s**; 256 tokens/block; **≈ 0.0043 blocks/s** |
| full generation (1 block) | ≈ 232.6 s (TTFT 0.71 s + 200.4 s steps + 31.5 s commit) |

Traced ≈ eager (~3%), so the denoise path is **op-cost bound, not dispatch-gap bound**: 98.8% of the
step is the per-layer backbone. Measured ~4176 ms/step is **~85–170× the ~24–49 ms bandwidth
roofline** → op-count bound (manual chunked-RoPE, staged-GQA fallback, chunked norms), which is the
identified next optimization target. Full detail in `work_log.md` §2/§3/§4.

## What changed (DiffusionGemma-local only)

- `tt/sampling.py`: new `argmax_last_dim()` (ROW_MAJOR multi-core argmax); used in `gumbel_max`.
- `tt/denoise_loop.py`: `denoise_step` uses `argmax_last_dim` for the clean commit argmax;
  `entropy_budget_accept` / `renoise` / `denoise_step` accept preallocated constants;
  `make_denoise_constants()` + `DenoiseConstants`; trace-safe fixed-step loop
  `run_fixed_denoise_steps()` + `denoise_step_next_canvas()` (device canvas feedback, no host
  readback, fixed ≤48-step count).

## Candidate tables (before/after)

- argmax method sweep — `work_log.md` §2b (ROW_MAJOR chosen; topk k=1/k=32 measured, slower).
- entropy variants — `share_z` (kept, small win), `chunked_entropy` (rejected, 45.4 > 43.1 ms).
- sort/cumsum/scatter placement (net-new accept chain over 256) — `work_log.md` §2e.

## Trace-safe fixed-step scheme

The optimized loop runs a **fixed `max_denoise_steps` (≤48)** count with the accepted canvas fed
step→step **on device** (no host readback of the argmax/entropy/cutoff, no `torch.equal` halt).
Early-halt is data-dependent and cannot shorten a static trace, so the trace-safe shape runs the
full budget; the entropy-budget cutoff stays a device tensor and the sorted scatter indices are
device-valued (`entropy_budget_accept`). Verified by `verify_trace_safe_loop.py` (traced replay ==
eager, device canvas feedback).

## Artifacts

- `work_log.md` — topology audit, per-op tables, candidate tables, roofline reconciliation.
- `perf_summary.json` — per-step / per-block / full-generation summary.
- `bench_sampling_step.py` — traced terminal-step microbench (variants).
- `prof_denoise_step.py` — reduced-layer traced denoise step + prefill(TTFT) + commit profiling.
- `diag_sampling_ops.py`, `diag_argmax_alt.py`, `diag_accept_placement.py` — eager op diagnostics.
- `verify_trace_safe_loop.py` — trace-safe fixed-step loop correctness (device canvas feedback).
- `artifacts/*.log` — raw run logs; `tracy/` — tt-perf-report CSV/tables.
