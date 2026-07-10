# DiffusionGemma — optimize-perf stage (#47465)

> **Historical dg-08 snapshot.** The 4175.7 ms/step, 137.55 ms/layer, and
> sequential-commit numbers below predate true-sparse MoE, OPT-004, batched
> commit, traced denoise, and the L1-residency pass. Do not quote them as the
> current model. The 2026-07-10 final unset-default reproduction is
> **18.844 t/s @48**; the
> `DG_NORM_FULLCANVAS=1` measured 20.68 t/s historically but failed its
> decision-fidelity flip gate and is ineligible as the selected default. Start with
> `perf_campaign_worklog.md`, `selfcond_logits_l1.md`, `selfcond_prechunk.md`,
> `l1_residency.md`, and `early_halt.md`.

Per-device performance optimization of the DiffusionGemma **denoise step / per-block** path on
QB2 (`bh-qbge-06`, P150x4, mesh `(1,4)`, TP=4). The optimization unit is the **denoise step over
the 256-token canvas** (≤48 steps/block) plus the commit — *not* per-token autoregressive decode.
Precision policy (BF16 weights/activations/KV and the established BF16 ordered online-chunk
self-conditioning reduction) and the diffusion decisions (temperature 0.8→0.4, Gumbel-max,
entropy-budget accept, random-token renoise, commit = clean argmax) are preserved. The accepted
prechunk and logits-L1 batches change storage/copy placement only. No
`models/demos/gemma4/` edits.

See `work_log.md` for the full topology audit, per-op tables, candidate tables, and roofline; this
README is the summary + artifact index.

## Current selected default (2026-07-10)

The self-conditioning soft embedding still uses the exact existing sequence of 32 ordered
8192-vocabulary BF16 matmuls and additions, but its tied embedding table is now stored as 32
persistent chunks. This removes 32 repeated device slices per denoise step without changing values,
matmul shapes, or reduction order. Each matching dynamic logits slice, its immediate
`subtract -> exp`, denominator reduction, and ordered denominator accumulator remain in L1.
The chunk matmuls, ordered numerator accumulator, and final divide remain in DRAM.

| final default, selector unset/resolved enabled | value |
|---|---:|
| full 30L traced @48 steady block | **13.5849 s / 18.844 tokens/s** |
| full 30L traced @12 steady block (standalone process) | **4.3122 s / 59.366 tokens/s** |
| derived warmed traced step | **257.575 ms** |
| prior selected default @48 | 13.6817 s / 18.711 tokens/s |
| complete traced generation (prefill + 3 blocks) | 153.9791 s vs 153.341 s prior selected default (**+0.42% regression**) |
| committed/decision identity | exact commits plus all 48 steps × 6 recorded fields in argmax and production chunked-Gumbel modes |

The final reviewed L1-default reproduction is +0.71% over the prior selected default and preserves
the established `a9f0d18709b07d1e` three-block commit digest. Every persisted full-depth @48
clean argmax, sampled token, entropy, accept mask, renoised next-canvas, and explicit clean commit
candidate hash is also exact under identical initial canvas, Gumbel descriptors, and injected
renoise tokens. The production chunked-Gumbel path passes the same 48-step gate and a full-budget
256K capability smoke. Traced throughput remains explicitly RUN-first argmax because
`traced_denoise.py` rejects real Gumbel noise. Full evidence, commands, 256K capacity
accounting, watcher results, cross-process variance, the lack of a complete-generation win, and
final-default policy are in
`selfcond_logits_l1.md` and `selfcond_logits_l1_e2e.json`. The underlying prechunk batch remains
documented in `selfcond_prechunk.md` and `selfcond_prechunk_e2e.json`.

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
- `selfcond_prechunk.md` / `selfcond_prechunk_summary.json` — underlying embedding-prechunk A/B,
  synchronized component timing, final default reproduction, 256K capacity, and watcher evidence.
- `selfcond_prechunk_e2e.json` — exact 10 GiB trace-region provenance, TTFT, all block latencies,
  complete three-block generation time, @12 slope points, and control/candidate/unset-default rows.
- `verify_selfcond_prechunk_decisions.py` / `selfcond_prechunk_decisions.json` — full-depth @48
  exact per-step diffusion-decision gate under identical injected renoise tokens.
- `selfcond_prechunk_gumbel_decisions.json` — equivalent @48 gate with identical production
  chunked-Gumbel descriptors, including explicit per-step commit candidates.
- `qualitative_prechunk.py` / `selfcond_prechunk_qualitative.json` — prompt-correct traced qualitative
  control versus selected default.
- `selfcond_prechunk_256k_chunked.json` / `selfcond_prechunk_watcher_summary.json` — full-budget
  production-sampler 256K capability and complete four-device watcher attach/detach evidence.
- `selfcond_logits_l1.md` / `selfcond_logits_l1_e2e.json` — current selected-default L1 placement,
  independent-process A/B, synchronized component evidence, and required unset-default reproduction.
- `selfcond_logits_l1_decisions.json` / `selfcond_logits_l1_gumbel_decisions.json` — exact @48
  diffusion-decision gates for RUN-first argmax and production chunked-Gumbel.
- `selfcond_logits_l1_256k_chunked.json` / `selfcond_logits_l1_watcher_summary.json` — L1-default
  full-depth 256K production-sampler capability and separate watcher evidence.
- `selfcond_logits_split_rejection.md` / `.json` — post-prechunk dynamic-logits `ttnn.split`
  experiment; targeted component unchanged and canonical warmed @48 throughput -0.12%, so removed.
- `selfcond_vocab_chunk_rejection.md` / `.json` — larger online-softmax grouping reached
  +0.95% warmed @48 but changed the canonical clean-commit digest, so the selector was removed.
- `artifacts/*.log` — raw run logs; `tracy/` — tt-perf-report CSV/tables.

## Exact causal-prefill MoE geometry (2026-07-10)

The stock Gemma4 expert prefill runs the dense all-128-expert `sparse_matmul` in 32-token
chunks with `in0_block_w=1` and an N-divisor-limited core grid. DiffusionGemma now keeps
that exact graph, chunk size, routing, expert set, dtype, and fidelity while selecting the
measured Blackhole TP=4 program geometry locally (`tt/prefill_moe.py`). Gate/up use a
`6x1` grid with K-block 44; down uses `11x4`, K-block 3, and two N tiles/core. The shared
`models/demos/gemma4/` source remains unchanged.

- Dense 256-token layer-0 MoE: **135.51 ms -> 21.16 ms (6.40x)**, elementwise exact
  (`torch.equal=True`, `max_abs=0`).
- Warmed full 30-layer 1024-token causal prefill: **16.3412 s -> 2.6155 s (6.25x)**,
  final logits elementwise exact (`max_abs=0`), or about **62.7 -> 391.5 prompt tok/s**.
- Larger 64/128-token sparse-matmul chunks were explicitly rejected despite small latency
  gains: PCC fell to roughly 0.64/0.48. The selected path retains the correct 32-token chunk.
- The exact geometry is default-on for the supported QB2 shape; set
  `DG_PREFILL_MOE_TUNED=0` for the stock fallback. Unsupported shapes automatically fall back.
- Evidence: `bench_chunk_sweep.py` (component geometry/exactness) and
  `bench_prefill_e2e.py` (alternating warmed full-backbone A/B).

## OPT-004 — matmul-geometry tuning of the 5 sparse-MoE matmuls (rank 2)

The sparse MoE's 5 `ttnn.matmul` calls (`tt/sparse_moe.py`) were never given a `program_config` — the
Lever-A prototype let the op auto-select, reading the expert bank at only ~46 GB/s (~18% of the @256
roofline). OPT-004 adds explicit core-grid + `in0_block_w` geometry (batched gate/up/down force
`per_core_N==Nt` and distribute the 128 experts across the grid → 128 cores / 1 expert each on BH;
gather/combine use 2D configs), opt-in via **`DG_SPARSE_MOE_TUNED=1`** (flag-off = byte-identical
prototype). Targets MoE 10.5 → ~5–6 ms/layer.

- `opt004_matmul_geometry.md` — per-matmul shape/tile/grid/`in0_block_w`/subblock/L1-budget rationale +
  the TTNN op-contract facts (`per_core_N==Nt`, `split_work_to_cores`, 2D M-over-y/N-over-x) that fix
  the geometry, and the expected-impact reconciliation.
- `bench_opt004_matmul_geometry.py` — device verify + candidate-sweep bench: untuned-vs-tuned per matmul
  (PCC ≈ 1.0), a geometry sweep per role, and full-MoE off-vs-on latency + PCC-vs-dense.
  **Write-only; run on QB2 when the device is free.**

## Commit batching (#47557) — the 31.5 s/block commit

The commit row above (256 single-token decode-appends = **31.5 s/block**) is the next lever. The
batched commit collapses those 256 forwards into **one causal masked prefill-append** over the
256-token canvas (~7× commit, ~1.25× block t/s), opt-in via `DG_COMMIT_BATCHED=1`.

- `commit_batching.md` — design + the code-inspection bit-exactness argument (batched KV writes
  == the 256 sequential appends: same positions / per-head norm / RoPE / K/V projections / causal
  masking / cache layout, differing only in prefill-vs-decode kernel numerics).
- `verify_commit_batching.py` — device verify: asserts per-layer KV PCC (batched vs sequential)
  and reports commit_ms before/after. **Write-only; run when the QB2 device is free.**
- Implementation: `tt/commit_batched.py` (+ `reference/attention_mask.py` `causal=True`).
