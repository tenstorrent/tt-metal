# DiffusionGemma — optimize-perf stage (#47465)

> **CURRENT CONTRACT — 2026-07-22.** Use `plan.md` Part 0 first. There is exactly one supported
> Metal denoise trace path: model-lifetime up-front capture with reveal masking, a materialized
> full-vocabulary Gumbel source (`device` by default since 2026-07-24, `host` as the IID reference),
> K=48, and one-step/window early halt. Ordinary eager execution is the only fallback. All
> fixed-budget, grouped/multistep, frozen-prefix, per-request, and argmax trace results below are
> historical evidence, not executable current-path guidance.

Current guardrails:

- Up-front trace capture is **default-on** as of 2026-07-24 (`DG_UPFRONT_CAPTURE` defaults to `1`);
  a launch no longer has to set it. `DG_UPFRONT_CAPTURE=0` is the documented opt-out and is the
  required setting when eager per-step trajectory records are needed — replayed traces do not
  produce them.
- Up-front launches still set `DG_UPFRONT_PREFILL_WARMUP_LENS=<every admitted aligned prefill
  length>` and `DG_TRACE_REGION_SIZE=<validated positive reservation>`; both stay **fail-loud** and
  are deliberately not defaulted (the admitted prefill shape list cannot be derived from anything
  the wrapper knows, and the reserved trace region cannot be read back from the device — Metal takes
  it as an open-time constructor argument with no getter — so defaulting it would silence the guard
  without reserving anything, while a trace-region overflow poisons the device and needs `tt-smi -r`).
  `DG_VLLM_MAX_DENOISE_STEPS=48` is still required by the 48-step startup capture.
  `DG_DENOISE_REVEAL_PMAX=<positive tile-aligned served cap>` is now **optional**: when unset the
  fixed span is derived as the tile-rounded served `max_model_len` and logged at startup; an
  explicit value still wins and both paths get identical validation.
- `DG_VLLM_GUMBEL_MODE` defaults to **`host`**. It was `device` from 2026-07-24 for the throughput
  the on-device permuted-vocab RNG buys (no per-step host RNG, no replicated PCIe copy); **reverted
  2026-07-25** because that default corrupts generated text — matched 4-seed A/B, single variable:
  `host` correct 4/4, `device` corrupted 2/4. Root cause is `ttnn.rand`, which is not IID along the
  axis the permuted draw puts the canvas positions on; see
  `doc/decision_fidelity/gumbel_position_correlation.md`. `device` stays selectable for throughput
  work; `chunked` and `argmax` are not materialized full-tensor sources and are rejected under
  up-front capture.
- Reveal masking, non-lazy startup capture, and window-1 early halt are intrinsic. Do not add legacy
  selector flags for them. Every admitted prefill shape must compile before capture; unseen runtime
  shapes fail loudly.
- The historical 18.844 t/s row is warmed same-shape argmax replay with prompt-only prefix
  visibility. It is not first-request TTFT or current multi-request throughput.
- Pure prefill, serving prefill, and prefill+block-0 TTFT are different metrics.
- `DG_PREFILL_RAGGED_LONG` defaults on. Every multi-token prefill uses the ragged top-8 path;
  sequences above 4096 are processed in 4096-token slices. Setting it to `0` is a diagnostic
  control that reproduces the historical dense all-128-expert fallback and 4K→16K cliff.
- `DG_SPARSE_MOE` (the denoise-step MoE) defaults **on** as of 2026-07-24: the true-sparse
  token-gather path is the optimized default (~5× faster/step than dense-128; MoE is ~89% of the
  denoise step). `DG_SPARSE_MOE=0` now **fails loud** (`RuntimeError`) — the ~5×-slower dense-128
  reference is no longer a silent runtime fallback; set `DG_ALLOW_DENSE_MOE=1` to run it explicitly
  for A/B / PCC baselines. `DG_SPARSE_MOE_TUNED` remains default-on (OPT-004 matmul geometry).
- The current pure full-depth 64K-build artifact is
  `context_window_prefill_only_chunkedlong_20260713_msl65536.{json,md}`:
  1K 0.78 s, 4K 1.37 s, 8K 4.15 s, 16K 5.55 s, 32K 10.84 s, 64K 35.58 s.
  These are synchronized first executions per shape and include shape-specific first-use
  compilation. The similarly named artifact without `chunkedlong` is the superseded dense-fallback
  control, not the current default.
- The July-15 fidelity control shows coherent TT output at the intrinsic bf16 floor. Persistent
  serving garbage is not an accepted performance tradeoff. The #48291 doc-0 garbage
  (`níní…1111…`) was root-caused (2026-07-24) as a malformed thinking-template contract in the eval
  invocation — a manual `<|think|>` system message with server `enable_thinking` not applied, so
  the checkpoint template emits an empty-closed thought suffix — NOT a sampler/precision/trace
  defect; fixed by the server-side `enable_thinking=true` contract and device-confirmed
  (doc-0 `exact_match=1`, `\boxed{C}`). See `official_sampler_earlyhalt_20260722.md`.

## Up-front denoise capture (accepted 2026-07-22; default path 2026-07-24)

Up-front capture is the default serving path (`DG_UPFRONT_CAPTURE` defaults to `1`). It captures the
reveal-mask denoise trace during `warmup_model_prefill` and retains its adapter/controller for the
model lifetime. Each request prefills the model-owned KV cache, rebinds the fixed-span adapter in
place, replays the startup trace, and detaches without releasing it. The `DG_UPFRONT_CAPTURE=0`
opt-out retains per-request construction and teardown as the eager fallback; it does not select
another trace implementation. The wrapper destructor invokes the idempotent persistent-release path
before inherited model/mesh teardown.

The mode fails at startup unless `DG_TRACE_REGION_SIZE` is positive. `DG_DENOISE_REVEAL_PMAX` is
optional: when unset it is derived as the tile-rounded served `max_model_len` (passed through
`initialize_vllm_model`) and logged; an explicit value must still be positive and tile aligned.
Reveal masking, non-lazy startup capture, and one-step/window early halt are intrinsic. The fixed
`p_max` is also enforced as the served `prompt + generated` cap before denoise/commit.
Under vLLM, `DG_UPFRONT_PREFILL_WARMUP_LENS` must list every aligned prefill length the server will
admit. Those shapes compile before denoise capture; an unseen runtime length fails loudly because
compiling a new prefill program while traces are active can corrupt trace/CCL state.

Full 30-layer QB2 evidence:

- `upfront_reuse_across_prompts.json`: 32→320→32 aligned prompt spans, A/B outputs differ, A is
  exact on repeat, and `capture_events` remains 1. The 320-token prefill also overwrites the mock
  commit span `[32:288]` (2-step deterministic mechanics gate).
- `upfront_bit_exactness.json`: bounded-memory chunked-Gumbel up-front, existing per-request
  reveal trace, and eager committed SHA256 are identical (2-step deterministic mechanics gate;
  not a sampling-distribution gate).
- `upfront_multi_request_smoke.json`: full K=48 tuned trace, A→B→A, one capture / 192 trace
  executions, exact A roundtrip, prompt-distinct output, coherent decoded text for both prompts,
  checkpoint chat-template metadata, and exact prompt-B equality to a fresh-process per-request
  reveal-trace control.
- `upfront_earlyhalt_gpqa_20260722.{json,md}`: eight sequential real GPQA-Diamond requests with
  traced argmax early halt (K=10–43), one startup trace set, eight clean releases, and zero
  recapture.
- `official_sampler_earlyhalt_20260722.md`: IID full-vocabulary Gumbel sampling plus one-step
  traced early halt, full 30 layers, two sequential requests halting at K=17/19 while exactly
  matching eager controls and reusing one 48-trace startup capture.

These are direct wrapper/session runs on 4× Blackhole p300c; no Tracy or live-server profiling was
used. Commands and the initial untuned timeout control are recorded in `work_log.md`.

### Removed legacy executable paths

The old fixed-step, grouped/multistep, frozen-prefix, per-request, and argmax trace drivers were
deleted. Their Markdown and JSON results remain unchanged as historical evidence. Removed legacy
knobs may still appear in those dated artifacts: `DG_VLLM_TRACE`, `DG_DENOISE_TRACED`,
`DG_DENOISE_TRACED_MULTISTEP`, `DG_DENOISE_MULTISTEP_GROUP`, `DG_DENOISE_EARLY_HALT`,
`DG_DENOISE_EARLY_HALT_WINDOW`, `DG_DENOISE_FROZEN_PREFIX`, `DG_DENOISE_REVEAL_MASK`, and
`DG_DENOISE_LAZY_CAPTURE`. `DG_DENOISE_DEVICE_LOOP` remains a non-Metal eager diagnostic and is
not a traced-serving mode.

> **Historical dg-08 snapshot.** The 4175.7 ms/step, 137.55 ms/layer, and
> sequential-commit numbers below predate true-sparse MoE, OPT-004, batched
> commit, traced denoise, and the L1-residency pass. Do not quote them as the
> model current at that revision. The 2026-07-10 final unset-default reproduction is
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

## Historical selected default (2026-07-10)

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
256K capability smoke. At the time of this July-10 artifact traced throughput was RUN-first argmax
only; production Gumbel tracing landed later and is documented in
`doc/vllm_integration/traced_chunked_gumbel_20260713.json`. Full evidence, commands, 256K capacity
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
device-valued (`entropy_budget_accept`). The retired fixed-step verifier recorded traced replay ==
eager with device canvas feedback; its result is historical and the executable was removed.

## Prefix-cost work off #51080 (2026-07-24)

The #51080 analysis produced a ranked list of levers against the `p_max`-proportional denoise
prefix cost. Outcomes so far, including the one that was refuted:

| item | outcome | evidence |
|---|---|---|
| SDPA `q_chunk` 32 → 64/128 | **refuted** — bit-exact but ≤1%, inside noise. Default stays 32. | `qchunk_sweep_20260724.md`, `sweep_denoise_qchunk.sh` |
| Borrow the KV cache instead of cloning it per step | **landed, default ON** (`DG_PREFIX_BORROW=0` to opt out). Bit-identical `committed_sha256`, ~5.5% off the steady block. | `verify_prefix_borrow.sh` |
| HF sliding-layer key retention on the 25 sliding layers | **landed, default OFF** (`DG_DENOISE_SLIDING_WINDOW=1`). Fidelity fix; unbound blocks bit-identical, enforcement proven live on device. | `per_layer_prefix_spans.md`, `verify_denoise_sliding_window.sh` |
| Bounded 1024-row sliding read (block-resident buffers) | **landed, gated** (`DG_DENOISE_SLIDING_SPAN=1`). SDPA key rows/step 130560 -> 53760 (2.43x), `committed_sha256` bit-identical. | `per_layer_prefix_spans.md`, `verify_sliding_span.sh` |
| Canvas-tail workspace + `fill_cache` | **landed but NOT worth enabling** (`DG_DENOISE_CANVAS_TAIL`, default OFF). Bit-identical, ~1.4% *slower*: the bounded span above already removed the concat it targets. | `canvas_tail_workspace.md`, `verify_canvas_tail.sh` |

Two method notes worth carrying forward, both learned the hard way here:

* **Never time block 0.** It carries program compilation (≈5.5 s vs ≈1.7 s steady in the q-chunk
  sweep), and measuring it produced a bogus "+50% regression" before the harness was fixed to use
  `mean(per_block_latency_s[1:])`.
* **Borrowed tensors need a *buffer* audit, not an object audit.** `ttnn.to_memory_config` returns
  a fresh Tensor object that can alias its input's buffer, so an `is not` check deallocated the
  model KV cache and the next op died with `Input Tensor is not allocated`. See
  `_is_distinct_buffer` in `tt/diffusion_attention.py`.

`demo/serving_smoke.py --upfront` was added to exercise the traced path (48-trace startup capture
+ replay) under the same fail-loud contract the vLLM wrapper enforces, without standing up a
server. It is the vehicle for all three device gates above, and `--num-layers 2` turns a 15-minute
reproduction into ~2 minutes.

## Artifacts

- `work_log.md` — topology audit, per-op tables, candidate tables, roofline reconciliation.
- `perf_summary.json` — per-step / per-block / full-generation summary.
- `bench_sampling_step.py` — traced terminal-step microbench (variants).
- `prof_denoise_step.py` — reduced-layer traced denoise step + prefill(TTFT) + commit profiling.
- `diag_sampling_ops.py`, `diag_argmax_alt.py`, `diag_accept_placement.py` — eager op diagnostics.
- Historical fixed-step trace-safety evidence — retained in this README/work log; its obsolete
  executable verifier was removed.
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
- `selfcond_logits_l1.md` / `selfcond_logits_l1_e2e.json` — July-10 selected-default L1 placement,
  independent-process A/B, synchronized component evidence, and required unset-default reproduction.
- `selfcond_logits_l1_decisions.json` / `selfcond_logits_l1_gumbel_decisions.json` — exact @48
  diffusion-decision gates for RUN-first argmax and production chunked-Gumbel.
- `selfcond_logits_l1_256k_chunked.json` / `selfcond_logits_l1_watcher_summary.json` — L1-default
  full-depth 256K production-sampler capability and separate watcher evidence.
- `selfcond_logits_split_rejection.md` / `.json` — post-prechunk dynamic-logits `ttnn.split`
  experiment; targeted component unchanged and canonical warmed @48 throughput -0.12%, so removed.
- `selfcond_vocab_chunk_rejection.md` / `.json` — larger online-softmax grouping reached
  +0.95% warmed @48 but changed the canonical clean-commit digest, so the selector was removed.
- `context_window_prefill_only_chunkedlong_20260713_msl65536.{json,md}` — current pure-prefill
  64K-build table with long-context chunked-ragged prefill enabled.
- `chunked_ragged_prefill.md` / `chunked_ragged_prefill_bitident.json` /
  `verify_chunked_ragged_prefill.py` — root cause, implementation, exactness, and throughput evidence
  for removing the historical >4096 dense-MoE fallback.
- `artifacts/*.log` — raw run logs; `tracy/` — tt-perf-report CSV/tables.

## Current pure prefill — chunked ragged long (2026-07-13)

The default `tt/prefill_moe.py` dispatcher keeps routing and expert execution on the ragged top-8
path for every multi-token sequence. Above `RAGGED_PREFILL_CHUNK=4096`, it slices the token axis and
reuses the validated ragged program. The 64K-build pure-prefill sweep at `233b88276ab` measured
**2950 tok/s at 16K**, **3022 tok/s at 32K**, and **1842 tok/s at 64K**. The 64K decline is the
separate long-context attention-chunking cost, not a return to dense MoE. See
`context_window_prefill_only_chunkedlong_20260713_msl65536.md`.

## Historical exact causal-prefill MoE geometry evidence (2026-07-10)

The stock Gemma4 expert prefill runs the dense all-128-expert `sparse_matmul` in 32-token
chunks with `in0_block_w=1` and an N-divisor-limited core grid. DiffusionGemma now keeps
that exact graph, chunk size, routing, expert set, dtype, and fidelity while selecting the
measured Blackhole TP=4 program geometry locally (`tt/prefill_moe.py`). Gate/up use a
`6x1` grid with K-block 44; down uses `11x4`, K-block 3, and two N tiles/core. The shared
`models/demos/gemma4/` source remains unchanged.

- Dense 256-token layer-0 MoE: **135.51 ms -> 21.16 ms (6.40x)**, elementwise exact
  (`torch.equal=True`, `max_abs=0`).
- Warmed full 30-layer 1024-token model forward: **16.3414 s -> 2.6158 s (6.247x)**,
  or about **62.7 -> 391.5 model-forward prompt tok/s**. This excludes host embedding and
  logits readback, so it is not an end-to-end TTFT claim.
- Production `prefill_prompt_tokens` (host embedding included, logits readback absent):
  flag-off **16.3419 s** -> selector-unset/default-on **2.6173 s (6.244x)**.
- Final logits and every KV-cache shard across all 30 layers and four devices are elementwise
  exact (`max_abs=0`). A non-aligned 1001-token prompt padded to 1024 is also exact.
- Larger 64/128-token sparse-matmul chunks were explicitly rejected despite small latency
  gains: PCC fell to roughly 0.64/0.48. The selected path retains the correct 32-token chunk.
- The exact geometry is default-on only for the measured Blackhole QB2 `(1,4)` TP=4,
  11x10-grid, BF16, 128-expert shape; set `DG_PREFILL_MOE_TUNED=0` for the stock fallback.
  A context-local dispatcher prevents concurrent Gemma4 calls from inheriting the selection.
- Evidence: `bench_chunk_sweep.py` (component geometry/exactness) and
  `bench_prefill_e2e.py` (alternating warmed full-backbone A/B);
  `prefill_moe_e2e.json` retains commands, raw samples, scopes, and correctness gates.
- Watcher: a separate two-layer production-prefill run passed with
  `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`; all four devices attached/detached,
  the error scan had zero matches, and exact logits/KV/non-aligned gates passed. See
  `prefill_moe_watcher.json` and `prefill_moe_watcher_summary.json`.
- The pre-chunked-long full-depth session context rerun (30 layers, QB2, synchronized wall time)
  measured **239.7–362.9
  prefill tok/s through 32K**, 275.5 tok/s at 65,024 tokens, and 218.4 tok/s at 130,560
  tokens. Relative to the 2026-07-08 issue table, prefill improved **4.05–6.78x**:
  32K fell 562.4 -> 99.4 s, 65K fell 1146.1 -> 236.0 s, and 130K fell
  2418.3 -> 597.7 s. See `context_window_sweep_20260710_summary.json` and its three
  `context_window_sweep_20260710_msl*.json` source artifacts. The original prompt recipe was
  not retained, so this is a directional historical comparison, not a same-input A/B.

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
