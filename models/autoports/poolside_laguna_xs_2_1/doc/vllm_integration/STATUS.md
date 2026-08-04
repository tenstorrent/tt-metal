# Laguna-XS-2.1 on Tenstorrent — STATUS (single record)

**Last updated 2026-08-03.** This is the one canonical status record; it supersedes the per-topic
status/finding docs (listed at the bottom). Keep alongside: `README.md`, `doc/context_contract.json`,
`smoke_test.md` (runbook), `resource_utilization_plan.md` (optimization backlog), `scripts/` (tooling),
`sweep_vllm.tsv` (cited data), and the cited `smoke/` result inputs.

**Serving latency + agent-benchmark report (HTML):**
https://claude.ai/code/artifact/aa902432-303a-43ee-b387-56dcd6bab3b3

---

## Model / mesh
~31B MoE, 40 layers (10 full-attention + 30 sliding-window(512)), 256 experts top-8 + shared, hidden 2048.
P150×4 (1×4 Blackhole mesh), TP=4 / EP=4, FABRIC_1D_RING. Precision: BF16 acts/norms/router, BFP8
attn/dense/shared/KV/LM-head, BFP4 routed experts, fp32/HiFi4 SDPA.

## Implemented (default ON unless noted; each `TT_LAGUNA_*`-gated)
- **Decode SDPA program config — default `k_chunk=64`** (`TT_LAGUNA_DECODE_SDPA_PC=1`), teacher top1 **0.95**.
  Keeps the `max_cores≈16` parallel KV-scan long-context speed. (NOT k128 — correct any doc that says so.)
- **Fused HF rotate-half decode RoPE** (`TT_LAGUNA_FUSED_ROPE`, `rotary_embedding_hf`).
- **Fused MoE combine reduce** (`TT_LAGUNA_FUSED_REDUCE`, `deepseek_moe_fast_reduce_nc`).
- **Prefix caching** (`TT_LAGUNA_PREFIX_CACHE=1`) — Phase B, warm-read bug fixed, full-hit bit-exact; ~95%
  hit in agentic loops; warm TTFT ≪ cold (e.g. ~0.36 s vs 29.8 s @32k).
- **Bounded long-context prefill** (`TT_LAGUNA_PIPE_CHUNK=2048`, commit e3aa655) — routes prefill >2048 onto
  the warmed pipelined path; interleaved-prefill wedge fixed (allocation-free warmed prefill; **no device-
  buffer allocation under a resident trace**).
- **`TT_LAGUNA_PREFILL_FAST` (NEW, default OFF, opt-in)** — restores the larger 8192 outer prefill chunk to
  recover cold-prefill speed at ≤128k. Validated (below). Do NOT enable with `--max-model-len > 131072`
  until an OOM check at that context is done.
- **Batched-decode corruption fix** — plugin `reset_batch=True` full per-step refresh (22–29/100 → 0/100).
  Lives on an **uncommitted plugin diff**.
- **Tooling/serving** — thinking-mode split via `--reasoning-parser deepseek_r1`; native tool-calling via the
  `glm47` parser (both required for agent evals).
- **Spec-decode (opt-in, `tt/spec_decode.py`)** — generator-level ngram, token-exact vs greedy. See Caveats
  for what actually ships.

## Current numbers
- **Advertised = servable context 131072** (HF declares 262144; serving OOMs at 262144). ISL 128…131072 served.
- **Decode config sweep (2026-08-03, teacher top1 / short-ctx t/s/u):** k64 **0.950 / 28.48** (WINNER),
  ttnn-default(=0) 0.95 / 28.4, k128 0.58 / 28.7 (lossy), k32 0.010 / 19.26 (broken). Single-chip layer-PCC is
  NOT a valid discriminator (uniformly degraded by any program config); teacher top1 on the multichip full
  model is.
- **Latency — COLD (APC off, single user, OSL 1024; measured E2EL):**
  | ISL | E2EL | TTFT | decode t/s/u | agg tok/s |
  |---|---|---|---|---|
  | 1,024 | 41.2 s | 0.7 s | 25.3 | 24.9 |
  | 16,384 | 56.1 s | 12.9 s | 23.7 | 18.3 |
  | 32,768 | 75.9 s | 29.8 s | 22.2 | 13.5 |
  | 130,048 | 374 s | 299 s | 13.5 | 2.7 |
  Decode t/s/u holds ~22–25 to 32k (`max_cores` win); E2EL is prefill-dominated at long context. **Warm/cached
  (agent norm)** per-turn TTFT is far lower (~0.36 s @32k, ~1.05 s @131k prior measurement) — cold is worst case.
- **Concurrency:** aggregate scales at short ISL (agg 24.9→63→83 tok/s @ conc 1/8/16, ISL 1k); long-ISL
  all-at-once collapses (prefill saturation — a benchmark artifact, needs ramped arrival). Supported cap
  `--max-num-seqs 8`. Decode t/s/u is batch-flat.
- **Decode config A/B (long-context speed, k64 vs =0, both 0.95 accuracy):** k64 13.5 t/s/u vs ttnn-default 6.6
  t/s/u @~130k (~2×); k128 27.6 t/s/u but lossy (rejected). So k64 = fastest ACCURATE config, not fastest overall.
- **Bounded-prefill fix validation (2026-08-03):** prefill-PCC 33/33 pass (accuracy-neutral), no OOM ≤128k,
  cold TTFT @130k **298.7 s → 208.3 s = 1.43×** with `TT_LAGUNA_PREFILL_FAST=1`. Partial recovery (see Deferred).
- **Agent benchmarks (batch-1, small-N directional, public scaffold):** SWE-bench Verified **1/4 resolved,
  3/4 patched** (astropy-12907 resolved = the canonical 1-line fix; 13398 wrote a full new 97-line transform
  module — right shape, tests failed; 13236 hit step-limit). Terminal-Bench 2.0 **0/1** (make-mips-interpreter:
  agent timed out at the terminus-2 30-min limit after 68 episodes / 2.0M input tokens). poolside self-reported
  reference: SWE-bench Verified 70.9%, TB2.0 37.5% (private agent, multi-attempt).
- **Accuracy:** HumanEval pass@1 ~48.2% (completion mode); prefill top1 0.94 / top5·100 1.00 vs AIME24.
- **Idle silicon (resource_utilization_plan.md):** FLOPs ~1.6–1.9%, DRAM BW ~2.2–2.9%, ~48–60/120 cores,
  power flat vs batch (~324 W).

## Caveats
- **Decode SDPA k128 is LOSSY** (teacher top1 0.58, not bit-identical) — it is **spec-verify-only**; the serving
  default is the accurate k64. Prefill is unaffected by k128. (Corrects the old `decode_sdpa_pc_finding.md`.)
- **Spec-decode — VIABLE ~2× via traced decode-verify (2026-08-03 re-eval; corrects the earlier NO-GO).**
  A clean re-run at K=4 shows the **traced batched decode-verify captures fine and runs at ~63.8 t/s/u @4k**
  (mean committed 2.68/verify) ≈ **~2× the production traced decode (~30 t/s/u)**. The earlier
  `warmup_verify_decode` crash was a **transient** (dirty mesh / K=8), NOT a hard ttnn block. Prefill-verify is
  still break-even (0.93× @32k) and eager-verify is NOT a win vs traced — the **traced** path is the one that
  works. **Accuracy nuance:** the output is a valid greedy trajectory (all tokens target-verified by the
  on-device sampler) but **not bit-identical to plain decode at bf16 near-ties** (verify uses batched/k128 logits
  + on-device sampler vs plain decode's single/k64) — the same class of tradeoff as the shipped k64 config, not a
  correctness break. **CONFIRMED via `--mode selfcheck` (2026-08-03):** g_traced == g_eager_sampler == g_eager
  (host argmax) — the traced verify is provably correct, NO trace bug (the earlier "anchor bug" was a
  misdiagnosis). **Speed is accept-rate-dependent** (batched K+1 verify ≈ one decode step, so mean(m+1) must
  exceed ~2 to win) — but on the TARGET agentic workload accept is HIGH: **B1's real-trajectory replay (679 real
  SWE agent turns) = mean(m+1) 2.5 → ~2×**, because agent context accumulates the agent's own outputs/tool-results
  and its next output re-quotes them (paths, code, diffs) — exactly ngram's sweet spot, and it improves as the
  session grows. On device: 4k code prompt accept 1.68 → ~2× (confirms). The **32k accept 0.00 → 0.79×** was a
  SYNTHETIC cycled-repo prompt whose generation didn't copy — NOT representative of a real agent turn.
  **Verdict: a real ~2× win on real agentic workloads**; ship with a runtime accept-guard (auto-disable below
  break-even) as cheap insurance for the occasional low-copy turn so it never regresses. **To ship:** a
  generation-quality A/B
  (HumanEval spec-on vs -off), a scale test (higher K / longer ctx), and wiring into the served generator
  (batch-1; plug-in point in `spec_decode_eager_design.md`). Refs: `spec_decode_accept/{traced_retry.log,
  b2_verdict.md}`.
- **Concurrent agent load crashes the engine (Bus error)** — a 4-way concurrent SWE attempt died ~2 min in
  (`EngineDeadError`); **agents must run batch-1**. The batched-decode *token-corruption* is separately fixed
  (`reset_batch=True`), but sustained concurrent long-context agent decode is not stable.
- **Prefix-cache partial-hit is NOT bit-reproducible** (FP non-determinism); set `TT_LAGUNA_PREFIX_CACHE=0` for
  bit-exact runs. Determinism contract is in README.
- **Concurrency cap 8** — conc-32 collapses (TTFT into hundreds of s). Decode is correctness-valid to batch 32;
  the cap is latency under contention.
- **BFP4→BFP8 experts: DISPROVEN** as an accuracy win — do not re-litigate.
- **Terminal-Bench harness gotcha:** terminus-2 → litellm needs an `api_key` (any value) via `llm_call_kwargs`
  / `OPENAI_API_KEY`, and non-standard sampling (`top_k`, `chat_template_kwargs`) must go through
  `llm_call_kwargs.extra_body` (bare `--agent-kwarg` values are dropped). Without these it errors before the model.

## Blockers / deferred
- **SESSION UPDATE 2026-08-03 (spec-serving investigation; notes: `SESSION_STATE_specserve.md`, commits
  a19682f/6c46e8e):**
  - **Spec-decode adaptive-K + runtime accept-guard + multi-trace warmup hang-fix SHIPPED** (opt-in, standalone
    driver). 32k low-accept guard 24.4→29.1 t/s/u (~1.0×); 4k copy-heavy ~2.5×. Adversarial-review-clean.
  - **Hybrid-KV re-diagnosed:** NOT the one-line arch-scan fix below. The MODEL is fully hybrid-wired
    (`_HYBRID_KV_CACHE_GROUPS_ENABLED=True`, `get_kv_cache_spec` emits SlidingWindowSpec, grouped-PT plumbing)
    and the plugin already prefers the TT-prefixed arch (`worker.py:191-195`) with `support_hybrid_kv_cache=True`.
    Yet serve_diag boot ran **UNIFORM** (1 group, `num_gpu_blocks_override=2568` → 164,352 tok, 1.25× ctx). So
    the hook still doesn't fire at serving → a **real capacity project** (pool-sizing sentinel `1<<46` + why the
    group split doesn't happen), NOT a one-liner, and **NOT a spec-serving prerequisite** (serving passes the
    uniform page_table). Bug 2 (verify per-layer PT, committed a19682f) future-proofs the hybrid path.
  - **In-adapter spec-serving BLOCKED (architectural):** the spec verify writes KV look-ahead (P..P+K) but vLLM
    allocates blocks incrementally → OOB. **Unblock (no scheduler change): bounded-K within the current block's
    slack** (`K_cap = 63 - pos%64`, block_size 64) keeps writes in already-allocated blocks; adaptive-K/guard
    already cap K/step. Still needs eager-verify-under-resident-decode-trace validation + greedy-parity smoke
    test (mesh-present). Deferred, not shipped.
  - **Served baseline (this boot, APC-on):** ISL1024/OSL128/C1 = **28.2 t/s/u** (Mean ITL 35.5 ≈ P99 39.9),
    healthy through C1→C16→C1. **W1 single-request stall did NOT reproduce** (the 32k/C8 Mean-ITL-131 spike is
    prefill-saturation contention, not the recompile stall — no unsafe-alloc/compile events in that window).
- **Hybrid KV silently dead (correctness-grade — resource_util W5a):** the live plugin `worker.py` scans
  `model_config.architectures` while `platform.py` only prefixes `hf_config.architectures` → the KV-cache spec
  hook never fires → all 40 layers carry full KV (**4.30 GB/dev vs 1.08**), and the pool still pays the
  sliding-window tax. Log proof: `num_gpu_blocks=70368744177664` ⇒ group_size 1; 197,632 tokens (hybrid would
  show 4 groups → 49,408). One-line hook fix + coupled pool resize (W5b) **must ship together** (fixing the hook
  alone cuts full-attn capacity 197,632 → 49,408 tokens). **[2026-08-03: re-diagnosed above — the model IS
  wired + plugin arch-lookup already fixed, yet serving still uniform; deeper than a one-liner.]**
- **Bounded-prefill fix is only partial (1.43×, not the projected 2.4×):** residual ~1.65× gap remains — leading
  suspect the inner `TT_LAGUNA_PREFILL_SDPA_CHUNK` (untouched by the fix) and/or other prefill edits since
  2026-07-24. Follow-up: A/B the inner SDPA chunk + bisect. (`bounded_prefill_regression.md`.)
- **Reproducible multi-minute decode stall (resource_util W1):** 25.3 → 1.92 t/s/u on unseen `(N,w)` batch
  shapes; leading hypothesis = program recompile / buffer alloc under a resident trace. Outranks tuning.
- **262144 context OOMs** on the serving path; restoring it depends on the hybrid-KV + RoPE-share frees.
- **Op-count / core-grid wins NOT DONE (resource_util W2/W3, dispatch_gap):** `max_cores=32` cap vs 64; qkv/norm/
  gate layout micro-clusters. `fused_moe_analysis`: fused reduce-scatter + score-combine are unavailable at
  Laguna's 1×4/DP=1/TP=4 shape (hand-fusing only); only `deepseek_moe_fast_reduce_nc` was adoptable (landed).
- **Prefill serialization (resource_util W4):** all-prefill/all-decode batching; prefill eager, never traced.
- **Deferred: device weight cache (tier3 3.5)** — biggest iteration-speed win (`cache_file_name`/`as_tensor`);
  ~18 min cold boot / ~7 min per full-model build today.
- **Pre-commit BLOCKER (TODO / SESSION_STATE):** model Python (this repo) + the vLLM plugin
  (`model_runner.py` `reset_batch`, `platform.py`/`worker.py` hybrid-KV plumbing, suffix prefill) + ttnn
  `sliding_window_size` live on **uncommitted `.local` diffs across 3 trees** — must fork/branch before
  committing model work. An uninstalled plugin tree at `/home/ttuser/dispatch/...` still carries the unsafe
  changed-only refresh.

## How to serve + bench
- **Serve** (canonical, reconcile any 262144/32 references to the supported values): from `/tmp`,
  `TT_METAL_HOME`=installed tree + server `PYTHONPATH`, `TT_LAGUNA_PIPE_CHUNK=2048 TT_LAGUNA_PREFIX_CACHE=1`,
  `python -m models.common.readiness_check.run_vllm_server --model-dir <M> --hf-model poolside/Laguna-XS-2.1
  --mesh-device P150x4 --stages serve --max-num-seqs 8 --block-size 64 --max-model-len 131072 --tt-config
  '{"trace_region_size":1500000000,"fabric_config":"FABRIC_1D_RING"}'`. For agents add
  `--reasoning-parser deepseek_r1 --enable-auto-tool-choice --tool-call-parser glm47`. Full commands in
  `smoke_test.md`.
- **Bench:** ONLY `vllm bench serve` from `.venv_benchmarks_vllm`; report **ISL/OSL/E2EL + t/s/u + agg tok/s
  (never ms/tok)**. Latency numbers should state cold (APC off) vs warm (APC on).
- **Agents run batch-1** (`--workers 1` SWE, `--n-concurrent 1` TB). Board recovery: `tt-smi -r all` after any
  FABRIC_1D_RING hard-kill (also truncate `readiness_vllm/server.log` — a stale `EngineDeadError` marker aborts
  the next boot). Relaunch, never rebuild (uncommitted `.so`); `cd /tmp` to dodge the JIT-header trap.

---

### Superseded docs folded into this record (safe to delete)
`decode_config_sweep/results.md`, `decode_sdpa_pc_finding.md`, `spec_decode_accept/results.md`,
`spec_decode_plan.md`, `hybrid_kv_status.md`, `prefix_cache_status.md`, `batched_decode_corruption.md`,
`concurrency_envelope.md`, `performance_plan.md`, `dispatch_gap_analysis.md`, `fused_moe_analysis.md`,
`precision_accuracy_analysis.md`, `long_context_prefill_plan.md`, `tier3_status.md`, `TODO.md`,
`SESSION_STATE.md`, `triage/prefill_trace_safe_fix.md`. Retained references: `resource_utilization_plan.md`
(the live optimization backlog — W1/W2/W3/W4/W5 detail), `bounded_prefill_regression.md`, `smoke_test.md`.
