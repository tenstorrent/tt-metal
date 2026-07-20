# DiffusionGemma — traced serving decode + context/early-halt benchmarks (dg-09, #47466/#47465)

> **CURRENT SCOPE — 2026-07-17.** Trace is opt-in (`DG_VLLM_TRACE` defaults off), block 0 is
> capture-inclusive, and growing contiguous-prefix shapes recapture across blocks. July-09/10
> same-ID multi-block numbers below used prompt-only prefix visibility and are historical
> same-shape performance provenance. Use `README.md` and `plan.md` Part 0 for current launch and
> metric semantics.

Enables **Metal TRACE capture/replay in the serving decode path** (a perf feature — NOT Tracy/device
profiling; no profiler was run against a live server, per the `optimize`/`vllm-integration` skills),
then benchmarks output speed vs context and the realized early-halt steps. Companion to `README.md`
(the block-granular vLLM contract + live-serving evidence) and `vllm_speed_by_context.md` (the earlier
eager context sweep).

## The premise, corrected

The goal's premise was that serving hard-calls the eager `denoise_block` and bypasses the traced
controllers. **That is only true for the DEFAULT (no flag).** `tt/serving.py::decode_block` calls
`tt/generate.py::denoise_and_commit_block` *without* a `denoise_block_fn`, so it already routes through
`_resolve_default_denoise_block_fn()` — the same env-gated dispatcher the generator uses. dg-08's
`sweep_serving` already drove the traced + early-halt controllers *through this exact serving session*
(18.20 t/s traced @48). So the traced path was reachable from serving via the `DG_DENOISE_*` env flags;
what was missing was (a) an EXPLICIT, testable trace selection the vLLM adapter can set (not only a
global env flag), and (b) the adapter honoring its own `enable_trace`. Both are now wired.

## What changed (DG-local; no `models/demos/gemma4/` edits)

- `tt/generate.py` — public `select_denoise_block_fn()` (env-gated dispatcher) + `select_traced_denoise_block_fn()`
  (traced-preferring: early-halt > multistep > single-step traced, falling back to single-step traced
  when no per-variant flag is set) + `denoise_flags_select_traced()`.
- `tt/serving.py` — `BlockDiffusionServingSession(..., denoise_block_fn=None)`, threaded into
  `decode_block`'s `denoise_and_commit_block` call. `None` ⇒ the env dispatcher (current default,
  eager unless a flag is set — serving behavior unchanged). A caller can pass the traced loop to
  make trace explicit + deterministic in serving.
- `tt/generator_vllm.py` — honors `enable_trace`: `_resolve_trace_pref()` (from `DG_VLLM_TRACE`, else
  the env dispatcher) picks the session's `denoise_block_fn`. Argmax may use the configured traced
  variant; dynamic `host`, `device`, and `chunked` Gumbel modes force the single-step controller so
  their full-noise/device-seed input can be refreshed between replays. The session is created once
  per request in `prefill_forward` and its logits fn caches one traced controller. A trace set is
  replayed while its visible frozen-prefix length is unchanged. After commit, the prefix grows by
  256; the controller releases the old shape and captures the next block at the new prefix length.
  Eliminating this per-block recapture requires the paged/fixed-shape prefix-input work. Default
  follows the env dispatcher, so a plain launch is unchanged.

**Prefix-aware lifecycle:** the controller lives on the persistent session's `logits_fn`
(`_traced_denoise_controller` / `_traced_early_halt_controller`). `decode_block` calls
`denoise_and_commit_block`; commit advances the mutable contiguous-cache reader from
`prompt_len` to `prompt_len + 256*N`. The next block sees the changed prefix signature, releases
the prior traces, and captures against the expanded KV span.

## TASK 2 — eager vs traced at a fixed context (the trace win)

`bench_vllm_traced.py --max-seq-len 4096 --max-denoise-steps 48 --blocks 2`, QB2 (1,4), tuned MoE,
`DG_TRACE_REGION_SIZE=10 GB`, argmax sampling, prompt "Explain what a diffusion language model is in one
sentence." (runs the full 48 in all three loops, so it is an apples-to-apples @48 commit comparison).

| config | denoise loop | t/s (256/block) | steady block (s) | TTFT (s) | steps/block | halted | commit sha |
|---|---|---:|---:|---:|---|---|---|
| **eager** (current serving default) | `denoise_block` (5 readbacks/step) | **6.86** | 37.32 | 40.0 | [48,48] | [F,F] | `8f015a49e4e31a63` |
| **traced** | `traced_denoise_block` | **17.93** | 14.28 | 123.9¹ | [48,48] | [F,F] | `8f015a49e4e31a63` |
| early_halt | `traced_early_halt_block` | 17.85 | 14.34 | 123.9¹ | [48,48] | [F,F] | `8f015a49e4e31a63` |

¹ The traced TTFT includes the **one-time** 48-single-step-trace capture on block 0 (~110 s); the
served rate is the **steady** per-block latency (block 1). Eager captures nothing. So the capture is
amortized over the request's blocks, not paid per block.

**Trace win = 6.86 → 17.93 t/s = 2.61× on the serving decode block, byte-identical commit.** This is
the ~7→~18 t/s gap `vllm_speed_by_context.md` attributed to the traced loop being "not wired into the
vLLM path" — now closed by routing the serving session's `denoise_block_fn` to the traced loop.
early_halt ≈ traced (17.85; a no-op for this prompt, ~0.5% within block-timing noise).

## TASK 2 — output speed vs context window + DRAM

Post-build per-chip DRAM vs `max_model_len` (measured `ttnn.get_memory_view` DRAM, QB2 21.87 GiB/chip):

| max_model_len | DRAM used (GiB) | DRAM free (GiB) | vs prev | traced-serving fit |
|---:|---:|---:|---|---|
| 1024 | 13.27 | 8.60 | base (weights + KV) | ✓ (dg-08) |
| 4096 | 13.46 | 8.41 | +0.19 (+3072 tok) | ✓ live 48-trace (18.27 t/s @ prompt 256) |
| 8192 | 13.72 | 8.15 | +0.26 (+4096 tok) | ✓ live 48-trace |
| 16384 | 14.24 | 7.63 | +0.52 (+8192 tok) | ✓ live 48-trace |
| 32768 | 15.27 | 6.60 | +1.03 (+16384 tok) | ✓ live 48-trace |

Contiguous KV ≈ **66 KiB/tok** (`(15.27−13.46) GiB / (32768−4096) tok` = 66.2 KiB/tok;
`(13.46−13.27)/(4096−1024)` = 64.8), matching `dg-context-window-oom` (~66 KiB/tok).
The earlier estimate incorrectly treated `DG_TRACE_REGION_SIZE` as immediately resident DRAM.
The live server proves it is a capacity limit: the 48 traces increased used DRAM by only
~1.41–1.44 GiB/chip. At `max_model_len=32768`, both a 32-token prompt and a real 16384-token prompt
captured/replayed all 48 traces, leaving 5.17/5.16 GiB free while trace-resident. The bounded probes
therefore establish 32768 as passing, not as trace-region-gated; they do not establish the absolute
ceiling and did not force 256K.

Allocation scaling and actual-prompt scaling are distinct. A fixed 32-token prompt stayed flat at
18.83–18.89 t/s for allocated limits 4096→32768. The primary warmed,
compile-marker-free 32/256/1024/2048-token targets at msl=4096 measured
18.49/18.27/17.57/16.72 t/s over nine steady blocks each; the 3072 warmed rerun was intentionally
omitted. Longer 6144/8192/16384 prompts measured 12.68/11.88/9.49 t/s. The frozen-prefix read is
therefore material at real long context even though allocation alone is not. Full live metrics,
per-block timing, trace counters, and DRAM snapshots are in `live_context_sweep_results_20260710.md`.

### Fixed denoise-step cap scaling

The real OpenAI path was also swept with isolated servers at
K=1/4/8/12/16/20/24/32/40/48 while holding the logical prompt at 256 tokens.
Output speed was 166.80/108.28/72.94/54.88/44.46/37.06/32.00/25.54/21.34/18.28 t/s.
Warmed denoise time stayed approximately 251.3–251.8 ms/step for K=4–48; each row captured exactly
K traces once and made exactly `4*K` execute calls across four blocks. See
`live_denoise_step_sweep_results_20260710.md`.

This is performance-only evidence: 48 remains model-faithful under #48291, and smaller caps can
change output decisions or quality.

## TASK 3 — realized early-halt steps per block

`early_halt` config (`traced_early_halt_block`, scheme A, threshold 0.005) over a representative prompt
set. **Measured result: a uniform no-op — every prompt ran the full 48 steps (0/5 halted).** This
confirms and BROADENS dg-08's single-prompt finding across a 5-prompt set: under the released config
(seed 0, threshold 0.005, chat-templated prompts) the confidence (entropy) gate never clears 0.005, so
no block halts early. In principle early-halt is data-dependent (dg-08 proved it fires + commits
eager-faithfully when a block converges below the entropy threshold), but under #48291 that condition is
not met on real output here.

`bench_vllm_traced.py` `early_halt` config, seed 0, chat-templated, threshold 0.005, 1 block/prompt:

| prompt | steps/block | halted |
|---|---:|---|
| "Hello, how are you?" | 48 | False |
| "What is the capital of France?" | 48 | False |
| "Explain what a diffusion language model is in one sentence." | 48 | False |
| "Write a short poem about the ocean." | 48 | False |
| "List three uses of a hammer." | 48 | False |

**Realized: avg = 48.0 steps, min/median/max = 48, 0/5 blocks halted below the 48 budget.** So under
the released config (seed 0, threshold 0.005, chat-templated prompts) early-halt is a MEASURED NO-OP
across the whole prompt set — the confidence (entropy) gate never clears 0.005 (mean entropy floors
~0.14–0.51 nats, per dg-08's `probe_halt_gap`). This is the honest answer to "average early-halt
steps" and matches the goal's stated expectation (≈48).

Note on the "35-step halt" in `README.md` § Live serving: that live run used a raw (non-chat-templated)
6-token "Hello" prompt and a different seed/run; it does not reproduce under this measurement's config.
The mechanism is correct and ready (dg-08 proved it halts + commits eager-faithfully when the
confidence gate is satisfiable) — it simply does not fire on real output until #48291 lifts the entropy
floor. Enabling `DG_DENOISE_EARLY_HALT` in serving therefore only adds the per-block halt-scalar sync
for no benefit today: measured here at **~0.5%** (17.93 traced → 17.85 early_halt, within block-timing
noise; dg-08's per-step-sync upper bound was ~2%). So the fixed-48 traced loop is the serving default.

## Correctness

- Traced serving decode commits byte-identical to the eager serving path (and thus to the generator's
  committed argmax) on a prompt that runs the full budget in both — the `committed_sha` match below.
- Eager serving path is unchanged when no flag/trace is selected (`denoise_block_fn=None` ⇒ dispatcher ⇒
  eager `denoise_block`).
- Block-granular semantics preserved: one 256-token block per `decode_block`/`decode_forward`.

**Verdict (measured): all three serving loops commit `8f015a49e4e31a63`** on the diffusion prompt (2
blocks, seed 0) — eager == traced == early_halt, byte-identical. That sha also equals dg-08's 30L
no-halt / fixed-48 traced commit (cross-run byte-identical), so traced serving reproduces the
generator's committed argmax exactly, and the eager serving path is unchanged. Block granularity is
one 256-token block per decode call (`steps=[48,48]` → two committed blocks, `next_pos` advances by
`canvas_length`).

## Live vLLM server (context)

The live tenstorrent/vllm fork server serves DiffusionGemma end-to-end (see `README.md` § Live serving
verification): #47488 runner+scheduler patches applied, real OpenAI requests → HTTP 200, multi-block
serve `32→288→544`. The original 2026-07-03 run used the **eager** decode path (~7.3 t/s,
`vllm_speed_by_context.md`). The 2026-07-10 OpenAI sweep captured 48 traces on block 0 and replayed
the same IDs through later blocks, but that historical run held the denoise prefix at the initial
prompt length. It remains performance provenance, not correct growing-prefix evidence. The
2026-07-13 implementation advances prompt+committed KV and recaptures each changed prefix shape;
live OpenAI rerun remains a separate gate.

## Files
- `tt/generate.py`, `tt/serving.py`, `tt/generator_vllm.py` — the trace wiring.
- `doc/vllm_integration/bench_vllm_traced.py` — the harness.
- `doc/vllm_integration/vllmtraced_*.json` — raw benchmark JSON.

## 2026-07-13 — production Gumbel tracing

The single-step traced controller now supports injected/materialized Gumbel noise in addition to
argmax. It allocates one stable-address Gumbel input before capture and refreshes that buffer in
place immediately before each step replay. This preserves the caller's distinct per-step and
per-block noise without recapture. Materialized Gumbel is forced to one-step trace windows; sharing
one full-vocab tensor across multiple steps inside a grouped trace would silently reuse one draw.

Bounded-memory chunked Gumbel is also trace-enabled. Ordinary `ttnn.rand(seed=<Python int>)` would
bake block 0's seed into replay, so the traced path uses a DG-local `ttnn.generic_op` uniform kernel
that reads the seed from one persistent device tile. The controller refreshes that tile before each
single-step replay; all vocab chunks reuse one persistent uniform buffer. The chunk-selection
constants are also allocated during warmup so capture contains no host writes. The custom kernels
live under `models/experimental/diffusion_gemma/tt/kernels/`; no shared Gemma4 or TTNN source is
modified.

The QB2 gates cover:

```text
test_trace_seeded_uniform_refresh_matches_rand
test_traced_materialized_gumbel_refresh_matches_fixed_loop_across_blocks
test_traced_chunked_gumbel_dynamic_seed_matches_fixed_loop_across_blocks
test_trace_capture_guard_recovers_after_injected_failure

4 passed in 1.20s
```

The dynamic-seed uniform output is bit-identical to `ttnn.rand` for both the capture seed and a
different replay seed. The synthetic materialized and two-vocab-chunk tests keep prefix shape fixed,
capture once, and match their fixed-loop controls exactly. The real reduced growing-prefix run
matches eager end-to-end (`committed_sha256=7b7d…fbba`), while a frozen-prefix A/B has the same
block-0 hash and a different block-1 hash, proving committed KV affects later decisions. The
injected capture-failure gate ends/releases the aborted trace and then successfully captures and
replays a second trace on the same device.

Full-model QB2 serving evidence:

- Reduced 1-layer, K=2, two blocks: 37.19 output tok/s including block-1 recapture; two captures,
  four traces total, four execute calls.
- Full 30-layer, K=2, two blocks: 25.10 output tok/s including block-1 recapture; position
  `32→288→544`.
- Released full 30-layer K=48, two blocks: block-0 TTFT 179.36 s, block 1 including recapture
  180.82 s, **1.42 output tok/s**; 48 traces per prefix shape, 96 captured/96 executed total,
  clean release.

Compact evidence is `traced_chunked_gumbel_20260713.json`. Dynamic Gumbel modes intentionally use
one-step traces; grouped trace windows cannot refresh a per-step noise/seed input inside the window
without changing the captured graph.
