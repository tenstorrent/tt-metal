# DiffusionGemma — traced serving decode + context/early-halt benchmarks (dg-09, #47466/#47465)

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
  the env dispatcher; forced eager for non-argmax) picks the session's `denoise_block_fn`. The session
  is created ONCE per request in `prefill_forward` and its logits fn caches ONE traced controller
  (captured on block 0), `execute_trace`-replayed every `decode_forward` block — **not** re-captured
  per block. Default follows the env dispatcher, so a plain launch is unchanged.

**Capture-once / replay-many (confirmed by construction):** the controller lives on the persistent
session's `logits_fn` (`_traced_denoise_controller` / `_traced_early_halt_controller`). `decode_block`
calls `denoise_and_commit_block` → the traced entry, which does `getattr(logits_fn, "_traced…", None)`;
it is `None` only on block 0 (→ `_capture`), non-`None` after (→ `execute_trace`). One session = one
capture. `bench_vllm_traced.py` drives this exact session with the explicit `denoise_block_fn` the vLLM
adapter uses.

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
| 4096 | 13.46 | 8.41 | +0.19 (+3072 tok) | ✓ (traced 17.93 t/s) |
| 32768 | 15.27 | 6.60 | +1.81 (+28672 tok) | eager ✓; **traced trace-region-gated** (see below) |

Contiguous KV ≈ **66 KiB/tok** (`(15.27−13.46) GiB / (32768−4096) tok` = 66.2 KiB/tok;
`(13.46−13.27)/(4096−1024)` = 64.8), matching `dg-context-window-oom` (~66 KiB/tok). **DRAM-fit
finding:** the traced serving path also
reserves `DG_TRACE_REGION_SIZE` (~8 GiB for the 48 single-step @30L traces); at `max_model_len=32768`
the free DRAM (6.60 GiB) is already **below** that trace region, so **traced serving is DRAM-gated by
the trace region well before the eager KV ceiling**. Traced serving therefore fits a smaller context
than eager on QB2; right-size `--max-model-len` (or shrink the trace budget via a smaller
early-halt/multistep window, or the #47488 paged path) per `doc/context_contract.json` (target 262144).

Decode throughput is **context-flat** (per-block cost is 48×30-layer MoE compute, not
attention/prefix): eager **6.86 t/s @ max_model_len 4096 → 6.51 @ 32768** (the longer frozen-prefix
read adds only ~5%), matching `vllm_speed_by_context.md`'s eager 7.1–7.4 t/s across prompt_len 10→265;
trace scales the whole block down uniformly (17.93 t/s where the trace region fits). The served
context ceiling / `--max-model-len` follows `doc/context_contract.json` (target 262144); a smaller
`--max-model-len` here is an inner-loop / DRAM-fit choice, **not** a capability cut.

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
serve `32→288→544`. That live run used the **eager** decode path (~7.3 t/s, `vllm_speed_by_context.md`).
The decode compute traced here lives entirely inside `session.decode_block` — the identical code the
live adapter's `decode_forward` delegates to — so the eager→traced decode win measured via the
reduced-surface driver is the win the live server realizes once launched with `DG_VLLM_TRACE=1`
(or `DG_DENOISE_TRACED=1`) + a sized `DG_TRACE_REGION_SIZE`. A fresh live traced run was NOT re-stood-up
here to avoid the device-wedge risk the skills warn about (re-applying the fork patches + a long live
server run); the faithful decode-path measurement is the session driver the adapter wraps.

## Files
- `tt/generate.py`, `tt/serving.py`, `tt/generator_vllm.py` — the trace wiring.
- `doc/vllm_integration/bench_vllm_traced.py` — the harness.
- `doc/vllm_integration/vllmtraced_*.json` — raw benchmark JSON.
