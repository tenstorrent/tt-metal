# Official sampler + traced early halt — and the #48291 doc-0 garbage root cause (2026-07-22)

Status: current for the #48291 doc-0 root cause and its eval-invocation fix contract; everything
about `host` Gumbel is provenance-only (that mode was deleted 2026-07-28) and the TTFT table
measures two MoE paths that no longer exist.
Owns: the thinking-template root cause of the #48291 doc-0 GPQA garbage, and the released-sampler
contract (schedule, inert HTTP params, trace selection).
See also: [refuted list](../REFUTED.md), [early halt](early_halt.md), [optimize-perf hub](README.md).

## #48291 doc-0 garbage — root cause and fix (2026-07-24)

**REFUTED causes** of the doc-0 GPQA garbage (`níní…1111… the the…`): it was **not** the host-Gumbel
sampler, **not** bf16 precision, **not** the reveal mask and **not** the trace.

**ROOT CAUSE — a malformed thinking-template contract in the eval invocation.** An eval invocation
injected a literal `<|think|>` *system* message while the server's `enable_thinking=true` was not
applied. The checkpoint `chat_template.jinja` then (under `add_generation_prompt` + `not
enable_thinking`) appends an empty, already-closed thought channel `<|channel>thought\n<channel|>`.
Asked to answer after a finished-empty thought, block 0 never converges (48 steps,
`halted=false`), commits the clean argmax of a near-flat canvas into frozen KV, and later blocks
condition on that noise and collapse to a `1`/backslash repetition attractor. That attractor
**halts early** — a constant canvas is stable with near-zero entropy — so "halted early" is by
itself **not** evidence of a good answer.

**DEVICE-FREE REPRODUCTION** of the template defect: call the checkpoint tokenizer's
`apply_chat_template`; `enable_thinking != true` yields the empty-closed-thought suffix,
`enable_thinking=true` yields a clean `<|turn>model\n` prompt.

**FIX — an eval-invocation contract, not an inference-server code change.** Rely on server-side
`--default-chat-template-kwargs '{"enable_thinking":true}'` and never inject a manual `<|think|>`
token. `run_upfront_gpqa.sh THINKING_MODE=1`, the tt-inference-server DG model spec and its GPQA
eval config (`use_chat_api=True`) already use this contract; the garbage artifacts came from an
older ad-hoc invocation that bypassed it.

**Device confirmation** (P150x4, 2026-07-24, `msl=4096 gen=3072`, thinking): doc-0 `exact_match=1`,
ends `\boxed{C}`, every block halting early (K=16/9/8/5), against the failing run's blocks 0–8 all
48-step `halted=false`. The malformed prompt is detectable as `prompt_len` 161 vs the correct 157.

**OPEN ROBUSTNESS GAP, never closed here:** the K-step cap commits a non-halted block's argmax
anyway (matching upstream Transformers `generation_diffusion_gemma.py`), so ONE bad block poisons
every later block through the KV. A flag/abort-on-non-halt guard in `tt/generate.py` /
`tt/traced_denoise.py` is still owed.

> **OPEN CONTRADICTION (unexplained):** this file's 2026-07-28 banner credits the canvas-attends-
> prefill-pad-keys language-drift fix to `d0936d4da4f`, and
> [gumbel overlap](upfront_gumbel_overlap_devicemode_20260724.md) says the same; elsewhere the tree
> attributes the shipped default-ON fix to `205e87956cc`. Both attributions are in the tree and they
> disagree. Not explained.

## Released-sampler contract

- Released denoise temperature schedule `t_max=0.8` → `t_min=0.4`; released entropy budget `0.1`;
  released halt thresholds `stability_threshold=1`, `confidence_threshold=0.005`. The halt criterion
  itself and its firing-status contradiction are owned by [early halt](early_halt.md).
- HTTP `temperature`, `top_k`, `top_p` and `seed` are **not** checkpoint sampler parameters and are
  not wired into the model-owned denoise loop, so the GPQA launcher does not present them as
  effective settings.
- Dynamic Gumbel previously selected the fixed-budget single-step trace even with early halt
  enabled; it now selects `traced_early_halt_block` when the halt window is one. **A halt window
  larger than 1 must fail loudly**, because the Gumbel seed/noise has to be refreshed between
  denoise steps.
- `chunked` is refused for the official-quality launcher (1024-wide TT RNG, known innermost-axis
  distribution bias) and the up-front validator refuses `chunked`/`argmax` as non-materialized — see
  [refuted list](../REFUTED.md). The current Gumbel-mode default is owned by the
  [optimize-perf hub](README.md).
- The full-vocabulary device Gumbel temporary padded to an 8 GiB allocation and OOM'd on the
  full-depth configuration before the shape fix — see
  [gumbel overlap](upfront_gumbel_overlap_devicemode_20260724.md).

## Device gate (full 30-layer, 4× Blackhole p300c, TP=4)

Two sequential prompts through ONE persistent capture, reveal `p_max=4096`, 48-step cap, early-halt
window 1: halted at **K=17** and **K=19**; committed blocks, halt decisions and realized K matched
eager early halt exactly under the same canvas/Gumbel/renoise seeds; `capture_events` stayed 48;
request 1 reported `gumbel_mode=materialized` and `reveal_mask_reuse`; both released normally.

## TTFT / block throughput — provenance only (2026-07-24, `msl=4096`, `gen=3072`, thinking)

DiffusionGemma "TTFT" means time to the first whole 256-token block (prefill + block-0 denoise) and
is structurally unlike autoregressive TTFT; the three-metrics rule is in the
[optimize-perf hub](README.md).

| metric | dense-128 (path deleted) | token-gather sparse MoE (path deleted) |
| --- | --- | --- |
| per denoise step | ~4.95 s | ~0.9 s |
| doc-0 TTFT (first block) | 79.5 s | 14.5 s |
| doc-0 steady t/s (decode-only) | 5.2 | 33.1 |
| doc-1 TTFT | 145.4 s | 18.7 s |
| doc-1 steady t/s | 2.6 | 20.4 |

Both columns describe MoE paths deleted 2026-07-29, so neither is a current cost; the ~89% MoE share
quoted with them was later measured at 56.9% on the same doomed path. Cross-run block counts
differed under the tuned sparse path (bf16 chaos): doc-0 ran 5 vs 7 blocks but stayed correct
(`\boxed{C}`), while doc-1 flipped answer across bf16 configs — a #48291 decision-fidelity
sensitivity, not this bug. bf16 chaos amplification is owned by
[decision fidelity](../decision_fidelity/README.md).

> **OPEN CONTRADICTION (unexplained):** the denoise per-step cost is stated as ~0.9 s here,
> ~465–540 ms in [context speed sweep](context_speed_sweep_20260722.md), ~428–496 ms in
> [gumbel overlap](upfront_gumbel_overlap_devicemode_20260724.md) and 4–5.6 s in the deleted
> `ttft_ts_sweep.md`. Each was measured on a different, now-superseded MoE path and nothing in the
> tree reconciles them. Not explained.
