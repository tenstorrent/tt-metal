# Realized early-halt K on real GPQA prompts, and the second-request prefill hang (2026-07-22)

Status: current for the hang root cause and the warmup fix; the realized-K numbers are provenance
(measured on the token-gather denoise MoE deleted 2026-07-29) and are the subject of an open
contradiction below.
Owns: realized early-halt K on the same 8 real GPQA-Diamond prompts, both arms; the root cause of
the second-request prefill hang and why the prefill-warmup whitelist is fail-loud.
See also: [refuted list](../REFUTED.md), [early halt](early_halt.md),
[optimize-perf hub](README.md).

## Realized K — both arms, same 8 GPQA-Diamond prompts

Both arms ran the full 30-layer 26B-A4B on 4× Blackhole p300c, TP=4, canvas 256, K≤48 cap, argmax.
The **eager** arm (`max_seq_len=4096`, 2 blocks/prompt, no trace flags) measured the data-dependent
halt directly; the **up-front traced** arm served eight sequential `r1_gpqa_diamond` requests
through one startup capture with a 10 GiB trace region, `num_concurrent=1` and one 256-token output
block per question. The halt criterion (stable clean argmax + mean entropy < 0.005,
`stable_steps_to_halt=1`, host-evaluated) is owned by [early halt](early_halt.md).

| idx | eager prompt_len | eager block-0 K | halted | eager block_s / tok·s | up-front prompt/cache | up-front K | TTFT (s) |
|--:|--:|--:|:-:|--:|--:|--:|--:|
| 0 | 152 | 12 | yes | 7.5 / 34.4 | 154/160 | 13 | 6.62 |
| 1 | 157 | 10 | yes | 5.6 / 45.4 | 159/160 | 16 | 7.76 |
| 2 | 182 | 7  | yes | 4.3 / 59.6 | 184/192 | 12 | 6.08 |
| 3 | 163 | 48 | no  | 23.3 / 11.0 | 165/192 | 43 | 18.63 |
| 4 | 377 | 10 | yes | 8.1 / 31.6 | 380/384 | 10 | 5.38 |
| 5 | 229 | 48 | no  | 23.6 / 10.8 | 231/256 | 13 | 6.49 |
| 6 | 189 | 15 | yes | 7.9 / 32.2 | 191/192 | 19 | 8.85 |
| 7 | 476 | 13 | yes | 7.0 / 36.6 | 478/480 | 17 | 8.18 |

Eager aggregate: realized K min/median/max/mean **7 / 12.5 / 48 / 20.4** in block 0 and
**5 / 14.5 / 48 / 18.0** in block 1; halted-early **75% (6/8)** block 0 and **87.5% (7/8)** block 1;
output tok/s min/median/max 10.8 / **33.3** / 59.6 (block 0) and 11.0 / 30.8 / 77.6 (block 1) —
roughly 3× the ~11 tok/s synthetic K=48 worst case in
[context speed sweep](context_speed_sweep_20260722.md);
prefill on these 152–476-token prompts took ~1–2.8 s (~150–330 tok/s), far below the long-context
sweep rate because short prompts cannot amortize. Up-front arm: K = 13/16/12/43/10/13/19/17, TTFT
5.38–18.63 s, all eight released cleanly.

Two readings of the K=48 tail. **Non-convergence and degenerate output are the same event**: eager
idx 5 emitted "the the the … ,,,,, backslash" and never halted in either block. But **a block-0
K=48 is not automatically a failure**: eager idx 3 hit the cap in block 0 and halted at K=16 in
block 1 — a slow start, not a collapse.

> **OPEN CONTRADICTION (unexplained):** these halts (eager 6/8 at K=7–15; up-front K=10–43) were
> measured on the token-gather denoise MoE that the tree elsewhere says can NEVER halt, because its
> denoise entropy plateaus around 0.46 against the 0.005 threshold — the stated reason for deleting
> it in `7417bd7d69d`. Both readings are recorded in the tree (here and in
> [early halt](early_halt.md)) and nothing reconciles them. Not explained.

**MEASUREMENT TRAP.** The up-front arm's lm-eval exact-match score was 0 purely because a one-block
256-token cap truncates GPQA reasoning before its final answer. This run is a **trace-lifecycle
gate, never an accuracy claim**. GPQA's own scoring traps are owned by
[decision fidelity](../decision_fidelity/README.md).

## Second-request prefill hang — root cause and fix

Reproduced exactly: request 0 completed at K=13; request 1 reached `prefill_device_begin` but never
`prefill_device_end`; live `tt-triage` found all four devices in the causal-prefill `AllBroadcast`
writer, waiting on its semaphore.

**ROOT CAUSE.** Up-front capture had warmed only the 32-token mock prefill, so the first real
160-token prompt compiled a NEW prefill shape while denoise traces were already active. That
post-capture compilation/allocation corrupted trace/CCL state, and the next causal prefill stalled
in the `AllBroadcast` writer.

**REFUTED:** the hang is not early-halt-specific — see [refuted list](../REFUTED.md).

**CONFIRMED FIX.** Warm every admitted aligned prefill length during vLLM's compile-only warmup
phase, *before* denoise trace capture, and fail loudly for any runtime request whose aligned length
was not warmed. This is exactly why `DG_UPFRONT_PREFILL_WARMUP_LENS` is fail-loud and deliberately
not defaulted. Prompts are padded up to the warmed aligned length (154→160, 165→192, 380→384,
478→480), which is what the whitelist has to enumerate. The rest of the up-front capture contract
is owned by the [optimize-perf hub](README.md).

## Artifacts and repro

- `upfront_earlyhalt_gpqa_20260722.json`; hang triage
  `triage/upfront_earlyhalt_gpqa_hang_tt-triage.txt` and
  `triage/upfront_earlyhalt_gpqa_hang_summary.txt`.
- `gpqa_halt_bench_20260722.json` (eager arm). GPQA-Diamond is loaded from a **local CSV** — the HF
  dataset is gated. Harness `gpqa_halt_bench.py --max-seq-len 4096 --num-prompts 8 --num-blocks 2
  --output <out>.json` (env: see [plan](../../plan.md); leave `DG_DENOISE_*` unset for the eager
  data-dependent halt) is **not present in the repo**, so the eager arm cannot be re-run as
  recorded — only the JSON survives.
