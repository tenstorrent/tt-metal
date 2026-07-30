# Prefill + denoise speed vs context — 65536 build (2026-07-22)

Status: mixed — the prefill table and the cold/warm rule are current; every denoise column is
**provenance only** (measured on the token-gather denoise MoE deleted 2026-07-29).
Owns: serving prefill vs context on one 65536 build; the cold-vs-warm prefill rule; the
`ttft_sweep/` artifacts absorbed from the deleted `ttft_ts_sweep.md`.
See also: [refuted list](../REFUTED.md), [optimize-perf hub](README.md).
Over the 40-line dated-report cap because it absorbs a deleted file and an open contradiction.

- QB2 / P150x4 (4× Blackhole p300c), mesh (1,4), TP=4; full 30-layer 26B-A4B, bf16 weights/KV.
- One `max_seq_len=65536` build (18.8 s, 17.3 GiB DRAM). Eager path, argmax, canvas 256, 2 blocks
  per context (steady = block 1). Synchronized device wall time.
- The env names this run called "the production perf profile" are deleted and do nothing — see
  [flag triage](flag_triage_20260728.md).

## Serving prefill (current)

| context | prefill_s (cold) | prefill_s (warm) | warm tok/s |
|--:|--:|--:|--:|
| 256   | 1.34  | 0.90  | 286   |
| 1024  | 3.87  | 1.11  | 919   |
| 4096  | 3.66  | 1.79  | 2,285 |
| 16384 | 7.63  | 5.29  | 3,095 |
| 32768 | 12.48 | 10.98 | 2,984 |

Cold = first prefill of a new shape and includes shape-specific kernel compile; warm = second.
**Never quote one for the other** (3.87 s vs 1.11 s at context 1024). Serving prefill, pure prefill
and DiffusionGemma "TTFT" are three different metrics — see the three-metrics rule in the
[optimize-perf hub](README.md); pure prefill is [chunked ragged prefill](chunked_ragged_prefill.md).

## Denoise (provenance — token-gather MoE, deleted 2026-07-29)

| context | denoise ms/step | commit s | K=4 block / tok·s | K=48 block / tok·s |
|--:|--:|--:|--:|--:|
| 256   | 463 | 1.16 | 3.01 s / 84.9 | 23.4 s / 10.9 |
| 1024  | 477 | 1.10 | 3.01 s / 85.1 | 24.0 s / 10.7 |
| 4096  | 540 | 0.76 | 2.92 s / 87.8 | 26.7 s / 9.6  |
| 16384 | 699 | 1.05 | 3.85 s / 66.5 | 34.6 s / 7.4  |
| 32768 | 939 | 1.37 | 5.13 s / 49.9 | 46.4 s / 5.5  |

The block columns are a **two-point FIT**, not a measurement: `block(K) ≈ commit + K·step` from the
K=48 and K=4 pair, `tok·s = 256 / block`. The cost model itself lives in [early halt](early_halt.md).

Two findings that survive the MoE replacement: the step has **two regimes** — flat ~465–540 ms up to
context 4096, then rising with prefix cross-attention to ~700 ms at 16384 and ~940 ms at 32768, so
8× context costs only ~1.7× step — and **batched commit is ~1 s per block**, replacing the old ~31 s
sequential commit.

> **OPEN CONTRADICTION (unexplained):** the denoise per-step cost is stated four ways across dated
> reports — ~465–540 ms here; ~0.9 s with MoE ~89% in
> [official sampler](official_sampler_earlyhalt_20260722.md); ~428 ms traced / ~496 ms eager-sync
> with MoE 56.9% in [gumbel overlap](upfront_gumbel_overlap_devicemode_20260724.md); and 4–5.6 s in
> the deleted `ttft_ts_sweep.md` (artifacts below). Each was taken on a different, now-superseded
> MoE path and nothing in the tree reconciles them. Not explained.

**MEASUREMENT TRAP.** The sweep used synthetic deterministic prefix tokens, on which the trajectory
never converges: `halted=false` at every context, so the K=48 column is the **full-budget worst
case**, not early-halt speed. Real-prompt realized K is in
[up-front + eager early halt](upfront_earlyhalt_gpqa_20260722.md).

## Artifacts and repro

- `context_speed_sweep_20260722_msl65536.json`.
- Harness `context_speed_sweep.py --max-seq-len 65536 --prompt-lengths 256,1024,4096,16384,32768
  --output <out>.json` (`PYTHONPATH=/home/zni/tt-metal`; env: see [plan](../../plan.md)) is **not
  present in the repo**, so this sweep cannot be re-run as recorded — only the JSON survives.

### Absorbed from the deleted `ttft_ts_sweep.md` (30 layers, `build_Release`, Tracy OFF)

Two structural findings still hold. **TTFT is dominated by block-0's denoise steps, not prefill**:
solving `TTFT = prefill + block0_steps × per_step` put prefill at ~1.6 s for an 18-token prompt and
~14 s for a 373-token prompt against a 152–211 s TTFT. **Tokens/s tracks the adaptive denoise-step
count, not context** — the medium prompt's second block stopping at 18 steps gave it the highest t/s
of the three contexts, so t/s is content-dependent and not monotonic in context.

Its headline numbers (4–5.6 s/step, 1.3–2.3 t/s, TTFT 152.5 / 197.9 / 211.4 s, and the "~85–110×
the ~49 ms/step weight-traffic roofline" arithmetic built on them) are pre-MoE-work and
**provenance only**; the current step cost and t/s live in the [optimize-perf hub](README.md). The
short-prompt run emitted degenerate text (`text_chars=4`), the expected #48291 argmax-fidelity
effect at that revision. Raw metrics: `doc/optimize_perf/ttft_sweep/metrics_{short,medium,long}.json`.
Repro (env: see [plan](../../plan.md)):

```
DG_CKPT=… python -u models/experimental/diffusion_gemma/demo/serving_smoke.py \
  --num-blocks 2 --canvas-length 256 --max-denoising-steps 48 --max-seq-len 2048 \
  --prompt "<prompt>" --metrics-json out.json
```
