# Stage A — decode SDPA config accuracy sweep

## Metric choice (important)

The **single-chip** `test_optimized_decoder.py` decode-PCC is **NOT a valid discriminator** for the served
decode config: it is uniformly degraded by *any* SDPA program config (k32/k64/k128 all fail at pos
513/2048), a known artifact of the non-served single-chip path (the summary's "single-chip 6/12 known
sensitivity"). The valid discriminator is **teacher top-1 on the 40-layer multichip full model** — see
`teacher_sweep.log` / the table below.

## Single-chip sweep (harness validation only — see `sweep.log`)

Run in-process across configs on layers 0 (FULL_DENSE) and 4 (FULL_MOE), positions 513 / 2048. Value: it
**reproduced the recorded k128 lossiness exactly** (layer-4 pos-2048 PCC = **−0.01658**, pos-513 = +0.359),
confirming the harness computes real PCC — but every SDPA-pc config (including the shipped k64) fails on the
single-chip path, so this run does **not** choose the config. Retained only as harness validation + a record
of the single-chip sensitivity.

## Decision — teacher top-1 (multichip full model)

Weight cache disabled (`TT_LAGUNA_WEIGHT_CACHE_DISABLE=1`) so the accuracy measurement is untainted by the
unvalidated cache. Populated by `teacher_sweep.log`:

| decode config | teacher top1 | decode t/s/u (short-ctx) | verdict |
|---|---|---|---|
| k128 (old default) | 0.58 (recorded, finding) | 28.7 | LOSSY — rejected |
| ttnn default (`TT_LAGUNA_DECODE_SDPA_PC=0`) | 0.95 (recorded, finding) | 28.4 | accurate but slow @long-ctx (151.7 ms/tok @128k) |
| **k64 (SHIPPED default)** | **0.950** (95/100; top5/100=1.00) | **28.48** | **WINNER — accurate AND fast** |
| k32 (sweep alternative) | 0.010 (1/100 — broken) | 19.26 | rejected: less accurate AND slower |

Measured 2026-08-03, `teacher_sweep.log`, weight-cache off. **Decisive:** k64 wins on both axes — it is the
only config that is both accurate (0.95) and fast (28.48). k32 is catastrophic (top1 0.01) *and* slower
(19.26); k128 is fast but lossy (0.58). So the shipped default needs **no change**; the sweep confirms it.
The `TT_LAGUNA_DECODE_K/EXP/MAXCORES` env knobs remain (defaults reproduce k64) for future sweeps.

Long-context decode **speed** for k64 vs the accurate-but-slow ttnn default (`=0`) is measured **served** in
Stage D (=1 vs =0 at ISL 32k/128k) — short-ctx teacher t/s/u does not expose the long-context win.
