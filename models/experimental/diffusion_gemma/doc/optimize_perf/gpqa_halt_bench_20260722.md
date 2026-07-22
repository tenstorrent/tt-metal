# DiffusionGemma real early-halt speed on GPQA-Diamond (2026-07-22)

Real-prompt follow-up to `context_speed_sweep_20260722.md`. The synthetic sweep's K=48 column
was a worst case — synthetic random tokens never converge, so early-halt never fired. This runs
**real GPQA-Diamond questions** through the DG chat template so the eager loop's data-dependent
early-halt actually fires and we measure the **realized step count K**. Data:
`gpqa_halt_bench_20260722.json`.

- Hardware: QB2 / P150x4 (4× Blackhole p300c), mesh (1,4), TP=4 · full 30L 26B-A4B, bf16.
- Config: **eager** path (no trace flags → default `denoise_block`, host-halts on stable
  clean-argmax + mean entropy < 0.005, `stable_steps_to_halt=1`), K=48 cap, argmax, canvas 256,
  `max_seq_len=4096`, 2 blocks/prompt · production perf profile · 8 real GPQA-Diamond prompts.

## Per-prompt (block 0 = the answer block)

| idx | prompt_len | prefill_s | realized K | halted | block_s | tok/s | coherent? |
|--:|--:|--:|--:|:-:|--:|--:|:-:|
| 0 | 152 | 1.01 | 12 | ✅ | 7.5 | 34.4 | yes |
| 1 | 157 | 0.96 | 10 | ✅ | 5.6 | 45.4 | yes |
| 2 | 182 | 1.01 | 7  | ✅ | 4.3 | 59.6 | yes |
| 3 | 163 | 0.94 | 48 | ❌ | 23.3 | 11.0 | yes (b1 halted K=16) |
| 4 | 377 | 2.76 | 10 | ✅ | 8.1 | 31.6 | yes |
| 5 | 229 | 0.99 | 48 | ❌ | 23.6 | 10.8 | **degenerate** ("the the the…") |
| 6 | 189 | 1.78 | 15 | ✅ | 7.9 | 32.2 | yes |
| 7 | 476 | 1.16 | 13 | ✅ | 7.0 | 36.6 | yes |

## Aggregate

| metric | block 0 | block 1 |
|--|--:|--:|
| realized K (min / median / max / mean) | 7 / 12.5 / 48 / 20.4 | 5 / 14.5 / 48 / 18.0 |
| % halted early | 75% (6/8) | 87.5% (7/8) |
| output tok/s (min / median / max) | 10.8 / **33.3** / 59.6 | 11.0 / 30.8 / 77.6 |
| prefill | ~1–2.8 s, ~150–330 tok/s (plen 152–476) | — |

## Findings

1. **Early-halt fires on real prompts.** 6/8 block-0 halted at **K = 7–15** (median 12.5),
   giving **median ~33 tok/s** and up to ~60 tok/s — vs the synthetic worst case of ~11 tok/s
   (K=48). So on real GPQA data the denoise loop runs **~3× fewer steps** and is **~3× faster**
   than the synthetic K=48 number. Decoded block-0 text is coherent (step-by-step reasoning),
   confirming the halt is genuine, not degenerate.
2. **The 2 non-halting prompts are informative, not noise.** idx 3 didn't converge in block 0
   (K=48) but its block 1 halted (K=16) — a slow-start, not a failure. idx 5 produced
   **degenerate output** ("the the the … ,,,,, \\ \\") and never halted in either block — a
   #48291-class quality failure that correlates exactly with non-convergence (no convergence →
   no halt → K=48). So the K=48 tail = degenerate/hard prompts, not the common case.
3. **Real serving tok/s ≈ 30–35 median**, matching the estimate from the synthetic
   `block(K) ≈ commit + K·step` fit (K≈12 → ~33 tok/s). The two benchmarks are consistent.

## Reproduce

```
DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 DG_DEDUP_ARGMAX=1 PYTHONPATH=/home/zni/tt-metal \
python -u gpqa_halt_bench.py --max-seq-len 4096 --num-prompts 8 --num-blocks 2 --output <out>.json
```
(GPQA-Diamond CSV loaded locally; leave DG_DENOISE_* unset for the eager data-dependent halt.)
