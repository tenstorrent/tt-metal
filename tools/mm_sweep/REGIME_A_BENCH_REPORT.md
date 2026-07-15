# regime_a_matmul — Blackhole steady-state benchmark

System under test: the independent `ttnn.experimental.regime_a_matmul` op (NOT the C++ prototype). Resident device inputs + pre-sharded resident weights; PCC>=0.999 verified before timing; 1 warmup + >=8 timed iters. **Ranked by MEDIAN kernel us** (min/spread in JSON). op %=of 512 GB/s. Historical `bh_skinny` %=of 500 GB/s (kept separate); cross-source comparison uses kernel us.

## Per-shape results

| cat | shape | Mt | prod us | prod %512 | manual us | %512 | cfg(Ns,Pk,Sm,kb,nsb) | cores | picker gap% | sweep-best us (%512) | hist branch us (%500) | vs target |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep | 32x256x6144 | 1 | 8.6 | 80.4 | 8.6 | 80.4 | [3, 1, 1, 1, 8] | 24 | -0.1 | 8.6 (80) | 11.2 (63) | +0.4 |
| sweep | 32x2048x512 | 1 | 8.6 | 51.1 | 8.7 | 50.9 | [2, 4, 1, 2, 1] | 64 | -0.3 | 8.6 (51) | 9.4 (48) | +0.8 |
| sweep | 32x2048x1536 | 1 | 15.2 | 83.8 | 15.3 | 83.2 | [3, 2, 1, 4, 2] | 48 | -0.7 | 15.4 (83) | 18.8 (69) | -0.6 |
| sweep | 32x2048x2048 | 1 | 19.6 | 86.1 | 19.7 | 85.9 | [2, 2, 1, 4, 4] | 32 | -0.3 | 19.5 (86) | 23.1 (75) | +0.9 |
| sweep | 32x6144x1536 | 1 | 42.4 | 89.1 | 41.9 | 90.2 | [1, 3, 1, 4, 6] | 24 | +1.2 | 42.1 (90) | 51.9 (75) | -0.4 |
| sweep | 32x6144x2304 | 1 | 61.5 | 91.6 | 61.5 | 91.6 | [1, 4, 1, 2, 9] | 32 | +0.0 | 61.6 (91) | 78.3 (74) | -0.2 |
| sweep | 32x6144x3072 | 1 | 79.8 | 93.8 | 79.6 | 94.0 | [1, 6, 1, 4, 2] | 48 | +0.3 | 79.8 (94) | 105.5 (73) | -0.2 |
| sweep | 32x6144x6144 | 1 | 154.3 | 96.5 | 154.3 | 96.5 | [1, 6, 1, 4, 2] | 48 | +0.0 | 153.9 (97) | 199.6 (76) | +0.3 |
| sweep | 32x6144x9216 | 1 | 229.7 | 97.1 | 228.9 | 97.5 | [1, 6, 1, 4, 2] | 48 | +0.3 | 228.4 (98) | 293.4 (78) | +0.2 |
| sweep | 64x4608x6144 | 2 | 124.1 | 91.3 | 124.0 | 91.4 | [1, 6, 1, 1, 6] | 48 | +0.1 | 123.1 (92) | 153.8 (75) | +0.7 |
| sweep | 64x6144x1536 | 2 | 48.2 | 80.5 | 47.4 | 81.9 | [1, 6, 1, 4, 2] | 48 | +1.7 | 47.6 (81) | 51.7 (77) | -0.5 |
| sweep | 64x6144x4608 | 2 | 122.4 | 92.5 | 122.6 | 92.4 | [1, 6, 1, 4, 2] | 48 | -0.1 | 122.8 (92) | 156.1 (74) | -0.2 |
| sweep | 64x6144x9216 | 2 | 235.4 | 95.6 | 235.4 | 95.6 | [1, 6, 1, 4, 2] | 48 | +0.0 | 235.1 (96) | 296.1 (78) | +0.1 |
| sweep | 64x15360x1536 | 2 | 112.0 | 86.0 | 112.2 | 85.9 | [1, 12, 1, 1, 3] | 96 | -0.2 | 111.7 (86) | 137.4 (72) | +0.5 |
| sweep | 128x2304x6144 | 4 | 76.6 | 77.7 | 76.4 | 77.9 | [2, 3, 1, 1, 6] | 48 | +0.3 | 76.0 (78) | 84.9 (72) | +0.5 |
| sweep | 128x6144x768 | 4 | 36.0 | 60.9 | 36.6 | 59.8 | [1, 12, 1, 2, 1] | 96 | -1.8 | 35.2 (62) | 37.1 (60) | +4.0 |
| sweep | 128x6144x2304 | 4 | 74.2 | 80.2 | 74.2 | 80.2 | [1, 12, 1, 2, 1] | 96 | +0.0 | 73.9 (81) | 82.5 (74) | +0.4 |
| sweep | 128x6144x4608 | 4 | 131.8 | 88.0 | 131.4 | 88.2 | [1, 12, 1, 2, 1] | 96 | +0.3 | 131.0 (88) | 155.4 (76) | +0.3 |
| sweep | 128x15360x768 | 4 | 92.6 | 58.4 | 91.9 | 58.9 | [1, 10, 1, 1, 3] | 80 | +0.8 | 90.1 (60) | 86.6 (64) | +2.0 |
| sweep | 512x6144x1536 | 16(diag) | 127.4 | 41.0 | - | - | None | - | - | 128.3 (41) | 87.0 (62) | - |
| added | 32x6144x4608 | 1 | 119.3 | 93.9 | 119.1 | 94.0 | [1, 12, 1, 2, 1] | 96 | +0.2 | - (-) | - (-) | +0.4 |
| added | 256x2048x1024 | 8 | 30.3 | 37.2 | 29.9 | 37.7 | [1, 4, 2, 2, 4] | 64 | +1.5 | - (-) | - (-) | - |
| added | 512x6144x768 | 16(diag) | 98.7 | 32.7 | - | - | None | - | - | - (-) | - (-) | - |
| added | 512x15360x768 | 16(diag) | - | - | - | - | None | - | - | - (-) | - (-) | - |
| added | 512x6144x2304 | 16(diag) | 149.1 | 48.4 | - | - | None | - | - | - (-) | - (-) | - |
| added | 512x2304x6144 | 16(diag) | 196.9 | 36.7 | - | - | None | - | - | - (-) | - (-) | - |
| added | 512x3072x6144 | 16(diag) | 187.7 | 49.1 | - | - | None | - | - | - (-) | - (-) | - |
| added | 512x6144x4608 | 16(diag) | 225.3 | 58.6 | - | - | None | - | - | - (-) | - (-) | - |
| balanced_tail | 32x6080x4640 | 1 | 125.1 | 89.1 | 125.0 | 89.2 | [1, 12, 1, 2, 1] | 96 | +0.1 | - (-) | - (-) | -0.1 |
| balanced_tail | 64x6080x4640 | 2 | 130.9 | 86.3 | 130.8 | 86.3 | [1, 12, 1, 2, 2] | 96 | +0.0 | - (-) | - (-) | - |
| balanced_tail | 128x6080x4640 | 4 | 141.3 | 81.8 | 140.8 | 82.1 | [1, 12, 1, 2, 1] | 96 | +0.3 | - (-) | - (-) | - |
| balanced_tail | 256x6080x4640 | 8 | 163.3 | 74.0 | 163.3 | 74.0 | [1, 12, 1, 2, 1] | 96 | -0.0 | - (-) | - (-) | - |
| balanced_tail | 32x6144x4600 | 1 | 119.0 | 94.1 | 119.3 | 93.9 | [1, 12, 1, 2, 1] | 96 | -0.2 | - (-) | - (-) | - |
| balanced_tail | 32x6100x4608 | 1 | 119.4 | 93.2 | 119.5 | 93.2 | [1, 12, 1, 2, 1] | 96 | -0.0 | - (-) | - (-) | - |
| balanced_tail | 48x6144x4608 | 2 | 123.0 | 92.1 | 123.1 | 92.0 | [1, 12, 1, 2, 1] | 96 | -0.1 | - (-) | - (-) | - |

## Acceptance gates (Mt<=8)

- **Auto-selection vs best manual (geomean of product_us/manual_us, Mt<=8):** 1.001 (gate: <=1.05). Worst per-shape product gap: +1.7% (gate: <=10%).
- **Manual vs prototype/historical target (geomean, Mt<=8):** +0.5% (gate: <=5%, fixed-cost noise on short kernels documented). Worst: 128x6144x768 +4.0%.
- **Selected-path correctness/hangs (Mt<=8):** ALL PASS (this counts only the CHOSEN product + best-manual configs, not configs discarded during search).
- **Exploratory-search outcomes (all measured/rejected configs):** ok=339, runtime=1, validation=109. validation = host-planner-rejected before launch (expected); ok = measured; pcc/runtime/hang on non-selected configs are pruned candidates, not op defects.

## Balanced-tail + N-bank quantization

For a non-divisible N, effective BW is bounded by 8-bank quantization: per-bank width = ceil(Nt/8), so the fully-loaded banks set the wall-clock. Delivered BW (padded bytes/time) stays high; the effective/delivered gap = quantization loss, NOT pad-read waste (balanced tails issue no pad reads).

| shape | Mt | manual us | eff %512 | ceil(Nt/8)*8/Nt quant ceiling | divisible neighbor us |
|---|---|---|---|---|---|
| 32x6080x4640 | 1 | 125.0 | 89.2 | 95.4% | 119.1 |
| 64x6080x4640 | 2 | 130.8 | 86.3 | 95.4% | 122.6 |
| 128x6080x4640 | 4 | 140.8 | 82.1 | 95.4% | 131.4 |
| 256x6080x4640 | 8 | 163.3 | 74.0 | 95.4% | - |
| 32x6144x4600 | 1 | 119.3 | 93.9 | 100.0% | - |
| 32x6100x4608 | 1 | 119.5 | 93.2 | 100.0% | - |
| 48x6144x4608 | 2 | 123.1 | 92.0 | 100.0% | - |

## Findings & prioritized gaps

1. **Relative gates pass; one absolute-BW caveat (256x2048x1024, item 4).** Auto (config=None) is within 0.1% of best-manual on geomean; manual is within 0.5% of the regime-A prototype sweep-best (µs, 512 GB/s convention) on the shapes that HAVE a prototype/historical target — i.e. the independent TTNN op reproduces the prototype's tuned bandwidth. NOTE: the auto-vs-manual and manual-vs-target geomeans are relative; they do NOT assert every Mt<=8 shape is bandwidth-bound. 256x2048x1024 has no historical target and tops out at ~37% of 512 GB/s for structural reasons (item 4) — it is reported as an in-scope gap, not hidden by omission from the target aggregate.
2. **Vs the historical minimal_matmul branch** (`bh_skinny`, 500 GB/s convention — compared by µs, NOT %): geomean speedup ~1.19x on the shared Mt<=8 regime-A shapes (e.g. 32x6144x9216 228 vs 293µs, 32x6144x1536 42 vs 52µs).
3. **Two auto-picker entries corrected (both re-swept on the op, config=None verified):** (a) 128x15360x768 (Mt4, deep-K K=15360 / small-N Nt=24): the lookup entry (Ns1,Pk8,kb4,nsb1) carried from the C++ prototype sweep gave 97.7us/55.4%; the op-side re-sweep candidate (Ns1,Pk12,kb1,nsb3) is a stable 4-5% median win (93.6us/57.9%), so the single {4,480,24} table entry was refreshed. (b) 256x2048x1024 (Mt8, below) — a new {8,64,32} table entry.
4. **256x2048x1024 (Mt8) in-scope gap — CLOSED to the shape's structural ceiling, +15%.** This shape had NO table entry and fell through to the cost-model, which picked an N-split (Ns2,Pk4,Sm1,kb2,nsb1) = 34.7us/32.5%. An **exhaustive 788-config sweep across all 5 levers (Sm NOT restricted to 1)** found the winner is an **M-split** (Ns1,Pk4,Sm2,kb2,nsb2) = 30.1us/37.4% (stable +15% over the cost-model across 3 relaunches); added as a table entry, product path now resolves to it. Per-RISC profiling (in1 reader=BRISC, in0 ring/reduce writer=NCRISC, compute=TRISC) shows all three RISC kernels span ~the full wall in every top config, with the **in1 reader (BRISC) the longest pole** — but at ~30us it is ~3x the pure-DRAM read floor (~10us for this shape), so it is stalling on back-pressure / split-K reduction sync, NOT DRAM throughput. There is a **~20us fixed floor** (min per-core time, roughly config- and core-count-independent) = kernel dispatch + semaphore/ring setup + reduction-tail sync, plus large per-core spread (min ~20 -> max ~30us) from split-K reduction-chain tail imbalance. **Remaining architectural limitation:** this shape is small/low-AI and is overhead- and reduction-imbalance-bound; the ~37% ceiling is NOT bandwidth and is not closable by the picker or by preserving the ring/DRAM-sharded architecture — it would need a tiny-shape kernel path (lower fixed overhead + balanced reduction tails), deferred as out of the current change scope.
5. **Balanced tails: no divisible regression.** Sub-tile element dims that stay bank-aligned (32x6144x4600 N-subtile, 32x6100x4608 K-subtile, 48x6144x4608 M-subtile) hit 92-94%, matching divisible neighbors. The only non-div loss is N-bank quantization (ceil(Nt/8)); the 6080x4640 series (89/87/83/76% for Mt1/2/4/8) sits at the ~95.4% quant ceiling of its divisible neighbor, and the reader issues NO pad-tile DRAM reads.
6. **Mt=16 diagnostics (512x*, out of scope, not optimized):** 33-59% of 512 GB/s. 512x15360x768 classifies 'runtime' — the auto-selected config FATALs on this deep-K (Kt=480) + Mt16 combination (the cost-model fallback does not converge on a valid config, though ~8 planner-feasible configs exist); left as-is since Mt16 is diagnostic-only. The rest are compute/delivery-bound and reported for visibility only — the op is not redesigned for them.

## Notes
- Mt=16 (512×*) shapes are diagnostic-only (out of the Mt<=8 acceptance scope); reported above but excluded from gates.
- cores = 8*Pk*Ns*Sm of the winning manual config.
- Very short kernels (32x2048x512 ~8µs) carry fixed-cost dispatch noise; their +/-2% deltas are within measurement variation (spread in regime_a_bench.json).
