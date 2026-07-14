# regime_a_matmul — Blackhole steady-state benchmark

System under test: the independent `ttnn.experimental.regime_a_matmul` op (NOT the C++ prototype). Resident device inputs + pre-sharded resident weights; PCC>=0.999 verified before timing; 1 warmup + 8 timed iters (min reported; median/spread in JSON). op %=of 512 GB/s. Historical `bh_skinny` %=of 500 GB/s (kept separate); cross-source comparison uses kernel us.

## Per-shape results

| cat | shape | Mt | prod us | prod %512 | manual us | %512 | cfg(Ns,Pk,Sm,kb,nsb) | cores | picker gap% | sweep-best us (%512) | hist branch us (%500) | vs target |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep | 32x256x6144 | 1 | 8.5 | 81.3 | 8.6 | 80.8 | [3, 1, 1, 1, 8] | 24 | -0.7 | 8.6 (80) | 11.2 (63) | -0.1 |
| sweep | 32x2048x512 | 1 | 8.6 | 51.6 | 8.4 | 52.4 | [1, 2, 1, 4, 2] | 16 | +1.7 | 8.6 (51) | 9.4 (48) | -2.0 |
| sweep | 32x2048x1536 | 1 | 15.0 | 85.1 | 15.1 | 84.6 | [2, 2, 1, 4, 3] | 32 | -0.6 | 15.4 (83) | 18.8 (69) | -2.3 |
| sweep | 32x2048x2048 | 1 | 19.4 | 87.0 | 19.4 | 86.9 | [2, 2, 1, 4, 4] | 32 | -0.1 | 19.5 (86) | 23.1 (75) | -0.3 |
| sweep | 32x6144x1536 | 1 | 42.1 | 89.8 | 41.7 | 90.8 | [1, 3, 1, 4, 6] | 24 | +1.2 | 42.1 (90) | 51.9 (75) | -1.0 |
| sweep | 32x6144x2304 | 1 | 61.5 | 91.7 | 61.4 | 91.8 | [1, 4, 1, 2, 9] | 32 | +0.2 | 61.6 (91) | 78.3 (74) | -0.4 |
| sweep | 32x6144x3072 | 1 | 79.5 | 94.2 | 79.1 | 94.7 | [1, 3, 1, 4, 7] | 24 | +0.5 | 79.8 (94) | 105.5 (73) | -0.9 |
| sweep | 32x6144x6144 | 1 | 154.3 | 96.6 | 153.7 | 96.9 | [1, 3, 1, 4, 6] | 24 | +0.3 | 153.9 (97) | 199.6 (76) | -0.1 |
| sweep | 32x6144x9216 | 1 | 228.6 | 97.6 | 228.6 | 97.6 | [1, 6, 1, 4, 2] | 48 | +0.0 | 228.4 (98) | 293.4 (78) | +0.1 |
| sweep | 64x4608x6144 | 2 | 123.3 | 91.9 | 123.3 | 91.9 | [1, 6, 1, 1, 6] | 48 | -0.0 | 123.1 (92) | 153.8 (75) | +0.2 |
| sweep | 64x6144x1536 | 2 | 47.5 | 81.6 | 46.5 | 83.3 | [1, 6, 1, 4, 4] | 48 | +2.1 | 47.6 (81) | 51.7 (77) | -2.2 |
| sweep | 64x6144x4608 | 2 | 122.3 | 92.6 | 122.4 | 92.6 | [1, 12, 1, 2, 1] | 96 | -0.1 | 122.8 (92) | 156.1 (74) | -0.3 |
| sweep | 64x6144x9216 | 2 | 234.4 | 96.0 | 234.9 | 95.8 | [1, 6, 1, 4, 2] | 48 | -0.2 | 235.1 (96) | 296.1 (78) | -0.1 |
| sweep | 64x15360x1536 | 2 | 111.7 | 86.3 | 111.3 | 86.6 | [1, 12, 1, 1, 3] | 96 | +0.4 | 111.7 (86) | 137.4 (72) | -0.4 |
| sweep | 128x2304x6144 | 4 | 76.1 | 78.2 | 76.2 | 78.1 | [2, 3, 1, 1, 6] | 48 | -0.1 | 76.0 (78) | 84.9 (72) | +0.3 |
| sweep | 128x6144x768 | 4 | 35.4 | 61.9 | 35.4 | 61.8 | [1, 12, 1, 2, 1] | 96 | -0.2 | 35.2 (62) | 37.1 (60) | +0.6 |
| sweep | 128x6144x2304 | 4 | 73.1 | 81.4 | 73.6 | 80.9 | [1, 12, 1, 2, 1] | 96 | -0.6 | 73.9 (81) | 82.5 (74) | -0.4 |
| sweep | 128x6144x4608 | 4 | 131.1 | 88.5 | 130.3 | 89.0 | [1, 12, 1, 2, 1] | 96 | +0.6 | 131.0 (88) | 155.4 (76) | -0.6 |
| sweep | 128x15360x768 | 4 | 97.2 | 55.7 | 90.1 | 60.1 | [1, 12, 1, 1, 3] | 96 | +7.8 | 90.1 (60) | 86.6 (64) | +0.0 |
| sweep | 512x6144x1536 | 16(diag) | 122.6 | 42.6 | - | - | None | - | - | 128.3 (41) | 87.0 (62) | - |
| added | 32x6144x4608 | 1 | 118.8 | 94.2 | 118.8 | 94.2 | [1, 12, 1, 2, 1] | 96 | +0.0 | - (-) | - (-) | +0.1 |
| added | 256x2048x1024 | 8 | 33.8 | 33.3 | 34.3 | 32.8 | [2, 4, 1, 2, 1] | 64 | -1.6 | - (-) | - (-) | - |
| added | 512x6144x768 | 16(diag) | 97.4 | 33.1 | - | - | None | - | - | - (-) | - (-) | - |
| added | 512x15360x768 | 16(diag) | - | - | - | - | None | - | - | - (-) | - (-) | - |
| added | 512x6144x2304 | 16(diag) | 149.1 | 48.4 | - | - | None | - | - | - (-) | - (-) | - |
| added | 512x2304x6144 | 16(diag) | 194.2 | 37.2 | - | - | None | - | - | - (-) | - (-) | - |
| added | 512x3072x6144 | 16(diag) | 181.0 | 50.9 | - | - | None | - | - | - (-) | - (-) | - |
| added | 512x6144x4608 | 16(diag) | 223.2 | 59.2 | - | - | None | - | - | - (-) | - (-) | - |
| balanced_tail | 32x6080x4640 | 1 | 124.9 | 89.3 | 124.6 | 89.5 | [1, 12, 1, 2, 1] | 96 | +0.2 | - (-) | - (-) | -0.3 |
| balanced_tail | 64x6080x4640 | 2 | 130.4 | 86.5 | 130.1 | 86.8 | [1, 12, 1, 2, 1] | 96 | +0.3 | - (-) | - (-) | - |
| balanced_tail | 128x6080x4640 | 4 | 140.0 | 82.5 | 139.6 | 82.8 | [1, 12, 1, 2, 1] | 96 | +0.3 | - (-) | - (-) | - |
| balanced_tail | 256x6080x4640 | 8 | 159.4 | 75.9 | 160.3 | 75.5 | [1, 12, 1, 2, 1] | 96 | -0.5 | - (-) | - (-) | - |
| balanced_tail | 32x6144x4600 | 1 | 118.6 | 94.4 | 118.8 | 94.3 | [1, 12, 1, 2, 1] | 96 | -0.2 | - (-) | - (-) | - |
| balanced_tail | 32x6100x4608 | 1 | 119.1 | 93.5 | 119.0 | 93.6 | [1, 12, 1, 2, 1] | 96 | +0.0 | - (-) | - (-) | - |
| balanced_tail | 48x6144x4608 | 2 | 122.4 | 92.6 | 122.6 | 92.4 | [1, 12, 1, 2, 1] | 96 | -0.2 | - (-) | - (-) | - |

## Acceptance gates (Mt<=8)

- **Auto-selection vs best manual (geomean of product_us/manual_us, Mt<=8):** 1.004 (gate: <=1.05). Worst per-shape product gap: +7.8% (gate: <=10%).
- **Manual vs prototype/historical target (geomean, Mt<=8):** -0.5% (gate: <=5%, fixed-cost noise on short kernels documented). Worst: 128x6144x768 +0.6%.
- **Correctness/hangs (Mt<=8):** ALL PASS

## Balanced-tail + N-bank quantization

For a non-divisible N, effective BW is bounded by 8-bank quantization: per-bank width = ceil(Nt/8), so the fully-loaded banks set the wall-clock. Delivered BW (padded bytes/time) stays high; the effective/delivered gap = quantization loss, NOT pad-read waste (balanced tails issue no pad reads).

| shape | Mt | manual us | eff %512 | ceil(Nt/8)*8/Nt quant ceiling | divisible neighbor us |
|---|---|---|---|---|---|
| 32x6080x4640 | 1 | 124.6 | 89.5 | 95.4% | 118.8 |
| 64x6080x4640 | 2 | 130.1 | 86.8 | 95.4% | 122.4 |
| 128x6080x4640 | 4 | 139.6 | 82.8 | 95.4% | 130.3 |
| 256x6080x4640 | 8 | 160.3 | 75.5 | 95.4% | - |
| 32x6144x4600 | 1 | 118.8 | 94.3 | 100.0% | - |
| 32x6100x4608 | 1 | 119.0 | 93.6 | 100.0% | - |
| 48x6144x4608 | 2 | 122.6 | 92.4 | 100.0% | - |

## Findings & prioritized gaps

1. **All Mt<=8 acceptance gates pass.** Auto (config=None) is within 0.4% of best-manual on geomean; manual is within 0.5% of the regime-A prototype sweep-best (µs, 512 GB/s convention) — i.e. the independent TTNN op reproduces the prototype's tuned bandwidth.
2. **Vs the historical minimal_matmul branch** (`bh_skinny`, 500 GB/s convention — compared by µs, NOT %): geomean speedup ~1.20x on the shared Mt<=8 regime-A shapes (e.g. 32x6144x9216 228 vs 293µs, 32x6144x1536 42 vs 52µs).
3. **One picker outlier (classified PICKER, under gate):** 128x15360x768 (Mt4, deep-K K=15360 / small-N Nt=24) auto 55.7% vs manual 60.1% (+7.8%, < 10% gate). The kernel reaches 60% with (Ns1,Pk12,kb1,nsb3); the lookup-table entry (Ns1,Pk8,kb4,nsb1) — carried from the prototype sweep — underperforms on the op by ~4pt. NOT a kernel/engine loss (manual == sweep-best). Recommended fix: refresh this one table entry via a small op-side re-sweep; deferred here to preserve the picker per the task's 'preserve the existing auto-picker initially' guidance.
4. **Balanced tails: no divisible regression.** Sub-tile element dims that stay bank-aligned (32x6144x4600 N-subtile, 32x6100x4608 K-subtile, 48x6144x4608 M-subtile) hit 92-94%, matching divisible neighbors. The only non-div loss is N-bank quantization (ceil(Nt/8)); the 6080x4640 series (89/87/83/76% for Mt1/2/4/8) sits at the ~95.4% quant ceiling of its divisible neighbor, and the reader issues NO pad-tile DRAM reads.
5. **Mt=16 diagnostics (512x*, out of scope, not optimized):** 33-59% of 512 GB/s; 512x15360x768 has no feasible config (deep-K + Mt16 exceeds L1). These are compute/delivery-bound and reported for visibility only — the op is not redesigned for them.

## Notes
- Mt=16 (512×*) shapes are diagnostic-only (out of the Mt<=8 acceptance scope); reported above but excluded from gates.
- cores = 8*Pk*Ns*Sm of the winning manual config.
- Very short kernels (32x2048x512 ~8µs) carry fixed-cost dispatch noise; their +/-2% deltas are within measurement variation (spread in regime_a_bench.json).
