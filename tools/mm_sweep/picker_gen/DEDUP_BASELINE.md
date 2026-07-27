# Cross-Ns in0-dedup golden baselines (production path)

Tuples are `(Ns,Pk,Sm,kb,nsb)`. Current production op, UNFUSED, resident BF16 inputs, 1 warmup + 8 timed iters, >=5 fresh-process/device relaunches. Peak DRAM reference = 512 GB/s. No kernel/picker change.

## Environment

- commit `ce79cca7f79`; version `v0.73.0-dev20260605-184-gce79cca7f79`; build **Release** (Tracy on)
- device: **Blackhole p150b**, PCI a1, 1.35 GHz; firmware bundle **19.5.0**; KMD **2.4.1**
- per-RISC: BRISC/NCRISC = data-movement kernels, TRISC = compute. Production kernels expose only whole-RISC `-KERNEL` zones, so the requested fine phases (in0 read / in0 ring / in1 read / compute / reduction / output) are **not separable** without adding kernel zones (out of scope). Per-RISC spans + per-core spread are reported instead.

## Summary

| shape | config (Ns,Pk,Sm,kb,nsb) | median us | relaunch medians us | spread% / IQRus | PCC (min) | eff / del GB/s | %512 | wall/ideal · excess us | per-RISC us (B/N/T) · core-spread% | redundant bytes (%) | dedup DRAM-ideal us | ceiling x (not achievable) | hist us (Δ%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256x2048x2048 | (2, 2, 3, 4, 4) | 36.79 | 36.4, 37.1, 36.9, 36.8, 36.6 | 2.1 / 0.86 | 0.99999 | 285 / 314 | 61.2 | 1.63 · 14.3 | 36.79/35.72/35.01 · 33.2 | 1,048,576 (9.1%) | 20.5 | 1.100 | 36.86 (-0.2) |
| 256x2048x6144 | (3, 2, 2, 2, 4) | 84.40 | 84.6, 84.8, 84.6, 83.9, 85.0 | 1.3 / 1.75 | 0.99999 | 348 / 373 | 72.8 | 1.37 · 23.0 | 83.69/84.09/83.52 · 37.9 | 2,097,152 (6.7%) | 57.3 | 1.071 | 84.18 (+0.3) |
| 512x6144x2304 | (2, 6, 1, 2, 1) | 170.69 | 169.9, 170.5, 170.4, 171.2, 170.8 | 0.8 / 2.35 | 1.00001 | 216 / 253 | 49.5 | 2.02 · 86.2 | 170.49/165.69/169.62 · 20.1 | 6,291,456 (14.6%) | 72.2 | 1.170 | 197.67 (-13.7) |
| 512x6144x4608 | (2, 6, 1, 4, 1) | 223.65 | 223.0, 224.9, 223.8, 223.6, 223.6 | 0.9 / 2.62 | 1.00004 | 302 / 330 | 64.6 | 1.55 · 79.3 | 222.69/219.8/222.74 · 15.8 | 6,291,456 (8.5%) | 132.1 | 1.093 | 267.17 (-16.3) |

## Traffic model (bytes = 2·elements, bf16)

logical = 2(MK+KN+MN); delivered = 2(Ns·MK+KN+MN); redundant in0 = 2(Ns−1)MK; dedup = delivered − redundant. `pad_delta%` = physical (planner shard-padded) delivered vs logical delivered.

| shape | logical B | delivered B | redundant B (%) | dedup B | pad Δ% | DRAM-only dedup ideal us | delivered/dedup ceiling |
|---|---|---|---|---|---|---|---|
| 256x2048x2048 | 10,485,760 | 11,534,336 | 1,048,576 (9.1%) | 10,485,760 | +0.0% | 20.5 | 1.100x |
| 256x2048x6144 | 29,360,128 | 31,457,280 | 2,097,152 (6.7%) | 29,360,128 | +0.0% | 57.3 | 1.071x |
| 512x6144x2304 | 36,962,304 | 43,253,760 | 6,291,456 (14.6%) | 36,962,304 | +0.0% | 72.2 | 1.170x |
| 512x6144x4608 | 67,633,152 | 73,924,608 | 6,291,456 (8.5%) | 67,633,152 | +0.0% | 132.1 | 1.093x |

## Notes

- **%512** = delivered traffic GB/s ÷ 512. **eff GB/s** = logical(useful) bytes ÷ wall; **del GB/s** = delivered(actual, Ns-redundant) bytes ÷ wall.
- **DRAM-only dedup ideal** = dedup_bytes ÷ 512 GB/s: a perfect-overlap DRAM floor, **not** an achievable speedup (compute/reduction/forward costs are excluded). **ceiling x** = delivered/dedup is the pure-DRAM upper bound only.
- **Historical Δ**: sanity check vs supplied numbers (36.86/84.18/197.67/267.17 us); configs were not tuned to reproduce them. Discrepancies >5% are discussed below.
- Cached-program replay verified every relaunch (`replay_matches_all`, cached PCC ≥ golden PCC).

## Historical-discrepancy investigation (>5%)

Within noise (<5%): 256x2048x2048 (-0.2%), 256x2048x6144 (+0.3%). These match the supplied numbers to within a fraction of a percent, confirming the measurement methodology (kernel-wall = max over cores, 8 timed iters) is consistent with how the historical figures were taken.

Exceeding 5% — all are **faster** now, not regressions:

- **512x6144x2304** (2, 6, 1, 2, 1): 170.7 us vs historical 197.67 us (-13.7%). Stable across 5 relaunches (spread 0.8%, IQR 2.35 us) with valid PCC (1.00001), so this is a real device-time difference, not measurement noise or throttling.
- **512x6144x4608** (2, 6, 1, 4, 1): 223.6 us vs historical 267.17 us (-16.3%). Stable across 5 relaunches (spread 0.9%, IQR 2.62 us) with valid PCC (1.00004), so this is a real device-time difference, not measurement noise or throttling.

**Explanation.** The two shapes that differ are the large-Mt (M=512 => Mt=16), Sm=1, deep-split-K (Pk=6) cases; the two that match are the smaller Mt=8 cases. The current commit contains the full optimized production chain landed after the historical figures were recorded (PARETO physical ring order, progressive in0 waits, pipelined drain, coalesced contiguous in1 reads, forward-signal-first in1 delivery). Those optimizations target exactly the in1-read / in0-ring / reduction costs that dominate large deep-K shapes, so they speed up shapes 3-4 (~14-16%) while the smaller shapes 1-2 were already near their floor and are unchanged. We did NOT tune the configs to reproduce the historical numbers; the current values are the trustworthy current-path golden. (Exact historical commit not available for a line-by-line diff; the pattern - improvement concentrated on large deep-K - is consistent with those specific optimizations.)
