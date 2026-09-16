# Quasar pool `num_threads=4` (PR #55001): ZeBu emulator before/after

Date: 2026-09-16. Emulator: `emu-quasar-2x3` (2 functional clusters, logical grid 2x1), slow dispatch,
device profiler (`TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1`). Metric: kernel cycles of the
**pool2d program** (first `*-KERNEL` zone start to last `*-KERNEL` zone end over all cores and RISCs), 1 warmup
+ 5 measured calls per case, median. Iteration spread is under 1% (worst 2%). The Quasar L1-only profiler keeps
only the last launched program per RISC, so the halo and reallocate programs of a ttnn pool call are not in the
numbers (verified: identical cycles with `reallocate_halo_output` on/off); the PR did not touch halo.

## Legs

| leg | tree | commit | what it is |
|---|---|---|---|
| before | /localdev/wransom/tt-metal-emu-before | 1ffe1303ac4 | parent of the merge: two split readers on two DM cores + one NEO |
| after | /localdev/wransom/tt-metal-emu | 19f3e3cb644 | the merged PR: one 4-thread reader + 4 NEOs |
| after_t1 | /localdev/wransom/tt-metal-emu-after-t1 | 824f7f31dfe | after with `kQuasarNeosPerCluster = 1` (the PR table's "threads=1" methodology) |
| after_nofill | /localdev/wransom/tt-metal-emu-after-nofill | b5c0aeee0bc | DIAGNOSTIC: after with the whole-ring identity fill removed (attribution only; k7x7 mismatches as expected) |

Cases are PR #55001's perf matrix re-hosted on the 2-cluster grid: single-core cases are identical to the
sim runs; cases the sim ran on 4 cores run on 2 (64 sticks/core), with `*_2c32` twins restoring the sim's
32 sticks/core; `block_2x2` becomes a 1x2 block stand-in; `stem_112x112_c64_2c` is the resnet50 stem maxpool
geometry (6272 sticks/core), which the sim could not run.

## Results (pool2d kernel cycles, median of 5)

| case | before | after | before->after | after_t1 | T1->T4 emu | T1->T4 sim | after_nofill | cores (emu) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| k3x3_s1 | 547,269 | 449,573 | 1.22x | 1,076,175 | 2.39x | 2.29x | 309,481 | 2 x 32 sticks |
| k5x5_s2 | 495,711 | 415,643 | 1.19x | 986,661 | 2.37x | 1.37x* | 277,543 | 2 x 64 sticks |
| k7x7_s2_large | 1,746,947 | 1,070,827 | 1.63x | 3,455,355 | 3.23x | 2.95x | 934,457 | 1 x 128 sticks |
| k8x8_s2_large | 2,230,205 | 1,337,765 | 1.67x | 4,435,478 | 3.32x | 3.11x | 1,199,603 | 1 x 128 sticks |
| k9x9_s2_3chunks | 2,864,859 | 1,653,188 | 1.73x | 5,620,603 | 3.40x | 3.26x | 1,508,971 | 1 x 128 sticks |
| batch2 | 283,215 | 304,177 | 0.93x | 560,685 | 1.84x | 1.18x* | 166,695 | 2 x 64 sticks |
| tall_32x4 | 283,769 | 305,851 | 0.93x | 561,137 | 1.83x | 1.18x* | 166,403 | 2 x 64 sticks |
| wide_4x32 | 282,923 | 305,169 | 0.93x | 561,037 | 1.84x | 1.18x* | 166,056 | 2 x 64 sticks |
| wide_c280_3blocks | 641,748 | 541,075 | 1.19x | 1,184,079 | 2.19x | 2.06x | 337,511 | 1 x 32 sticks |
| avg_k3x3_s1 | 1,409,699 | 919,814 | 1.53x | 2,759,273 | 3.00x | 3.05x | 780,007 | 1 x 128 sticks |
| avg_k7x7_s1_large | 1,096,361 | 597,207 | 1.84x | 2,153,309 | 3.61x | 3.65x | 594,515 | 1 x 128 sticks |
| width_1x2_c128 | 150,137 | 232,114 | 0.65x | 302,733 | 1.30x | 1.18x | 94,089 | 1x2 width |
| block_1x2_c128 (for block_2x2) | 549,857 | 449,006 | 1.22x | 1,077,001 | 2.40x | 1.64x* | 311,595 | 1x2 block |
| k5x5_s2_2c32 | 256,391 | 287,101 | 0.89x | 515,581 | 1.80x | 1.37x* | 149,807 | 2 x 32 sticks |
| batch2_4x8_2c32 | 150,313 | 231,825 | 0.65x | 302,853 | 1.31x | 1.18x | 93,633 | 2 x 32 sticks |
| tall_16x4_2c32 | 150,357 | 232,390 | 0.65x | 302,929 | 1.30x | 1.18x | 93,949 | 2 x 32 sticks |
| wide_4x16_2c32 | 150,207 | 231,856 | 0.65x | 302,627 | 1.31x | 1.18x | 94,177 | 2 x 32 sticks |
| stem_112x112_c64_2c | 26,077,887 | 14,286,724 | 1.83x | 50,659,841 | 3.55x | - | 14,235,085 | 2 x 6272 sticks |

`*` sim ran these on 4 cores (or a 2x2 grid); the emulator has 2 clusters.

Geomeans over the 12 PR-table cases: before->after **1.23x** (sim claimed 2.00x); after_t1->after **2.42x**
(sim 2.00x); after->after_nofill 1.45x. Over all 18 cases: 1.11x, 2.19x, 1.58x.

## Findings

1. **The sim's threads=1 -> threads=4 deltas are realistic for what they compare.** On the seven single-core
   cases the emulator T1->T4 speedups are within about 10% of the sim's (k7x7 3.23x vs 2.95x, k9x9 3.40x vs
   3.26x, avg_k7x7 3.61x vs 3.65x, k3x3_s1 2.39x vs 2.29x, wide_c280 2.19x vs 2.06x). The larger emulator
   numbers on the multi-core cases come from the 2-vs-4 core difference; the `*_2c32` twins (same 32
   sticks/core as the sim) land at 1.30x vs the sim's 1.18x.

2. **The true commit delta (shipped code before vs after) is much smaller: 1.23x geomean on the PR cases**,
   1.6-1.8x on large kernels and the resnet stem, and a 0.65x regression at 32 sticks/core (0.93x at 64).
   The PR's baseline was the new code forced to one lane, which is itself 2x slower than the old
   two-split-reader kernel (k3x3_s1: 547k old vs 1,076k new@T=1).

3. **Two mechanisms explain the gap** (fixed + per-stick model from the 32/64-stick twins):

   | leg | fixed cycles per program | cycles per stick per core |
   |---|---:|---:|
   | before (2 DM readers, 1 NEO) | 17k | 4,153 |
   | after (4 DM lanes, 4 NEOs) | 159k | 2,283 |
   | after_t1 (1 DM lane, 1 NEO) | 45k | 8,067 |
   | after_nofill (diagnostic) | 21k | 2,264 |

   - The new tree adds ~140k cycles of fixed cost per program: every one of the 4 DM threads CPU-fills the
     whole input ring (16-bit `volatile` stores, `entry_size * 2 * 4` bytes each, then an L2 flush) at
     `reader_pool_2d.cpp` lines 315-325. Removing that fill (diagnostic leg) brings the fixed cost back to
     21k and makes the new tree beat the old one everywhere (94k vs 150k at 32 sticks/core). The old tree
     cleared 2 pages with one NoC zero-write. Break-even of after vs before is ~76 sticks/core.
   - The per-stick cost scales with the number of DM readers, not NEOs: 8.1k (1 DM) -> 4.15k (2 DM) ->
     2.28k (4 DM). Going from 2 readers + 1 NEO to 4 readers + 4 NEOs buys 1.82x, i.e. the kernel is
     reader/data-movement bound on RTL; the idle NEOs the PR set out to use were not the bottleneck.

4. Lanes are balanced at T=4 (the four DM spans and four NEO spans agree within 1%); all 18 cases pass the
   torch golden on before, after and after_t1 (the old tree's zero-fill max bug needs negative inputs to
   show; these inputs are in [0,1)).

## Recommendations

- Replace the redundant whole-ring CPU fill: each lane fills only its own strided entries (1/4 of the stores,
  no overlapping writes), or write the identity tile with the NoC as the old tree did. Expected effect from
  the diagnostic leg: 1.45x geomean on top of the merged code and no small-shape regression.
- Treat DM reader count per cluster (6 usable DM cores) as the perf lever for pool, not NEO count.
- craq-sim under-models CPU L1 store cost and DM-bound streaming; re-check fixed-cost-sensitive changes on
  the emulator (a full 18-case leg is 25-60 min).

## Files

- Harness: `/localdev/wransom/qpool_emu_perf/{qpool_emu_perf.py,run_leg.sh,qpool_emu_report.py}`
- Per-leg: `/localdev/wransom/qpool_emu_perf/results/<leg>/{results.json,run.log,profile_log_device.csv}`
- Full table incl. iteration spread and per-RISC lane balance: `results/report_final.txt`
- Branches pushed: `wransom/qpool_emu_perf_after_t1`, `wransom/qpool_emu_perf_after_nofill`
  (local only: `wransom/qpool_emu_perf_after` = 19f3e3cb644, `wransom/qpool_emu_perf_before` = 1ffe1303ac4)
