# flat_routed_expert: bf16 x / bf16 h — perf work log (2026-10-09)

Goal: find why bf16 x + h costs +57% at large M and speed it up (budget: 12 ideas), without regressing any of the four
x / h regimes (bfp8/bfp8, bfp8 x + bf16 h, bf16 x + bfp8 h, bf16/bf16). Perf focus: bf16 both. All numbers: one chip,
GLM shape (H 4096, I 2048, 36 bfp4 experts, capacity 8192), balanced routing, model call (indexed x, row-major y, fp32
down), LoFi, us per expert (test_flat_balanced_sweep.py, RT-profiler device time).

Baseline (before this log): bfp8/bfp8 M 128 / 256 / 512 / 2048: 40.9 / 50.6 / 73.5 / 275.4; bf16 x: 41.6 / 56.1 / 109.9 /
433.6; bf16 both: 41.3 / 61.3 / 115.9 / 441.2; bfp8 x + bf16 h: did not fit L1 (allocator failure).

## Diagnosis
Profile at M 2048 (MIMO_FL_ZONES, WAITZ): with bf16 x the plan ran gate/up on **32 cores (NP 2)** instead of 64: the x
ring sizes are tried 24 -> 16 -> 12 and the first that fits wins; at 24 slots only NP 2 fits the gate/up L1 budget with
2 KB x tiles. Relays: tilize waits on super-block space (TZ_OUT 6.6k vs 1.6k cycles), reader waits on a full RM CB
(XRD_FULL 4.8k vs 1.1k): back-pressure from the halved gate/up throughput.

## Ideas
1. Plan: with bf16 x pick the split with the most gate/up cores over all ring sizes (NP 1 at 16 slots).
   Result: bf16 x 433.6 -> 374.5 at M 2048 (512: 109.9 -> 101.8); bf16 both 441 -> 482 (NP 1 moved the limit to the
   h path; see 4). Kept (bf16 x only; bfp8 x plans unchanged).
   Infra (not an idea): x_bf16 / h_bf16 became config / plan parameters (program key); the arena budget now subtracts
   the static CBs, done words and relay scratch words (bf16 h no longer overflows L1: bfp8 x + bf16 h runs for the
   first time); the gate/up h_local CB moved into the arena (frees 24 KB of the program-wide static CB region).
2. bf16 h: 1.5-expert down weight ring (unpinned down schedule) so the row-major y out CB is not squeezed to ~1 row
   tile (down packer waited 1.7k cycles per row tile on y space). Result: no change (483). The out CB stall was a
   symptom. Kept (harmless, frees 92 KB on the down cores, used by 4).
3. 13 (and 26) down chains instead of 7: 469 (-3%) / 839. Dropped.
4. 64-row sub-blocks (mt 2) for bf16 h: h buffers 256 KB, 3 of them fit; gate/up one row pass per sub-block.
   bf16 both 483 -> 388 at M 2048, 129 -> 104 at 512, 71 -> 57 at 256.
   Diagnostic: x resident (no x movement) with mt 2 + bf16 h: bf16 x == bfp8 x (376 us) -> x delivery / gate/up bf16
   unpack are not the limit; the down side is: unpack-bound down matmul (1 bf16 h tile + 5 bfp4 weight tiles per K
   for 5 products: +33% per row) and the down cores still waiting for h (~40% of the sub-block).
5. h pieces / chain link depth (16 or 64 pieces, 6 in flight): 390.7 / 400.1 / 419.0 vs 388. Dropped.
6. rdown (readers compute 6, or 3, down columns as chain tails): 493.4 / 493.8. Dropped (the rdown path costs more
   than it moves off the down cores at this shape).
   Reference: 64-row sub-blocks with bfp8 h are no faster than with bf16 h (402 vs 388; bfp8 both 411 vs 275 at 128
   rows): at mt 2 a per-sub-block fixed cost dominates, not the h bytes.
7. 96-row sub-blocks (mt 3, 2 h buffers): 493. Dropped (two h buffers again).
8. 4 h buffers for bf16 h (GATH3 = semaphore 3, unused on down cores): 388 -> 384. Kept (with 4).
9. Gate/up fp32 full-sync DST for bf16 x with 128-row sub-blocks (one pass, x freed block by block; with row passes
   a 16-slot bf16 x ring holds exactly one sub-block and the relays cannot prefetch): bf16 x + bfp8 h M 160 / 256 /
   512 / 2048: 48.9 / 54.8 / 101.6 / 373.1 -> 44.2 / 49.7 / 91.4 / 366.5 (fixes the M 160 regression of 1). Kept
   (plan: gu_full_sync = x_bf16 && the sub-block needs row passes). With 64-row sub-blocks: worse (401). Not used there.
10. bf16 h (64-row sub-blocks): x ring 16 -> 32 slots: bf16 both 384 -> 374 at 2048, 102.9 -> 100.2 at 512. Kept.
11. Coordinator GO as 2 multicast semaphore sets instead of 64 increments: 374 -> 371 (bf16 both), bfp8 unchanged.
    Reverted (1% for a change in a synchronization path every regime uses).
12. 96-row sub-blocks with 3 h buffers (1-expert down ring) + full-sync gate/up: 399.5 at 2048, worse at small M.
    Dropped.
Probe switches left in (env, inert by default): MIMO_FL_MT, MIMO_FL_DRING, MIMO_FL_DCH, MIMO_FL_HPIECES,
MIMO_FL_LINK_DEPTH, MIMO_FL_RDOWN, MIMO_FL_PCD_R, MIMO_FL_GU_FULL_SYNC, MIMO_FL_WAITZ (+ GU_ACQ zone).

## Final
Perf (us per expert, M 32 / 128 / 160 / 256 / 512 / 2048 / 5120):
bfp8/bfp8 34.1 / 40.8 / 44.0 / 50.7 / 73.5 / 275.4 / 681.7 (plan and output unchanged, bit-identical);
bfp8 x + bf16 h 34.2 / 41.9 / 44.2 / 55.2 / 102.0 / 382.8 / 955.9; bf16 x + bfp8 h 34.2 / 41.0 / 44.1 / 49.5 / 91.4 /
366.2 / 917.0; bf16/bf16 34.3 / 41.8 / 44.2 / 55.7 / 100.2 / 374.5 / 933.5.
Accuracy (LoFi, layer 4 real input, rel L2 vs fp32 on the same bits): 0.0188 / 0.0130 / 0.0120 / 0.0101.
Unit tests (correctness): 21/21. Determinism (test_flat_stress.py, 2x4 mesh, regimes switching every 10k calls):
2,000,000 mesh calls (16M chip calls), 1,999,972 compared bit-exact, all markers 0, no hang (1 h 50 min).

## Attribution of the remaining cost (after the 12 ideas)
bfp8 x / h, M 2048, us per expert, 128-row vs 64-row sub-blocks: real 275.6 / 410.7; x never moves 273.1 / 413.8;
no y writes 273.5 / 276.0; neither 273.0 / 272.7 -> the 64-row cost is entirely the row-major y writes (the down y out
CB is sized from two bfp8 sub-blocks: ~2 row-major row tiles at 64 rows).
13. (over budget) 8-row y out CB for the bf16 regimes: bit-identical, no gain (bf16 both 374.2 at 2048, 990.8 at 5120
    vs 933.5). Reverted: for bf16 h the y out CB is not the limit.
bf16 x + h, M 512 / 2048: real 100.8 / 375.5; no y writes 91.1 / 335.1; x never moves 93.6 / 406.6; neither 82.4 /
294.4 (bfp8 core 70.0 / 273.0). Of bf16 both's ~100 us over bfp8 at 2048: ~21 us is intrinsic (bf16 h down unpack
and h exchange, x and y not moving), ~80 us is NoC traffic interaction: twice the x multicast (NOC0) and h exchange
(NOC1) bytes contending with the row-major y writes; the flows are coupled (removing x alone is slower).
Next lever (not tried): the y writes' route / timing against the bf16 x and h streams (e.g. y on the x multicast's
quiet windows, or the h exchange on NOC0), and h held row by row on the down cores (3 x 512 KB at 128-row sub-blocks).
Full isolation, M 2048, us per expert (normal / no y writes / x never moves and no y writes; bfp8 core 273):
bf16 x + bfp8 h (128-row): 366.4 / 366.2 / 321.1 -> bf16 x gate/up compute +48 (unpack-bound on 2 KB x tiles), x
delivery +45, y writes 0. bfp8 x + bf16 h (64-row): 383.1 / 350.2 / 295.1 -> bf16 h compute +22, x path +55 (the
64-row sub-blocks: x delivered in half-size blocks; it is not the x bytes), y writes +33 (contending with the doubled
h traffic). bf16 x + h: 375.5 / 335.1 / 294.4 -> +21 compute, +41 x path, +40 y writes. Two of the three parts come
from the 64-row sub-blocks that the bf16 h buffers force; the levers are 128-row sub-blocks with bf16 h (h held row by
row on the down cores) and the y writes kept off the h traffic.

## Next ideas (not tried)
Stakes: bf16 x + h is 375.5 vs 275.6 us per expert at M 2048 (+21 compute, +41 x path, +40 y writes). In the model's
final chunk the flat op is 101.0 ms (bf16/bf16) vs 82.8 ms (bfp8/bfp8) on the slowest chip (RT profiler).
A. 128-row sub-blocks with bf16 h: hold h on the down cores row tile by row tile. Today the down cores take h as whole
   sub-block buffers; at 128 rows a bf16 h buffer is 512 KB and three do not fit, so the plan drops bf16 h to 64-row
   sub-blocks (mt 2), and the 64-row sub-blocks cause both the x path cost (x delivered in half-size blocks, +41) and,
   through the small y out CB, part of the y cost. Instead: gate/up still computes 128-row sub-blocks (x ring and
   full-sync DST as for bfp8 h), but hands h over per 32-row row tile into a ring of row-tile slots on the down cores
   (128 KB each at I 2048; 4-6 slots), released as soon as the down matmul has consumed that row tile. Expected: the
   x path cost goes away (bfp8 h at 128 rows: 275.6 vs 410.7 at 64 rows), leaving +21 compute and the y share.
   Check: plan L1 budget at 1427 KB, the early down "done" per row tile, bit-identical output vs today's bf16 h.
B. Keep the row-major y writes off the h (and bf16 x) traffic. With bfp8 h the y writes cost ~0; with bf16 h they cost
   +33..40 us because they contend with the doubled h exchange on NOC1 (VC changes alone did not separate them).
   Options, cheapest first: (1) h exchange on NOC0, y stays on NOC1 (MIMO_FL_DN_NOC0 / H_VC probes are the starting
   point); (2) y writer moved to the down core's NCRISC on NOC0 (possible since "done" no longer waits on y);
   (3) y written in the h exchange's quiet windows (after a sub-block's h is consumed, before the next one arrives).
   Re-measure with the isolation switches (MIMO_FL_YRM_NOWRITE, MIMO_FL_X_RESIDENT): the target is "normal" close to
   "no y writes" (335.1 at M 2048).
Do A first: it shrinks the sub-block count and so also the number of h/y interleavings B has to manage.

## Other expert shapes (balanced sweep, one chip, 36 experts, us per expert; plan mt / hbuf)
M:                     32    64   128   256   512  1024  2048 | math90 @2048 | plan bfp8 / bf16
H 4096 I 2048 bfp8   34.1  36.3  40.9  51.0  73.5 140.5 275.5 | 75%          | mt 4 hb 3 / mt 2 hb 4
              bf16   34.2  36.5  42.0  55.6 100.5 191.1 374.0 | 55%          |
H 7168 I 2048 bfp8   61.5  64.0  70.4  82.8 119.2 223.7 437.2 | 83%          | mt 2 hb 3 / mt 2 hb 3
              bf16   61.8  64.3  71.8  87.1 159.4 304.8 595.9 | 61%          |
H 6144 I 2048 bfp8   52.4  55.2  61.0  73.4 114.6 222.1 437.1 | 71%          | mt 4 hb 3 / mt 2 hb 3
              bf16   52.3  55.1  62.1  82.5 139.9 265.2 515.7 | 60%          |
H 3584 I 3072 bfp8   48.4  50.3  55.3  67.4 104.1 204.6 405.1 | 67%          | mt 2 hb 3 / mt 2 hb 2 (np 2)
              bf16   48.9  51.2  57.6  91.9 174.6 339.4 668.9 | 41%          |
All pass. bf16 costs <= 4% up to M 128 at every shape. H 7168 runs 64-row sub-blocks even in bfp8 and still reaches
83%: the 64-row penalty seen at H 4096 is shape-specific. H 6144 bfp8 equals H 7168 bfp8 at M 2048 with 6/7 of the
work (71% vs 83%): not explained yet. I 3072: bf16 h only gets 2 h buffers (+65% at 2048) -> idea A matters most here.
