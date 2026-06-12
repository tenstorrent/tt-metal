# Composing N-slicing (S) with K-par (Pk) on N-sliced shapes

Device kernel duration (median of 24 reps, 2 warmup dropped), auto block sizer ON (no pinned config). `S=auto` = today's production N-slicer (Pk=1). Pk>1 uses the fused plan-B on-device column reduction (TT_MM_K_FUSED=1). M=1-tile shapes constrain S*Pk=8 (rows_per_group=1) so K-par adds no M-padding.

WH B0 8x8 grid.

## Findings

1. **K-par + N-slicing compose, and the current N-slicer leaves a lot on the table.** On the starved
   M=1 shapes the production N-slicer (built before K-par) over-slices N and ignores the idle cores the
   deep sequential-K reduction wastes. Re-spending some of the row budget on K-bands wins big:
   `32×2048×512` goes **2.26x** (19.6→8.7µs), `32×2048×1536` **1.53x**, `32×2048×2048` **1.45x**.
2. **The optimal (S, Pk) split is a real 2D tradeoff that shifts with N.** As N widens there is more
   output parallelism to harvest, so the best Pk falls and S rises: best is `(S2,Pk4)` at N=512,
   `(S2,Pk4)` at N=1536, `(S4,Pk2)` at N=2048. Pure K-par `(S1,Pk8)` is actively *bad* once N is wide
   (0.76–0.93x) — it starves N parallelism. So a good policy is NOT "max Pk"; it's "balance S·Pk=grid.y".
3. **Even a grid-filling shape benefits.** `32×2048×2048` has exactly 64 output tiles (auto S=8 "fills"
   the grid), yet `(S4,Pk2)` still wins 1.45x — because the bottleneck there is sequential K depth, not
   core occupancy, and K-bands halve the per-core K reduction.
4. **K-par correctly does nothing when not output-starved.** `256×2048×1024` (256 output tiles ≫ 64
   cores) is a wash at best (`(S2,Pk2)` 1.01x) and *regresses* as you over-spend on S or Pk (`(S4,Pk2)`
   0.61x). Today's auto S=2 is already right; K-par should stay off here.

**Implication:** the N-slice heuristic should be re-derived jointly with Pk for output-starved shapes
(`M_tiles·N_tiles < cores`) — pick S·Pk = grid.y splitting the row budget between output-N parallelism
and K-reduction depth, biased toward more Pk when N is narrow. For non-starved shapes keep Pk=1 and the
existing S. (All combos verified PCC 0.99999, so this is purely a scheduling-policy change.)


## Best combo per shape

| model | shape | best S | best Pk | best us | speedup vs auto |
|---|---|---|---|---|---|
| LTX | 32x2048x512 | 2 | 4 | 8.70 | 2.26x |
| LTX | 32x2048x1536 | 2 | 4 | 12.81 | 1.53x |
| LTX | 32x2048x2048 | 4 | 2 | 13.69 | 1.45x |
| LTX | 256x2048x1024 | 2 | 2 | 39.69 | 1.01x |


## LTX  32x2048x512

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 19.61 | 0.99999 | 1.00x |
| 1 | 8 | 10.16 | 0.99999 | 1.93x |
| 2 | 4 | 8.70 | 0.99999 | 2.26x ⭐ |
| 4 | 2 | 11.74 | 0.99999 | 1.67x |
| 8 | 1 | 19.32 | 0.99999 | 1.02x |

## LTX  32x2048x1536

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 19.62 | 0.99999 | 1.00x |
| 1 | 8 | 21.10 | 0.99999 | 0.93x |
| 2 | 4 | 12.81 | 0.99999 | 1.53x ⭐ |
| 4 | 2 | 13.52 | 0.99999 | 1.45x |
| 8 | 1 | 19.59 | 0.99999 | 1.00x |

## LTX  32x2048x2048

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 19.87 | 0.99999 | 1.00x |
| 1 | 8 | 26.02 | 0.99999 | 0.76x |
| 2 | 4 | 15.09 | 0.99999 | 1.32x |
| 4 | 2 | 13.69 | 0.99999 | 1.45x ⭐ |
| 8 | 1 | 19.85 | 0.99999 | 1.00x |

## LTX  256x2048x1024

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 40.03 | 0.99999 | 1.00x |
| 2 | 1 | 40.04 | 0.99999 | 1.00x |
| 2 | 2 | 39.69 | 0.99999 | 1.01x ⭐ |
| 2 | 4 | 50.32 | 0.99999 | 0.80x |
| 4 | 1 | 62.36 | 0.99999 | 0.64x |
| 4 | 2 | 65.51 | 0.99999 | 0.61x |
