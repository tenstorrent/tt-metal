# minimal_matmul on Blackhole: optimized-main baseline vs branch

**Board:** single Blackhole p150b, compute grid **11×10** (gx=11, gy=10).
**Branch:** `cglagovich/minimal-matmul-mcast-prefetch` (+ two fixes made for this run, below).
**Config:** bf16 in/out, **fp32 accumulate**, HiFi2, packer L1 acc. 12 reps/shape (3 warmup dropped), median.
**Date:** 2026-06-12.

- **baseline** = the *optimized* main: pure unicast (`TT_MM_NO_LARGE_LEVERS=1`, every branch feature gated
  off), **swept over block sizes**, best block per shape. This is main given its best possible blocking —
  not a fixed 8/8/8. The winning block is shown in the tables as `(M/K/N sbHW)`.
- **branch** = the production auto path (K-cap + auto block sizer + subblock DST-maximizer + auto
  core-grid slicing + mcast/prefetch gate).
- **speedup = baseline / branch** (>1 ⇒ branch faster). All 82 shapes passed PCC ≥ 0.9999.

---

## Headline

| set | shapes | geomean | wins >1.05 | within ±5% | losses <0.95 | range |
|---|--:|--:|--:|--:|--:|--:|
| **combined** | 82 | **1.150x** | 38 | 31 | 13 | 0.74–1.85x |
| big | 65 | 1.145x | 30 | 25 | 10 | 0.74–1.74x |
| ltx | 17 | 1.170x | 8 | 5 | 4 | 0.87–1.85x |

Wormhole reference (same study, 8×8 grid) was **1.36x** on the big set. Blackhole comes in lower —
but the reason is structural, not a regression in the levers, as the decomposition shows:

| subset | shapes | geomean | notes |
|---|--:|--:|---|
| **slicing-engaged** (skinny/skewed) | 29 | **1.487x** | **every one wins** (1.26–1.85x) |
| non-slicing (bulk square-ish) | 53 | **1.000x** | dead wash |

**On Blackhole the branch's entire value is the auto core-grid slicing on skinny shapes.** The
block-sizer / subblock-maximizer / mcast-prefetch levers that net +36% on Wormhole net **~0%** on BH's
bulk shapes — they were tuned against WH's 8×8 square grid, and BH's larger 11×10 grid changes the
arithmetic. The big aggregate win is real but comes almost entirely from the 29 skinny shapes.

---

## Two fixes were required to get here

1. **Slicing crash (the blocker).** The auto core-grid slicing picks `num_slices` = nearest power-of-2 to
   `sqrt(max/min)` over `[2, grid.y]`, but never checked that the value *divides* `grid.y`. On WH
   `grid.y=8` every power-of-2 divides it, so it never fired. On **BH `grid.y=10`** the only legal slice
   counts are 1 and 2 (10%4≠0, 10%8≠0), but the gate happily picked 4/8 → `TT_FATAL` at factory build.
   This crashed **all 29 skinny shapes** (every shape where slicing engages). Fix: constrain the
   candidate loop to `grid.y % c == 0`
   (`ttnn/.../minimal_matmul/device/minimal_matmul_program_factory.cpp`). After the fix all 29 run and
   PCC-pass; they are the 1.487x subset above.

2. **Baseline block-seeding (fairness).** The sweep harness seeded candidate blocks assuming M→rows / N→cols,
   but the factory transposes the grid when M>N (M→`grid.x`, N→`grid.y`). Invisible on WH's square grid;
   on BH's 11×10 it mis-centered the main baseline's candidate blocks for every M>N shape, which would
   have *under-stated* the main baseline. Fix: mirror the transpose in the seed
   (`tests/ttnn/nightly/unit_tests/operations/experimental/test_mm_repro.py`). So the baseline here is a
   genuinely strong main.

> Note: BH `grid.y=10` only admits a 2-way slice, vs up to 8-way on WH's `grid.y=8`. The slicing wins are
> still large, but BH is leaving slicing parallelism on the table that WH could use — a non-square grid
> with a power-of-2 row count (or slicing on `grid.x=11`... also non-pow2) is the structural limiter.

---

## Where the branch loses (the follow-up targets)

All 13 losses are **non-slicing** shapes where the branch's *auto block choice* is simply worse than the
best swept main block — i.e. the auto block sizer is picking suboptimal blocking on BH:

| shape | baseline (block) | branch | speedup |
|---|--:|--:|:--:|
| 4096x6144x2304 | 530.6us (6/4/4 sb22) | 716.1us | **0.74x** |
| 512x128x1536 | 8.9us (2/4/2 sb22) | 11.8us | 0.75x |
| 2048x6144x4608 | 561.4us (7/8/7 sb11) | 679.1us | 0.83x |
| 512x6144x1536 | 95.0us (2/8/5 sb21) | 112.9us | 0.84x |
| 1216x2048x1024 | 47.8us (4/4/4 sb22) | 54.8us | 0.87x |
| 1024x2304x6144 | 166.2us (4/4/9 sb41) | 187.5us | 0.89x |
| 1216x4096x512 | 59.0us (4/8/2 sb22) | 64.4us | 0.92x |
| 4864x2048x1024 | 127.1us (7/4/4 sb14) | 138.8us | 0.92x |

These are the place to chase next: re-tune the auto block sizer's L1 budget / tiebreak for BH's grid and
L1 (the sizer's `L1_CB_BUDGET` is still WH-sized ≈1.25 MiB), since on the bulk set the levers currently
net exactly 1.000x.

---

## Full results — big (65)

baseline = best swept block, pure unicast (== main); branch = auto path.

**geomean 1.145x** | wins(>1.05) 30/65 | within ±5% 25 | losses(<0.95) 10 | min 0.74x max 1.74x

| shape | baseline us (block) | branch us | speedup |
|---|--:|--:|:--:|
| 32x256x6144 | 23.2 (1/4/3 sb11) | 13.4 | 1.74x |
| 128x6144x4608 | 237.2 (1/16/4 sb14) | 146.4 | 1.62x |
| 32x6144x4608 | 234.8 (1/16/4 sb14) | 145.1 | 1.62x |
| 2048x6144x128 | 113.5 (6/8/1 sb21) | 70.3 | 1.61x |
| 64x6144x4608 | 236.2 (1/16/4 sb14) | 146.5 | 1.61x |
| 128x6144x2304 | 127.1 (1/16/4 sb14) | 80.0 | 1.59x |
| 32x6144x2304 | 124.2 (1/16/4 sb14) | 78.4 | 1.58x |
| 32x6144x3072 | 162.7 (1/4/9 sb11) | 104.3 | 1.56x |
| 4096x6144x128 | 205.4 (3/16/1 sb11) | 136.1 | 1.51x |
| 64x6144x9216 | 439.9 (1/32/4 sb14) | 294.1 | 1.50x |
| 32x6144x9216 | 437.4 (1/16/4 sb14) | 293.3 | 1.49x |
| 512x6144x4608 | 244.2 (2/4/14 sb22) | 167.3 | 1.46x |
| 512x4608x6144 | 243.7 (2/4/18 sb22) | 167.3 | 1.46x |
| 576x6144x6144 | 316.1 (2/4/18 sb22) | 218.0 | 1.45x |
| 64x6144x1536 | 91.9 (1/8/5 sb11) | 63.9 | 1.44x |
| 32x6144x1536 | 91.0 (1/8/5 sb11) | 63.6 | 1.43x |
| 576x6144x9216 | 458.5 (2/16/7 sb21) | 324.1 | 1.41x |
| 512x6144x9216 | 453.5 (2/16/7 sb21) | 321.3 | 1.41x |
| 16384x6144x128 | 755.6 (6/32/1 sb21) | 539.9 | 1.40x |
| 8192x6144x128 | 377.2 (3/32/1 sb11) | 271.0 | 1.39x |
| 128x6144x768 | 61.6 (1/8/3 sb11) | 45.0 | 1.37x |
| 128x2304x6144 | 127.4 (1/4/9 sb11) | 94.3 | 1.35x |
| 64x15360x1536 | 208.5 (1/16/5 sb11) | 156.1 | 1.34x |
| 64x4608x6144 | 237.7 (1/8/9 sb11) | 184.3 | 1.29x |
| 32x6144x6144 | 306.3 (1/8/9 sb11) | 243.4 | 1.26x |
| 128x15360x768 | 133.9 (1/16/3 sb11) | 109.5 | 1.22x |
| 1024x6144x128 | 62.7 (3/8/1 sb11) | 53.6 | 1.17x |
| 4096x2304x6144 | 578.4 (7/4/18 sb12) | 523.4 | 1.11x |
| 16384x2304x6144 | 2277.4 (24/4/5 sb41) | 2118.9 | 1.07x |
| 512x6144x128 | 43.9 (2/16/1 sb21) | 41.1 | 1.07x |
| 8192x6144x4608 | 2007.7 (12/8/8 sb22) | 1913.3 | 1.05x |
| 16512x3072x6144 | 2826.7 (12/4/10 sb22) | 2724.4 | 1.04x |
| 4224x3072x6144 | 770.0 (7/4/18 sb12) | 749.3 | 1.03x |
| 8192x4608x6144 | 1961.3 (12/4/10 sb22) | 1917.2 | 1.02x |
| 8192x6144x1536 | 656.9 (12/8/5 sb41) | 647.9 | 1.01x |
| 16384x6144x4608 | 3979.6 (12/8/8 sb22) | 3935.9 | 1.01x |
| 16384x6144x2304 | 2003.5 (12/8/4 sb22) | 1986.0 | 1.01x |
| 16512x6144x4608 | 3967.3 (12/8/8 sb22) | 3937.3 | 1.01x |
| 2048x6144x9216 | 1045.0 (4/4/27 sb41) | 1037.1 | 1.01x |
| 8256x6144x6144 | 2533.4 (12/4/5 sb41) | 2514.3 | 1.01x |
| 1024x128x768 | 10.2 (3/4/3 sb11) | 10.1 | 1.00x |
| 8192x6144x9216 | 3814.0 (4/4/27 sb41) | 3808.8 | 1.00x |
| 8256x6144x9216 | 3813.9 (4/4/27 sb41) | 3809.3 | 1.00x |
| 4096x6144x768 | 224.5 (12/8/3 sb41) | 224.3 | 1.00x |
| 1024x6144x768 | 83.8 (3/8/3 sb11) | 83.8 | 1.00x |
| 2048x128x1536 | 27.4 (6/4/5 sb21) | 27.4 | 1.00x |
| 4096x128x768 | 30.1 (12/4/3 sb41) | 30.3 | 0.99x |
| 2112x6144x6144 | 707.6 (4/4/9 sb41) | 718.1 | 0.99x |
| 2112x6144x9216 | 1022.8 (4/4/27 sb41) | 1038.2 | 0.99x |
| 2048x6144x1536 | 184.6 (6/4/5 sb21) | 189.3 | 0.97x |
| 1024x6144x2304 | 162.4 (4/4/7 sb41) | 167.9 | 0.97x |
| 1152x3072x6144 | 224.6 (4/4/9 sb41) | 232.3 | 0.97x |
| 16384x128x768 | 113.2 (6/4/3 sb21) | 117.6 | 0.96x |
| 8192x128x1536 | 105.0 (24/4/5 sb41) | 109.3 | 0.96x |
| 1152x6144x4608 | 309.6 (4/4/7 sb41) | 323.5 | 0.96x |
| 1024x6144x4608 | 306.4 (4/4/7 sb41) | 324.0 | 0.95x |
| 2048x4608x6144 | 534.7 (4/8/9 sb41) | 566.9 | 0.94x |
| 4224x6144x4608 | 1085.0 (7/4/14 sb12) | 1154.0 | 0.94x |
| 4096x6144x4608 | 1009.9 (7/4/14 sb12) | 1085.8 | 0.93x |
| 16384x6144x768 | 797.9 (6/16/3 sb21) | 860.4 | 0.93x |
| 1024x2304x6144 | 166.2 (4/4/9 sb41) | 187.5 | 0.89x |
| 512x6144x1536 | 95.0 (2/8/5 sb21) | 112.9 | 0.84x |
| 2048x6144x4608 | 561.4 (7/8/7 sb11) | 679.1 | 0.83x |
| 512x128x1536 | 8.9 (2/4/2 sb22) | 11.8 | 0.75x |
| 4096x6144x2304 | 530.6 (6/4/4 sb22) | 716.1 | 0.74x |

## Full results — ltx (17)

**geomean 1.170x** | wins(>1.05) 8/17 | within ±5% 5 | losses(<0.95) 4 | min 0.87x max 1.85x

| shape | baseline us (block) | branch us | speedup |
|---|--:|--:|:--:|
| 32x2048x2048 | 43.7 (1/4/6 sb12) | 23.6 | 1.85x |
| 32x2048x1536 | 38.3 (1/4/5 sb11) | 22.5 | 1.71x |
| 1216x4096x32 | 54.7 (4/8/1 sb41) | 36.7 | 1.49x |
| 4864x4096x32 | 161.8 (2/16/1 sb21) | 108.9 | 1.49x |
| 32x2048x32 | 14.1 (1/8/1 sb11) | 9.5 | 1.48x |
| 32x2048x512 | 20.8 (1/8/2 sb12) | 14.7 | 1.41x |
| 256x2048x1024 | 26.9 (1/8/3 sb11) | 20.6 | 1.30x |
| 4864x4096x512 | 175.2 (7/8/2 sb12) | 135.7 | 1.29x |
| 1216x4096x4096 | 201.9 (4/4/12 sb22) | 203.2 | 0.99x |
| 1216x4096x3072 | 152.0 (4/4/9 sb41) | 156.6 | 0.97x |
| 1216x4096x1024 | 79.2 (4/4/4 sb22) | 82.2 | 0.96x |
| 4864x4096x3072 | 571.5 (14/8/5 sb21) | 593.5 | 0.96x |
| 4864x4096x4096 | 732.0 (14/4/7 sb21) | 764.7 | 0.96x |
| 4864x4096x1024 | 217.1 (7/4/4 sb14) | 231.2 | 0.94x |
| 1216x4096x512 | 59.0 (4/8/2 sb22) | 64.4 | 0.92x |
| 4864x2048x1024 | 127.1 (7/4/4 sb14) | 138.8 | 0.92x |
| 1216x2048x1024 | 47.8 (4/4/4 sb22) | 54.8 | 0.87x |

---

## Reproduce

```bash
source /home/cglagovich/bh_env.sh && source python_env/bin/activate
bash build_metal.sh                      # includes the slicing-divisor fix
bash tools/mm_sweep/run_sweep.sh tools/mm_sweep/shapes_big.txt /tmp/mm_bh_big
bash tools/mm_sweep/run_sweep.sh tools/mm_sweep/shapes_ltx.txt /tmp/mm_bh_ltx
python tools/mm_sweep/parse_sweep.py /tmp/mm_bh_big tools/mm_sweep/shapes_big.txt /tmp/mm_bh_big/results.md
python tools/mm_sweep/parse_sweep.py /tmp/mm_bh_ltx tools/mm_sweep/shapes_ltx.txt /tmp/mm_bh_ltx/results.md
```
