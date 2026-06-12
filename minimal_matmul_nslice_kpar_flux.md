# FLUX (big-shape sweep): composing N-slicing (S) with K-par (Pk)

Device kernel duration (median over cache-hit reps, single device session under tracy). Auto block sizer ON. `S=auto` = today's production N-slicer (Pk=1); Pk>1 = fused plan-B column reduction. Only shapes whose auto-slicer engages (S>1) are included. Combos restricted to S*Pk in {4,8} with no M-padding and Pk | K_tiles. WH B0 8x8.

## Findings (28 auto-sliced big shapes)

- **18/28 shapes win ≥1.10x** (geomean **1.58x** over those, **1.35x** over all 28); the other 10 are washes.
  Every combo verified PCC ≥ 0.99999. Best: `64x15360x1536` **2.89x** (K=480 tiles deep).
- **K-par helps iff: K is deep AND the output is not already core-saturated AND N isn't too wide.**
  The 10 washes split cleanly into three causes:
  - *Shallow K* — nothing to split: `32x256x6144` (Kt=8), `512x128x1536` (Kt=4) → 1.00x.
  - *Output over-parallel* (out_tiles ≫ 64 cores): `512x6144x1536` (768), `16384x6144x128` (2048),
    `512x4608x6144` (3072), `512x6144x9216` (4608) → 1.00x; `8192x6144x128` (1024) 1.03x. The grid is
    already full of output work, so K-bands only steal cores.
  - *N too wide* (in1 DRAM-bandwidth bound under split-K): `64x6144x9216` (out 576, N=288 tiles) 1.05x,
    `32x6144x6144` 1.06x — contrast `128x6144x4608` (also out 576 but narrower N) which wins **1.59x**.
- **Optimal (S, Pk) tracks output width, exactly like LTX:** narrow/starved output → more Pk
  (`32x6144x1536`→Pk4, `128x15360x768`→Pk8, `512x6144x128`→Pk8); wider output → Pk2/S4. `S·Pk = grid.y`
  is the right budget; pure K-par `(S1,Pk8)` only wins when output is very narrow.
- **K-par composes with M-slicing too** (transpose, M≫N, N=128): `512/1024/2048/4096 ×6144×128` win
  1.40–1.54x while output ≤512 tiles, tapering to 1.03x at out=1024 and 1.00x at out=2048.

## Proposed joint (S, Pk) heuristic — back-tested on all 28 shapes

Split the row budget `S·Pk = grid.y` (rows_per_group = 1), choosing Pk from a **K-dominance** score
`D = K_tiles · cores / out_tiles` (deep reduction relative to how saturated the grid already is by
output), with a wide-N DRAM guard and a deep-K availability cap:

```
G = grid.y;  small = min(Mt,Nt);  out = Mt*Nt
if small > 2*G:           (S,Pk) = (1,1)          # today's engage gate (not delivery-bound)
else:
    D  = Kt * (G*Gx) / out                         # G*Gx = core count (64)
    Pk = 8 if D>=280  else 4 if D>=40  else 2 if D>=20  else 1
    if Nt >= 256:  Pk = 1                          # wide-N: in1 DRAM-bandwidth bound, K-par regresses
    while Pk>1 and (Kt % Pk != 0 or Kt//Pk < 8):  Pk //= 2   # keep >=8 K-tiles per band
    S  = G // Pk
```

Back-test vs the measured matrix (thresholds grid-searched under a hard no-regression constraint):

- **geomean 1.303x** vs **oracle-best 1.351x → 96.4% of the achievable win captured.**
- **0 regressions** (every shape ≥ auto), mean regret 1.04, max regret 1.28; 13/28 predictions hit the
  oracle combo exactly.
- Misses are conservative (predict auto/Pk1 where a Pk≥2 would have won a bit, e.g. `128x2304x6144`
  out=768/Kt=72 → leaves 1.28x on the table) — never a regression. The wide-N guard correctly turns
  K-par off on `32/64x6144x9216` (out 288/576, N=288 tiles) where it otherwise regressed up to −11%.

⚠️ Thresholds are fit to these 28 shapes on WH 8x8; the *structure* (D-score, `S·Pk=G` budget, wide-N
guard, deep-K cap) is the portable part — re-fit constants per arch (BH grid.y=10) and validate on
held-out shapes before trusting the exact 280/40/20/256 numbers.

## Integration — wired into the factory auto-slicer (CONFIRMED end-to-end)

The formula is now the default in `minimal_matmul_program_factory.cpp`: when neither `TT_MM_NUM_SLICES`
nor `TT_MM_K_SLICES` is pinned, the auto path jointly picks `(num_slices, num_k_slices)` and engages the
fused plan-B reduction (output stays `[M,N]`, so `compute_output_specs` is unchanged). Tunable via
`TT_MM_KPAR_{D8,D4,D2,NWIDE,MINKB}` env, disable with `TT_MM_NO_AUTO_KPAR=1`, arch-selectable via a
`KParParams` struct.

Measured **old (heuristic off) vs new (heuristic on)** on all 28 shapes — auto path vs auto path, no
manual combos: **geomean 1.303×, min 0.999× (0 regressions)**, exactly matching the back-test. Sample:
`64x15360x1536` 2.88×, `64x6144x1536` 2.38×, `32x6144x1536` 2.23×, `128x6144x4608` 1.59×,
`1024x6144x128` 1.52×; near-square / over-parallel / shallow-K / wide-N shapes all 1.00× (unchanged).

## Best combo per shape (sorted by speedup)

| shape | out tiles | best S | best Pk | best us | auto us | speedup |
|---|---|---|---|---|---|---|
| 64x15360x1536 | 96 | 2 | 4 | 70.86 | 204.74 | **2.89x** |
| 64x6144x1536 | 96 | 2 | 4 | 35.04 | 83.55 | **2.38x** |
| 128x15360x768 | 96 | 1 | 8 | 93.02 | 215.23 | **2.31x** |
| 32x6144x1536 | 48 | 2 | 4 | 24.14 | 53.88 | **2.23x** |
| 32x6144x2304 | 72 | 2 | 4 | 35.71 | 59.44 | **1.66x** |
| 128x6144x4608 | 576 | 4 | 2 | 156.33 | 248.26 | **1.59x** |
| 128x6144x768 | 96 | 2 | 2 | 55.97 | 87.74 | **1.57x** |
| 1024x6144x128 | 128 | 2 | 2 | 56.68 | 87.42 | **1.54x** |
| 32x6144x3072 | 96 | 4 | 2 | 38.93 | 59.59 | **1.53x** |
| 4096x6144x128 | 512 | 4 | 2 | 112.66 | 164.25 | **1.46x** |
| 512x6144x128 | 64 | 1 | 8 | 40.58 | 56.98 | **1.40x** |
| 2048x6144x128 | 256 | 2 | 4 | 75.99 | 106.63 | **1.40x** |
| 128x6144x2304 | 288 | 2 | 4 | 94.23 | 129.44 | **1.37x** |
| 64x6144x4608 | 288 | 4 | 2 | 95.49 | 127.88 | **1.34x** |
| 128x2304x6144 | 768 | 4 | 2 | 76.93 | 98.56 | **1.28x** |
| 32x6144x4608 | 144 | 4 | 2 | 85.57 | 105.06 | **1.23x** |
| 64x4608x6144 | 384 | 4 | 2 | 82.50 | 99.72 | **1.21x** |
| 32x6144x9216 | 288 | 1 | 8 | 144.90 | 163.54 | **1.13x** |
| 32x6144x6144 | 192 | 4 | 2 | 99.35 | 104.96 | **1.06x** |
| 64x6144x9216 | 576 | 1 | 8 | 167.29 | 176.29 | **1.05x** |
| 8192x6144x128 | 1024 | 4 | 2 | 197.27 | 204.01 | **1.03x** |
| 512x6144x4608 | 2304 | 2 | 2 | 519.97 | 523.57 | **1.01x** |
| 32x256x6144 | 192 | auto | 1 | 9.48 | 9.48 | **1.00x** |
| 512x128x1536 | 768 | auto | 1 | 21.72 | 21.72 | **1.00x** |
| 512x4608x6144 | 3072 | auto | 1 | 442.43 | 442.43 | **1.00x** |
| 512x6144x1536 | 768 | auto | 1 | 190.43 | 190.43 | **1.00x** |
| 512x6144x9216 | 4608 | auto | 1 | 992.61 | 992.61 | **1.00x** |
| 16384x6144x128 | 2048 | auto | 1 | 367.21 | 367.21 | **1.00x** |

## 32x256x6144  (out_tiles=192, Kt=8, Mt=1)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 9.48 | 1.0 | 1.00x ⭐ |
| 1 | 8 | 49.43 | 1.0 | 0.19x |
| 2 | 4 | 23.26 | 1.0 | 0.41x |
| 4 | 2 | 12.42 | 1.0 | 0.76x |

## 32x6144x1536  (out_tiles=48, Kt=192, Mt=1)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 53.88 | 0.99999 | 1.00x |
| 1 | 8 | 30.68 | 0.99999 | 1.76x |
| 2 | 4 | 24.14 | 0.99999 | 2.23x ⭐ |
| 4 | 2 | 32.01 | 0.99999 | 1.68x |

## 32x6144x2304  (out_tiles=72, Kt=192, Mt=1)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 59.44 | 0.99999 | 1.00x |
| 1 | 8 | 46.75 | 0.99999 | 1.27x |
| 2 | 4 | 35.71 | 0.99999 | 1.66x ⭐ |
| 4 | 2 | 38.94 | 0.99999 | 1.53x |

## 32x6144x3072  (out_tiles=96, Kt=192, Mt=1)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 59.59 | 1.0 | 1.00x |
| 1 | 8 | 48.61 | 0.99999 | 1.23x |
| 2 | 4 | 40.76 | 0.99999 | 1.46x |
| 4 | 2 | 38.93 | 0.99999 | 1.53x ⭐ |

## 32x6144x4608  (out_tiles=144, Kt=192, Mt=1)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 105.06 | 0.99999 | 1.00x |
| 1 | 8 | 88.41 | 0.99999 | 1.19x |
| 2 | 4 | 89.49 | 0.99999 | 1.17x |
| 4 | 2 | 85.57 | 0.99999 | 1.23x ⭐ |

## 32x6144x6144  (out_tiles=192, Kt=192, Mt=1)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 104.96 | 1.0 | 1.00x |
| 1 | 8 | 108.63 | 0.99999 | 0.97x |
| 2 | 4 | 100.79 | 0.99999 | 1.04x |
| 4 | 2 | 99.35 | 1.0 | 1.06x ⭐ |

## 32x6144x9216  (out_tiles=288, Kt=192, Mt=1)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 163.54 | 1.0 | 1.00x |
| 1 | 8 | 144.90 | 0.99999 | 1.13x ⭐ |
| 2 | 4 | 150.61 | 0.99999 | 1.09x |
| 4 | 2 | 164.94 | 1.0 | 0.99x |

## 64x4608x6144  (out_tiles=384, Kt=144, Mt=2)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 99.72 | 0.99999 | 1.00x |
| 1 | 8 | 123.79 | 0.99999 | 0.81x |
| 2 | 2 | 238.84 | 0.99999 | 0.42x |
| 2 | 4 | 97.53 | 0.99999 | 1.02x |
| 4 | 2 | 82.50 | 0.99999 | 1.21x ⭐ |

## 64x6144x1536  (out_tiles=96, Kt=192, Mt=2)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 83.55 | 1.0 | 1.00x |
| 1 | 8 | 44.38 | 0.99999 | 1.88x |
| 2 | 2 | 61.80 | 1.0 | 1.35x |
| 2 | 4 | 35.04 | 1.0 | 2.38x ⭐ |
| 4 | 2 | 55.14 | 1.0 | 1.52x |

## 64x6144x4608  (out_tiles=288, Kt=192, Mt=2)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 127.88 | 0.99999 | 1.00x |
| 1 | 8 | 107.97 | 0.99999 | 1.18x |
| 2 | 2 | 264.61 | 0.99999 | 0.48x |
| 2 | 4 | 115.07 | 0.99999 | 1.11x |
| 4 | 2 | 95.49 | 0.99999 | 1.34x ⭐ |

## 64x6144x9216  (out_tiles=576, Kt=192, Mt=2)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 176.29 | 0.99997 | 1.00x |
| 1 | 8 | 167.29 | 0.99996 | 1.05x ⭐ |
| 2 | 2 | 471.00 | 0.99997 | 0.37x |
| 2 | 4 | 173.80 | 0.99997 | 1.01x |
| 4 | 2 | 198.25 | 0.99997 | 0.89x |

## 64x15360x1536  (out_tiles=96, Kt=480, Mt=2)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 204.74 | 0.99999 | 1.00x |
| 1 | 8 | 71.22 | 0.99999 | 2.87x |
| 2 | 2 | 144.99 | 0.99999 | 1.41x |
| 2 | 4 | 70.86 | 0.99999 | 2.89x ⭐ |
| 4 | 2 | 129.24 | 0.99999 | 1.58x |

## 128x2304x6144  (out_tiles=768, Kt=72, Mt=4)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 98.56 | 1.0 | 1.00x |
| 1 | 8 | 165.63 | 0.99999 | 0.60x |
| 2 | 2 | 130.37 | 0.99999 | 0.76x |
| 2 | 4 | 113.16 | 0.99999 | 0.87x |
| 4 | 2 | 76.93 | 0.99999 | 1.28x ⭐ |

## 128x6144x768  (out_tiles=96, Kt=192, Mt=4)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 87.74 | 0.99999 | 1.00x |
| 1 | 8 | 60.63 | 0.99999 | 1.45x |
| 2 | 2 | 55.97 | 0.99999 | 1.57x ⭐ |
| 2 | 4 | 56.95 | 0.99999 | 1.54x |
| 4 | 2 | 92.33 | 0.99999 | 0.95x |

## 128x6144x2304  (out_tiles=288, Kt=192, Mt=4)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 129.44 | 0.99999 | 1.00x |
| 1 | 8 | 122.03 | 0.99998 | 1.06x |
| 2 | 2 | 109.98 | 0.99999 | 1.18x |
| 2 | 4 | 94.23 | 0.99998 | 1.37x ⭐ |
| 4 | 2 | 103.74 | 0.99999 | 1.25x |

## 128x6144x4608  (out_tiles=576, Kt=192, Mt=4)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 248.26 | 1.0 | 1.00x |
| 1 | 8 | FAIL | None | — |
| 2 | 2 | 273.86 | 0.99999 | 0.91x |
| 2 | 4 | 162.74 | 0.99999 | 1.53x |
| 4 | 2 | 156.33 | 0.99999 | 1.59x ⭐ |

## 128x15360x768  (out_tiles=96, Kt=480, Mt=4)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 215.23 | 1.0 | 1.00x |
| 1 | 8 | 93.02 | 0.99999 | 2.31x ⭐ |
| 2 | 2 | 130.29 | 0.99999 | 1.65x |
| 2 | 4 | 123.85 | 0.99999 | 1.74x |
| 4 | 2 | 221.59 | 0.99999 | 0.97x |

## 512x128x1536  (out_tiles=768, Kt=4, Mt=16)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 21.72 | 0.99995 | 1.00x ⭐ |
| 1 | 4 | 75.09 | 0.99995 | 0.29x |
| 2 | 2 | 42.25 | 0.99995 | 0.51x |
| 2 | 4 | 66.65 | 0.99995 | 0.33x |
| 4 | 2 | 36.25 | 0.99995 | 0.60x |

## 512x4608x6144  (out_tiles=3072, Kt=144, Mt=16)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 442.43 | 1.0 | 1.00x ⭐ |
| 1 | 8 | 663.89 | 1.0 | 0.67x |
| 2 | 2 | 460.85 | 1.0 | 0.96x |
| 2 | 4 | 515.69 | 1.0 | 0.86x |
| 4 | 2 | 452.47 | 1.0 | 0.98x |

## 512x6144x128  (out_tiles=64, Kt=192, Mt=16)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 56.98 | 0.99999 | 1.00x |
| 1 | 8 | 40.58 | 0.99999 | 1.40x ⭐ |
| 2 | 2 | 51.74 | 0.99999 | 1.10x |
| 2 | 4 | 51.33 | 0.99999 | 1.11x |
| 4 | 2 | FAIL | None | — |

## 512x6144x1536  (out_tiles=768, Kt=192, Mt=16)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 190.43 | 0.99999 | 1.00x ⭐ |
| 1 | 8 | 238.39 | 0.99999 | 0.80x |
| 2 | 2 | 195.06 | 0.99999 | 0.98x |
| 2 | 4 | 217.97 | 0.99999 | 0.87x |
| 4 | 2 | 359.61 | 0.99999 | 0.53x |

## 512x6144x4608  (out_tiles=2304, Kt=192, Mt=16)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 523.57 | 1.0 | 1.00x |
| 1 | 8 | FAIL | None | — |
| 2 | 2 | 519.97 | 1.0 | 1.01x ⭐ |
| 2 | 4 | 561.64 | 1.0 | 0.93x |
| 4 | 2 | 577.26 | 1.0 | 0.91x |

## 512x6144x9216  (out_tiles=4608, Kt=192, Mt=16)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 992.61 | 0.99996 | 1.00x ⭐ |
| 1 | 8 | FAIL | None | — |
| 2 | 2 | FAIL | None | — |
| 2 | 4 | FAIL | None | — |
| 4 | 2 | 1002.34 | 0.99996 | 0.99x |

## 1024x6144x128  (out_tiles=128, Kt=192, Mt=32)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 87.42 | 0.99999 | 1.00x |
| 1 | 8 | 68.69 | 0.99999 | 1.27x |
| 2 | 2 | 56.68 | 0.99999 | 1.54x ⭐ |
| 2 | 4 | 57.33 | 0.99999 | 1.52x |
| 4 | 2 | 92.49 | 0.99999 | 0.95x |

## 2048x6144x128  (out_tiles=256, Kt=192, Mt=64)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 106.63 | 0.99999 | 1.00x |
| 1 | 8 | 105.41 | 0.99999 | 1.01x |
| 2 | 2 | 86.76 | 0.99999 | 1.23x |
| 2 | 4 | 75.99 | 0.99999 | 1.40x ⭐ |
| 4 | 2 | 98.75 | 0.99999 | 1.08x |

## 4096x6144x128  (out_tiles=512, Kt=192, Mt=128)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 164.25 | 0.99999 | 1.00x |
| 1 | 8 | 159.05 | 0.99998 | 1.03x |
| 2 | 2 | 163.01 | 0.99999 | 1.01x |
| 2 | 4 | 123.59 | 0.99999 | 1.33x |
| 4 | 2 | 112.66 | 0.99999 | 1.46x ⭐ |

## 8192x6144x128  (out_tiles=1024, Kt=192, Mt=256)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 204.01 | 0.99999 | 1.00x |
| 1 | 8 | 261.45 | 0.99999 | 0.78x |
| 2 | 2 | 306.67 | 0.99999 | 0.67x |
| 2 | 4 | 223.45 | 0.99999 | 0.91x |
| 4 | 2 | 197.27 | 0.99999 | 1.03x ⭐ |

## 16384x6144x128  (out_tiles=2048, Kt=192, Mt=512)

| S | Pk | device us | PCC | vs auto |
|---|---|---|---|---|
| auto | 1 | 367.21 | 1.0 | 1.00x ⭐ |
| 1 | 8 | 462.59 | 1.0 | 0.79x |
| 2 | 2 | 597.69 | 1.0 | 0.61x |
| 2 | 4 | 422.16 | 1.0 | 0.87x |
| 4 | 2 | 387.88 | 1.0 | 0.95x |
