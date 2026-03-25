# Conv3D Blocking Sweep Results — BH p150b LB (v2: Correct Production Shapes)

**Machine:** BH Loud Box, 8x p150b (130 cores, 13x10 grid)
**Branch:** `kevinmi/conv3d-blocking-sweep-v2`
**Production params:** WAN 2.2 T2V, 81 output frames, latent T=21
**Blocking format:** (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)

**v2 changes from v1:**
- Correct uncached temporal dimensions: T=23/22/43/42/83/81 (was T=3 for all)
- Galaxy configs use 12x10 grid override (was 13x10)
- Shapes verified by tracing actual decoder runs (see CONV3D_PRODUCTION_SHAPES.md)

---

## BH Galaxy 6U (bh_4x32) — h_factor=4, w_factor=32 — 12x10 grid

### Uncached 720p (Priority) — Brute-Force Optimized

Swept on BH p150b LB (1x1 mesh, 12x10 grid). Brute-force exhaustive search over all valid
(C_in_block, C_out_block, H_out_block, W_out_block) combinations. Results saved in `sweep_results_v2/`.

| Layer | C_in→C_out | Kernel | Shape (T,H,W) | Count | Staged Blocking | Staged us | Best Blocking | Best us | Speedup |
|-------|-----------|--------|---------------|-------|-----------------|-----------|---------------|---------|---------|
| conv_in | 32→384 | (3,3,3) | (23,25,7) | x1 | (32,128,1,4,16) | 232 | **(32,128,1,16,2)** | **189** | 1.23x |
| mid+up0 res | 384→384 | (3,3,3) | (23,25,7) | x10 | (128,64,1,8,2) | 1031 | **(128,64,1,8,4)** | **810** | 1.27x |
| up0 time_conv | 384→768 | (3,1,1) | (22,23,5) | x1 | (96,256,1,16,2) | 307 | **(128,256,1,8,4)** | **279** | 1.10x |
| up0 spatial | 384→192 | (1,3,3) | (41,48,12) | x1 | (96,96,1,16,8) | 783 | **(192,96,1,16,2)** | **605** | 1.29x |
| up1 res0 | 192→384 | (3,3,3) | (43,48,12) | x1 | (96,128,1,16,2) | 2011 | **(96,128,1,16,2)** | **1986** | 1.01x |
| up1 resblocks | 384→384 | (3,3,3) | (43,48,12) | x5 | (128,128,1,16,1) | 4859 | **(96,128,1,16,2)** | **3477** | **1.40x** |
| up1 time_conv | 384→768 | (3,1,1) | (42,46,10) | x1 | (384,256,1,8,2) | 936 | pending | | |
| up1 spatial | 384→192 | (1,3,3) | (81,94,22) | x1 | (128,96,1,32,8) | 4238 | pending | | |
| up2 resblocks | 192→192 | (3,3,3) | (83,94,22) | x6 | (96,96,1,16,4) | 7314 | **(96,96,1,8,8)** | **6767** | 1.08x |
| up2 spatial | 192→96 | (1,3,3) | (81,186,42) | x1 | (192,96,1,8,8) | 3388 | pending | | |
| up3 resblocks | 96→96 | (3,3,3) | (83,186,42) | x6 | (96,96,1,8,8) | 7997 | **(96,96,1,8,8)** | **8117** | ~same |
| conv_out | 96→3 | (3,3,3) | (83,186,42) | x1 | (96,32,1,16,8) | 5955 | pending | | |

**Key improvement: up1 resblocks (384→384) 1.40x faster** — staged sweep missed C_in=96,C_out=128 because it was locked to baseline C_in=128,C_out=128 during spatial sweep. Saves ~6.9 ms/frame (5 calls x 1382 us).

### Uncached 480p

Swept on BH p150b LB (1x1 physical mesh simulating 4x32 local dims, 12x10 grid override).

| Layer | C_in→C_out | Kernel | Shape (T,H,W) | Best Blocking | us |
|-------|-----------|--------|---------------|---------------|-----|
| conv_in | 32→384 | (3,3,3) | (23,17,5) | (32,64,1,16,4) | 137 |
| mid+up0 res | 384→384 | (3,3,3) | (23,17,5) | (128,64,1,16,1) | 506 |
| up0 time_conv | 384→768 | (3,1,1) | (22,15,3) | (96,256,1,8,2) | 221 |
| up0 spatial | 384→192 | (1,3,3) | (41,32,8) | (128,64,1,16,8) | 351 |
| up1 res0 | 192→384 | (3,3,3) | (43,32,8) | (96,128,1,4,8) | 933 |
| up1 resblocks | 384→384 | (3,3,3) | (43,32,8) | (128,128,1,8,2) | 2006 |
| up1 time_conv | 384→768 | (3,1,1) | (42,30,6) | (192,256,1,8,2) | 546 |
| up1 spatial | 384→192 | (1,3,3) | (81,62,14) | (192,96,1,16,8) | 1631 |
| up2 resblocks | 192→192 | (3,3,3) | (83,62,14) | (96,96,1,16,4) | 2993 |
| up2 spatial | 192→96 | (1,3,3) | (81,122,26) | (192,96,1,8,4) | 1384 |
| up3 resblocks | 96→96 | (3,3,3) | (83,122,26) | (96,96,1,8,4) | 3267 |
| conv_out | 96→3 | (3,3,3) | (83,122,26) | (96,32,1,16,8) | 2426 |

---

---

## BH Galaxy (bh_4x8) — h_factor=4, w_factor=8 — 12x10 grid

### Uncached 720p

Swept on BH p150b LB (1x1 physical mesh simulating 4x8 local dims, 12x10 grid override).

| Layer | C_in→C_out | Kernel | Shape (T,H,W) | Best Blocking | us |
|-------|-----------|--------|---------------|---------------|-----|
| conv_in | 32→384 | (3,3,3) | (23,25,22) | (32,128,1,16,16) | 513 |
| mid+up0 res | 384→384 | (3,3,3) | (23,25,22) | (128,128,1,4,4) | 2608 |
| up0 time_conv | 384→768 | (3,1,1) | (22,23,20) | (192,256,1,4,4) | 665 |
| up0 spatial | 384→192 | (1,3,3) | (41,48,42) | (96,96,1,16,16) | 2238 |
| up1 res0 | 192→384 | (3,3,3) | (43,48,42) | (96,128,1,8,4) | 6539 |
| up1 resblocks | 384→384 | (3,3,3) | (43,48,42) | (128,128,1,2,8) | 17908 |
| up1 time_conv | 384→768 | (3,1,1) | (42,46,40) | (384,256,1,2,16) | 2831 |
| up1 spatial | 384→192 | (1,3,3) | (81,94,82) | (128,96,1,16,16) | 13481 |
| up2 resblocks | 192→192 | (3,3,3) | (83,94,82) | (96,96,1,4,16) | 27725 |
| up2 spatial | 192→96 | (1,3,3) | (81,186,162) | (192,96,1,4,32) | 13281 |
| up3 resblocks | 96→96 | (3,3,3) | (83,186,162) | (96,96,1,8,8) | 32441 |
| conv_out | 96→3 | (3,3,3) | (83,186,162) | (96,32,1,8,16) | 23075 |

### Uncached 480p

Swept on BH p150b LB (1x1 physical mesh simulating 4x8 local dims, 12x10 grid override).

| Layer | C_in→C_out | Kernel | Shape (T,H,W) | Best Blocking | us |
|-------|-----------|--------|---------------|---------------|-----|
| conv_in | 32→384 | (3,3,3) | (23,17,15) | (32,128,1,16,8) | 210 |
| mid+up0 res | 384→384 | (3,3,3) | (23,17,15) | (128,128,1,16,1) | 1332 |
| up0 time_conv | 384→768 | (3,1,1) | (22,15,13) | (192,256,1,8,8) | 320 |
| up0 spatial | 384→192 | (1,3,3) | (41,32,28) | (128,64,1,8,16) | 1057 |
| up1 res0 | 192→384 | (3,3,3) | (43,32,28) | (96,128,1,16,2) | 3373 |
| up1 resblocks | 384→384 | (3,3,3) | (43,32,28) | (128,128,1,8,2) | 8132 |
| up1 time_conv | 384→768 | (3,1,1) | (42,30,26) | (192,256,1,4,4) | 1715 |
| up1 spatial | 384→192 | (1,3,3) | (81,62,54) | (128,96,1,32,8) | 6486 |
| up2 resblocks | 192→192 | (3,3,3) | (83,62,54) | (96,96,1,16,4) | 12286 |
| up2 spatial | 192→96 | (1,3,3) | (81,122,106) | (192,96,1,8,4) | 5797 |
| up3 resblocks | 96→96 | (3,3,3) | (83,122,106) | (96,96,1,8,8) | 13981 |
| conv_out | 96→3 | (3,3,3) | (83,122,106) | (96,32,1,16,8) | 9826 |

---

---

## BH Loud Box (bh_2x4) — h_factor=2, w_factor=4 — 13x10 grid

### Uncached 480p

Swept on BH p150b LB (1x1 physical mesh simulating 2x4 local dims, full 13x10 grid).

| Layer | C_in→C_out | Kernel | Shape (T,H,W) | Best Blocking | us |
|-------|-----------|--------|---------------|---------------|-----|
| conv_in | 32→384 | (3,3,3) | (23,32,28) | (32,128,1,16,16) | 625 |
| mid+up0 res | 384→384 | (3,3,3) | (23,32,28) | (128,128,1,8,2) | 4398 |
| up0 time_conv | 384→768 | (3,1,1) | (22,30,26) | (128,384,1,8,2) | 1023 |
| up0 spatial | 384→192 | (1,3,3) | (41,62,54) | (128,96,1,16,16) | 3099 |
| up1 res0 | 192→384 | (3,3,3) | (43,62,54) | (96,128,1,4,8) | 10534 |
| up1 resblocks | 384→384 | (3,3,3) | (43,62,54) | (128,128,1,4,4) | 26326 |
| up1 time_conv | 384→768 | (3,1,1) | (42,60,52) | (384,384,1,2,8) | 7216 |
| up1 spatial | 384→192 | (1,3,3) | (81,122,106) | (128,96,1,16,16) | 22030 |
| up2 resblocks | 192→192 | (3,3,3) | (83,122,106) | (96,96,1,8,8) | 45328 |
| up2 spatial | 192→96 | (1,3,3) | (81,242,210) | (192,96,1,16,8) | 22513 |
| up3 resblocks | 96→96 | (3,3,3) | (83,242,210) | (96,96,1,8,8) | 54564 |
| conv_out | 96→3 | (3,3,3) | (83,242,210) | (96,32,1,16,8) | 38012 |

---

## Cached Mode Notes

Cached mode uses T=3 for all conv3d layers (1 frame + CACHE_T=2 from activation cache). Spatial dims are the same as uncached. Cached sweeps completed for all mesh configs; some configs were deduplicated with uncached results at shared (C_in, C_out, kernel, T=3, H, W) shapes.

The blocking table is keyed by (h_factor, w_factor, C_in, C_out, kernel) — spatial blockings from the uncached sweep apply to cached mode as well since they share the same H/W dims.

---

## Sweep Status

- [x] bh_4x32 720p+480p uncached — 12x10 grid
- [x] bh_4x8 720p+480p uncached — 12x10 grid
- [x] bh_2x4 480p uncached — 13x10 grid
- [x] All cached configs (T=3, same spatial dims)
- [ ] wh_4x8 — needs WH hardware

---
*Last updated: 2026-03-25 — all uncached + cached configs swept with correct production shapes (81 frames, latent T=21).*
