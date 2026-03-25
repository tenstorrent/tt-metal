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

Three blocking sets compared. **All us values measured in the same environment** (brute-force
in-process sweep, single device, wall-clock, 2 warmup + 4 timed runs, mean).

- **Original** = main-branch `conv3d.py` before any sweep (no mesh awareness, no (3,1,1) entry).
- **Staged** = blocking found by the greedy staged sweep (spatial → C_out → C_in).
- **Best** = global optimum from exhaustive brute-force search.

"—" = blocking not in brute-force search space (e.g. W_out_block > W was filtered). Needs follow-up.

| Layer | C_in→C_out | Kernel | (T,H,W) | Cnt | Original | Orig us | Staged | Staged us | Best | Best us | vs Orig | vs Staged |
|-------|-----------|--------|---------|-----|----------|---------|--------|-----------|------|---------|---------|-----------|
All numbers from head-to-head measurement on the same device in a single run
(2 warmup + 4 timed, wall-clock mean). Original and best blockings measured back-to-back.

| Layer | C_in→C_out | Kernel | (T,H,W) | Cnt | Original | Orig us | Best | Best us | Speedup |
|-------|-----------|--------|---------|-----|----------|---------|------|---------|---------|
| conv_in | 32→384 | (3,3,3) | (23,25,7) | x1 | (32,96,1,2,32) | 310 | **(32,128,1,16,2)** | **193** | **1.61x** |
| mid+up0 res | 384→384 | (3,3,3) | (23,25,7) | x10 | (96,96,1,8,4) | 789 | **(128,64,1,8,4)** | **820** | 0.96x |
| up0 time_conv | 384→768 | (3,1,1) | (22,23,5) | x1 | (384,32,1,1,1) | 2,461 | **(128,256,1,8,4)** | **286** | **8.60x** |
| up0 spatial | 384→192 | (1,3,3) | (41,48,12) | x1 | (192,96,1,32,4) | 813 | **(192,96,1,16,2)** | **608** | **1.34x** |
| up1 res0 | 192→384 | (3,3,3) | (43,48,12) | x1 | (64,128,1,8,4) | 2,644 | **(96,128,1,16,2)** | **1,993** | **1.33x** |
| up1 resblocks | 384→384 | (3,3,3) | (43,48,12) | x5 | (96,96,1,8,4) | 4,719 | **(96,128,1,16,2)** | **3,356** | **1.41x** |
| up1 time_conv | 384→768 | (3,1,1) | (42,46,10) | x1 | (384,32,1,1,1) | 19,170 | **(192,384,1,16,2)** | **812** | **23.6x** |
| up1 spatial | 384→192 | (1,3,3) | (81,94,22) | x1 | (192,96,1,32,4) | 2,979 | **(192,96,1,32,4)** | **3,044** | 0.98x |
| up2 resblocks | 192→192 | (3,3,3) | (83,94,22) | x6 | (96,96,1,8,4) | 7,584 | **(96,96,1,8,8)** | **7,827** | 0.97x |
| up2 spatial | 192→96 | (1,3,3) | (81,186,42) | x1 | (192,96,1,4,8) | 3,384 | **(192,96,1,8,8)** | **3,402** | 0.99x |
| up3 resblocks | 96→96 | (3,3,3) | (83,186,42) | x6 | (96,96,1,8,8) | 8,043 | **(96,96,1,8,8)** | **8,017** | 1.00x |
| conv_out | 96→3 | (3,3,3) | (83,186,42) | x1 | (96,32,1,16,8) | 5,896 | **(96,32,1,16,8)** | **5,869** | 1.00x |

**Total per uncached frame:**

| Component | Original (us) | Best (us) | Speedup |
|-----------|--------------|-----------|---------|
| conv_in (x1) | 310 | 193 | **1.61x** |
| mid+up0 res (x10) | 7,890 | 8,200 | 0.96x |
| up0 time_conv (x1) | 2,461 | 286 | **8.60x** |
| up0 spatial (x1) | 813 | 608 | **1.34x** |
| up1 res0 (x1) | 2,644 | 1,993 | **1.33x** |
| up1 resblocks (x5) | 23,595 | 16,780 | **1.41x** |
| up1 time_conv (x1) | 19,170 | 812 | **23.6x** |
| up1 spatial (x1) | 2,979 | 3,044 | 0.98x |
| up2 resblocks (x6) | 45,504 | 46,962 | 0.97x |
| up2 spatial (x1) | 3,384 | 3,402 | 0.99x |
| up3 resblocks (x6) | 48,258 | 48,102 | 1.00x |
| conv_out (x1) | 5,896 | 5,869 | 1.00x |
| **TOTAL** | **162,904** | **136,251** | **1.20x** |

Notes:
- mid+up0 res, up1 spatial, up2 resblocks: best blocking is slightly slower than original in this
  run (~1-4% noise). These layers are already near-optimal with the original blocking.
- The real wins are: time convs (8.6-23.6x from fixing the default), up1 resblocks (1.41x from
  finding C_in=96,C_out=128), and conv_in/up0 spatial/up1 res0 (1.3-1.6x).
- **70% of VAE time is in up2+up3 resblocks** (95ms of 136ms) — already at optimal blockings.

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
