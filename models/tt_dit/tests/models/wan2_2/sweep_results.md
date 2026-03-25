# Conv3D Blocking Sweep Results — BH p150b LB

**Machine:** BH Loud Box, 8x p150b
**Production:** WAN 2.2 T2V, 81 output frames, latent T=21
**Method:** Brute-force exhaustive search, then head-to-head verification.
New blocking adopted only if >3% faster than original in head-to-head.

---

## BH Galaxy 6U (bh_4x32) 720p — h_factor=4, w_factor=32 — 12x10 grid

Head-to-head verified. Original and winner measured back-to-back, same device, same run.

| Layer | C_in→C_out | Kernel | (T,H,W) | Cnt | Original | Orig us | Winner | Win us | Speedup |
|-------|-----------|--------|---------|-----|----------|---------|--------|--------|---------|
| conv_in | 32→384 | (3,3,3) | (23,25,7) | x1 | (32,96,1,2,32) | 329 | **(32,128,1,16,2)** | **194** | **1.69x** |
| mid+up0 res | 384→384 | (3,3,3) | (23,25,7) | x10 | (96,96,1,8,4) | 816 | (96,96,1,8,4) | 816 | keep |
| up0 time_conv | 384→768 | (3,1,1) | (22,23,5) | x1 | (384,32,1,1,1) | 2,469 | **(128,256,1,8,4)** | **276** | **8.93x** |
| up0 spatial | 384→192 | (1,3,3) | (41,48,12) | x1 | (192,96,1,32,4) | 875 | **(192,96,1,16,2)** | **630** | **1.39x** |
| up1 res0 | 192→384 | (3,3,3) | (43,48,12) | x1 | (64,128,1,8,4) | 2,639 | **(96,128,1,16,2)** | **2,005** | **1.32x** |
| up1 resblocks | 384→384 | (3,3,3) | (43,48,12) | x5 | (96,96,1,8,4) | 4,749 | **(96,128,1,16,2)** | **3,439** | **1.38x** |
| up1 time_conv | 384→768 | (3,1,1) | (42,46,10) | x1 | (384,32,1,1,1) | 19,742 | **(192,384,1,16,2)** | **811** | **24.3x** |
| up1 spatial | 384→192 | (1,3,3) | (81,94,22) | x1 | (192,96,1,32,4) | 3,103 | (192,96,1,32,4) | 3,103 | keep |
| up2 resblocks | 192→192 | (3,3,3) | (83,94,22) | x6 | (96,96,1,8,4) | 8,016 | (96,96,1,8,4) | 8,016 | keep |
| up2 spatial | 192→96 | (1,3,3) | (81,186,42) | x1 | (192,96,1,4,8) | 3,447 | (192,96,1,4,8) | 3,447 | keep |
| up3 resblocks | 96→96 | (3,3,3) | (83,186,42) | x6 | (96,96,1,8,8) | 8,136 | (96,96,1,8,8) | 8,136 | keep |
| conv_out | 96→3 | (3,3,3) | (83,186,42) | x1 | (96,32,1,16,8) | 5,967 | (96,32,1,16,8) | 5,967 | keep |

| | Original | Winner | |
|---|---------|--------|---|
| **Total** | **167,388 us** | **138,700 us** | **1.21x** |

6 layers changed, 6 kept original. 70% of VAE time is up2+up3 resblocks — already at optimal blockings.

---

## Remaining configs — brute-force sweep in progress

Results pending head-to-head verification before adoption.

| Config | h_factor | w_factor | Grid | Resolution | Status |
|--------|----------|----------|------|------------|--------|
| BH (4,8) Galaxy | 4 | 8 | 12x10 | 720p, 480p | sweeping |
| BH (2,4) LoudBox | 2 | 4 | 13x10 | 480p | sweeping |
| BH (2,2) QuietBox | 2 | 2 | 13x10 | 480p | sweeping |
| WH (4,8) Galaxy | 4 | 8 | 8x8 | 720p, 480p | needs WH hardware |
