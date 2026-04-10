# Conv3D Production Shapes — WAN 2.2 VAE Decoder (Uncached, 81 Frames)

Traced from actual decoder runs on BH p150b Loud Box (1x1 mesh, simulating per-device dims).

## Temporal Dimension Flow (Same for All Mesh Configs)

| Stage | T | Notes |
|-------|---|-------|
| Latent input | 21 | (81-1)//4 + 1 |
| After +2 causal pad (3x3x3) | 23 | conv_in, mid, up0 resblocks |
| Time conv up0 input | 22 | frames[1:] = 20, +2 causal pad |
| After upsample3d | 41 | 1 + 2*20 = 41 |
| After +2 causal pad | 43 | up1 resblocks |
| Time conv up1 input | 42 | frames[1:] = 40, +2 causal pad |
| After upsample3d | 81 | 1 + 2*40 = 81 |
| After +2 causal pad | 83 | up2, up3 resblocks, conv_out |

## Unique Conv3D Shapes by Mesh Config

**Note:** H and W values are per-device dims WITHOUT spatial padding. The conv3d op applies `int_pad=(0,1,1)` internally for (3,3,3) and (1,3,3) kernels, and `int_pad=(0,0,0)` for (3,1,1) kernels.

### bh_4x32 720p (h_factor=4, w_factor=32)

Per-device: latent H=23, W=5 | mid H=46, W=10 | hi H=92, W=20 | full H=184, W=40

| # | Layer | C_in | C_out | Kernel | T | H | W | int_pad | Count |
|---|-------|------|-------|--------|---|---|---|---------|-------|
| 1 | conv_in | 32 | 384 | (3,3,3) | 23 | 23 | 5 | (0,1,1) | 1 |
| 2 | mid + up0 resblocks | 384 | 384 | (3,3,3) | 23 | 23 | 5 | (0,1,1) | 10 |
| 3 | up0 time_conv | 384 | 768 | (3,1,1) | 22 | 23 | 5 | (0,0,0) | 1 |
| 4 | up0 spatial_conv | 384 | 192 | (1,3,3) | 41 | 46 | 10 | (0,1,1) | 1 |
| 5 | up1 res0.conv1 | 192 | 384 | (3,3,3) | 43 | 46 | 10 | (0,1,1) | 1 |
| 6 | up1 resblocks | 384 | 384 | (3,3,3) | 43 | 46 | 10 | (0,1,1) | 5 |
| 7 | up1 time_conv | 384 | 768 | (3,1,1) | 42 | 46 | 10 | (0,0,0) | 1 |
| 8 | up1 spatial_conv | 384 | 192 | (1,3,3) | 81 | 92 | 20 | (0,1,1) | 1 |
| 9 | up2 resblocks | 192 | 192 | (3,3,3) | 83 | 92 | 20 | (0,1,1) | 6 |
| 10 | up2 spatial_conv | 192 | 96 | (1,3,3) | 81 | 184 | 40 | (0,1,1) | 1 |
| 11 | up3 resblocks | 96 | 96 | (3,3,3) | 83 | 184 | 40 | (0,1,1) | 6 |
| 12 | conv_out | 96 | 3 | (3,3,3) | 83 | 184 | 40 | (0,1,1) | 1 |

### bh_4x8 720p (h_factor=4, w_factor=8)

Per-device: latent H=23, W=20 | mid H=46, W=40 | hi H=92, W=80 | full H=184, W=160

| # | Layer | C_in | C_out | Kernel | T | H | W | int_pad | Count |
|---|-------|------|-------|--------|---|---|---|---------|-------|
| 1 | conv_in | 32 | 384 | (3,3,3) | 23 | 23 | 20 | (0,1,1) | 1 |
| 2 | mid + up0 resblocks | 384 | 384 | (3,3,3) | 23 | 23 | 20 | (0,1,1) | 10 |
| 3 | up0 time_conv | 384 | 768 | (3,1,1) | 22 | 23 | 20 | (0,0,0) | 1 |
| 4 | up0 spatial_conv | 384 | 192 | (1,3,3) | 41 | 46 | 40 | (0,1,1) | 1 |
| 5 | up1 res0.conv1 | 192 | 384 | (3,3,3) | 43 | 46 | 40 | (0,1,1) | 1 |
| 6 | up1 resblocks | 384 | 384 | (3,3,3) | 43 | 46 | 40 | (0,1,1) | 5 |
| 7 | up1 time_conv | 384 | 768 | (3,1,1) | 42 | 46 | 40 | (0,0,0) | 1 |
| 8 | up1 spatial_conv | 384 | 192 | (1,3,3) | 81 | 92 | 80 | (0,1,1) | 1 |
| 9 | up2 resblocks | 192 | 192 | (3,3,3) | 83 | 92 | 80 | (0,1,1) | 6 |
| 10 | up2 spatial_conv | 192 | 96 | (1,3,3) | 81 | 184 | 160 | (0,1,1) | 1 |
| 11 | up3 resblocks | 96 | 96 | (3,3,3) | 83 | 184 | 160 | (0,1,1) | 6 |
| 12 | conv_out | 96 | 3 | (3,3,3) | 83 | 184 | 160 | (0,1,1) | 1 |

### bh_4x32 480p (h_factor=4, w_factor=32)

Per-device: latent H=15, W=3 | mid H=30, W=6 | hi H=60, W=12 | full H=120, W=24

**Note:** W0 = (latent_W // w_factor) * 8 = (104 // 32) * 8 = 3 * 8 = 24. NOT W_out // w_factor = 832 // 32 = 26.

| # | Layer | C_in | C_out | Kernel | T | H | W | int_pad | Count |
|---|-------|------|-------|--------|---|---|---|---------|-------|
| 1 | conv_in | 32 | 384 | (3,3,3) | 23 | 15 | 3 | (0,1,1) | 1 |
| 2 | mid + up0 resblocks | 384 | 384 | (3,3,3) | 23 | 15 | 3 | (0,1,1) | 10 |
| 3 | up0 time_conv | 384 | 768 | (3,1,1) | 22 | 15 | 3 | (0,0,0) | 1 |
| 4 | up0 spatial_conv | 384 | 192 | (1,3,3) | 41 | 30 | 6 | (0,1,1) | 1 |
| 5 | up1 res0.conv1 | 192 | 384 | (3,3,3) | 43 | 30 | 6 | (0,1,1) | 1 |
| 6 | up1 resblocks | 384 | 384 | (3,3,3) | 43 | 30 | 6 | (0,1,1) | 5 |
| 7 | up1 time_conv | 384 | 768 | (3,1,1) | 42 | 30 | 6 | (0,0,0) | 1 |
| 8 | up1 spatial_conv | 384 | 192 | (1,3,3) | 81 | 60 | 12 | (0,1,1) | 1 |
| 9 | up2 resblocks | 192 | 192 | (3,3,3) | 83 | 60 | 12 | (0,1,1) | 6 |
| 10 | up2 spatial_conv | 192 | 96 | (1,3,3) | 81 | 120 | 24 | (0,1,1) | 1 |
| 11 | up3 resblocks | 96 | 96 | (3,3,3) | 83 | 120 | 24 | (0,1,1) | 6 |
| 12 | conv_out | 96 | 3 | (3,3,3) | 83 | 120 | 24 | (0,1,1) | 1 |

### bh_4x8 480p (h_factor=4, w_factor=8)

Per-device: latent H=15, W=13 | mid H=30, W=26 | hi H=60, W=52 | full H=120, W=104

| # | Layer | C_in | C_out | Kernel | T | H | W | int_pad | Count |
|---|-------|------|-------|--------|---|---|---|---------|-------|
| 1 | conv_in | 32 | 384 | (3,3,3) | 23 | 15 | 13 | (0,1,1) | 1 |
| 2 | mid + up0 resblocks | 384 | 384 | (3,3,3) | 23 | 15 | 13 | (0,1,1) | 10 |
| 3 | up0 time_conv | 384 | 768 | (3,1,1) | 22 | 15 | 13 | (0,0,0) | 1 |
| 4 | up0 spatial_conv | 384 | 192 | (1,3,3) | 41 | 30 | 26 | (0,1,1) | 1 |
| 5 | up1 res0.conv1 | 192 | 384 | (3,3,3) | 43 | 30 | 26 | (0,1,1) | 1 |
| 6 | up1 resblocks | 384 | 384 | (3,3,3) | 43 | 30 | 26 | (0,1,1) | 5 |
| 7 | up1 time_conv | 384 | 768 | (3,1,1) | 42 | 30 | 26 | (0,0,0) | 1 |
| 8 | up1 spatial_conv | 384 | 192 | (1,3,3) | 81 | 60 | 52 | (0,1,1) | 1 |
| 9 | up2 resblocks | 192 | 192 | (3,3,3) | 83 | 60 | 52 | (0,1,1) | 6 |
| 10 | up2 spatial_conv | 192 | 96 | (1,3,3) | 81 | 120 | 104 | (0,1,1) | 1 |
| 11 | up3 resblocks | 96 | 96 | (3,3,3) | 83 | 120 | 104 | (0,1,1) | 6 |
| 12 | conv_out | 96 | 3 | (3,3,3) | 83 | 120 | 104 | (0,1,1) | 1 |

### bh_2x4 480p cached (h_factor=2, w_factor=4, t_chunk_size=1)

Per-device: latent H=30, W=26 | mid H=60, W=52 | hi H=120, W=104 | full H=240, W=208

Cached T from compute_decoder_dims(480, 832, 2, 4, t_chunk_size=1, cached=True).
In cached mode, WanResample does NOT split off frame-0. cur_T grows across stages: 1 → 2 → 4.
  T_res = cur_T + 2  |  T_tconv = cur_T + 2  |  T_spatial = 2 * cur_T

| # | Layer | C_in | C_out | Kernel | T | H | W | int_pad | Count | Stage (cur_T) |
|---|-------|------|-------|--------|---|---|---|---------|-------|---------------|
| 1 | conv_in | 32 | 384 | (3,3,3) | 3 | 30 | 26 | (0,1,1) | 1 | 0 (1) |
| 2 | mid + up0 resblocks | 384 | 384 | (3,3,3) | 3 | 30 | 26 | (0,1,1) | 10 | 0 (1) |
| 3 | up0 time_conv | 384 | 768 | (3,1,1) | 3 | 30 | 26 | (0,0,0) | 1 | 0 (1) |
| 4 | up0 spatial_conv | 384 | 192 | (1,3,3) | 2 | 60 | 52 | (0,1,1) | 1 | 0→1 |
| 5 | up1 res0.conv1 | 192 | 384 | (3,3,3) | 4 | 60 | 52 | (0,1,1) | 1 | 1 (2) |
| 6 | up1 resblocks | 384 | 384 | (3,3,3) | 4 | 60 | 52 | (0,1,1) | 5 | 1 (2) |
| 7 | up1 time_conv | 384 | 768 | (3,1,1) | 4 | 60 | 52 | (0,0,0) | 1 | 1 (2) |
| 8 | up1 spatial_conv | 384 | 192 | (1,3,3) | 4 | 120 | 104 | (0,1,1) | 1 | 1→2 |
| 9 | up2 resblocks | 192 | 192 | (3,3,3) | 6 | 120 | 104 | (0,1,1) | 6 | 2 (4) |
| 10 | up2 spatial_conv | 192 | 96 | (1,3,3) | 4 | 240 | 208 | (0,1,1) | 1 | 2 |
| 11 | up3 resblocks | 96 | 96 | (3,3,3) | 6 | 240 | 208 | (0,1,1) | 6 | 3 (4) |
| 12 | conv_out | 96 | 3 | (3,3,3) | 6 | 240 | 208 | (0,1,1) | 1 | 3 (4) |

### bh_2x4 480p cached (h_factor=2, w_factor=4, t_chunk_size=7)

Per-device: latent H=30, W=26 | mid H=60, W=52 | hi H=120, W=104 | full H=240, W=208

Cached T from compute_decoder_dims(480, 832, 2, 4, t_chunk_size=7, cached=True).
In cached mode, WanResample does NOT split off frame-0. cur_T grows across stages: 7 → 14 → 28.
  T_res = cur_T + 2  |  T_tconv = cur_T + 2  |  T_spatial = 2 * cur_T

| # | Layer | C_in | C_out | Kernel | T | H | W | int_pad | Count | Stage (cur_T) |
|---|-------|------|-------|--------|---|---|---|---------|-------|---------------|
| 1 | conv_in | 32 | 384 | (3,3,3) | 9 | 30 | 26 | (0,1,1) | 1 | 0 (7) |
| 2 | mid + up0 resblocks | 384 | 384 | (3,3,3) | 9 | 30 | 26 | (0,1,1) | 10 | 0 (7) |
| 3 | up0 time_conv | 384 | 768 | (3,1,1) | 9 | 30 | 26 | (0,0,0) | 1 | 0 (7) |
| 4 | up0 spatial_conv | 384 | 192 | (1,3,3) | 14 | 60 | 52 | (0,1,1) | 1 | 0→1 |
| 5 | up1 res0.conv1 | 192 | 384 | (3,3,3) | 16 | 60 | 52 | (0,1,1) | 1 | 1 (14) |
| 6 | up1 resblocks | 384 | 384 | (3,3,3) | 16 | 60 | 52 | (0,1,1) | 5 | 1 (14) |
| 7 | up1 time_conv | 384 | 768 | (3,1,1) | 16 | 60 | 52 | (0,0,0) | 1 | 1 (14) |
| 8 | up1 spatial_conv | 384 | 192 | (1,3,3) | 28 | 120 | 104 | (0,1,1) | 1 | 1→2 |
| 9 | up2 resblocks | 192 | 192 | (3,3,3) | 30 | 120 | 104 | (0,1,1) | 6 | 2 (28) |
| 10 | up2 spatial_conv | 192 | 96 | (1,3,3) | 28 | 240 | 208 | (0,1,1) | 1 | 2 |
| 11 | up3 resblocks | 96 | 96 | (3,3,3) | 30 | 240 | 208 | (0,1,1) | 6 | 3 (28) |
| 12 | conv_out | 96 | 3 | (3,3,3) | 30 | 240 | 208 | (0,1,1) | 1 | 3 (28) |

## Comparison with Old Sweep

The old sweep had `T0_uncached=4` which gave these incorrect T values:
- Latent 3x3x3: T=3 (should be 23)
- Time conv up0: T=3 (should be 22)
- Mid 3x3x3: T=3 (should be 43)
- Time conv up1: T=4 (should be 42)
- Output 3x3x3: T=3 (should be 83)

The old sweep also added +2 to H and W for spatial padding (e.g., H0//8+2), but the actual conv3d call uses the raw dims with `int_pad=(0,1,1)` handling padding internally. When the sweep creates tensors directly and calls conv3d with `padding=(0,0,0)`, it must include the +2. When using the model's forward path, the int_pad handles it.

## Cached Mode (any t_chunk_size with vae_t_chunk_size != None)

In cached mode WanResample does NOT split off frame-0, so all frames go through time_conv.
Formulas: T_res = cur_T + 2, T_tconv = cur_T + 2, T_spatial = 2 * cur_T.
cur_T grows across stages because temporal doubling is not reduced by the frame-0 split.

Compare with uncached (vae_t_chunk_size=None): frame-0 split gives T_tconv=cur_T+1, T_spatial=2*(cur_T-1)+1.
Spatial dims are the same in both modes. Only T differs.

Blocking is optimized for the steady-state cached path (majority of iterations).
The first chunk (uncached) sees slightly smaller T values — this is safe because
smaller T means less data and always fits within the L1 budget of the cached blocking.

### t_chunk_size=1 (BH 2x4 480p)
  stage 0: cur_T=1 → T_res=3, T_tconv=3, T_spatial=2
  stage 1: cur_T=2 → T_res=4, T_tconv=4, T_spatial=4
  stage 2: cur_T=4 → T_res=6, T_spatial=4
  stage 3: cur_T=4 → T_res=6

### t_chunk_size=7 (BH 2x4 480p)
  stage 0: cur_T=7 → T_res=9, T_tconv=9, T_spatial=14
  stage 1: cur_T=14 → T_res=16, T_tconv=16, T_spatial=28
  stage 2: cur_T=28 → T_res=30, T_spatial=28
  stage 3: cur_T=28 → T_res=30

### t_chunk_size=11 (Galaxy 4x8 720p) — TODO: needs sweep
  stage 0: cur_T=11 → T_res=13, T_tconv=13, T_spatial=22
  stage 1: cur_T=22 → T_res=24, T_tconv=24, T_spatial=44
  stage 2: cur_T=44 → T_res=46, T_spatial=44
  stage 3: cur_T=44 → T_res=46
