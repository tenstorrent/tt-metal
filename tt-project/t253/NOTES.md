# t253: DiffVAE next lever (profile stage 5, then one opt-in lever)

## Profile (blx01 job 912, 2026-10-08 03:05 UTC, src be5d1bc045a on t238/b build, no drop)
Real decode 3.382 / 3.375 s (mean 3.379 s), md5 seed0 13802b012e19cf652be8d9c88cdd9316.
Deep tree (sync-inflated, 3820 ms): det stages 1409 ms (stage 0 dim 2048: 607, of which
linear-order attention 288), stage 5 2410 ms (8 blocks x ~277 ms + pre 64 + tail pull 99).
Per stage-5 block: neighborhood-sdpa 152.7 (55% of stage 5), halo+brick (k,v) 42.8
(CCL halo 2x10.4), qkv-lanes slice+norm+rope 27.7, mlp 25.7, qkv-proj 5.7, context 4.9,
out-proj 2.8, residual 2.8, norm+mod 5.3. No conv3d/groupnorm in stage 5.
Output: blx01 /var/tmp/fasth3/t253/out_P/stage_tree_def.txt.

## Lever: DIFFVAE_NA_BF8=1 (code 47e19132826, opt-in)
NA reads per chip per block: 9308 query chunks x 4 heads x (112 K + 112 V bricks x 2 tiles x 2 KB)
= ~34 GB -> 223 GB/s at 152.7 ms, i.e. at Wormhole DRAM bandwidth. tiles_per_kv_chunk is already 8
(DST max). bfloat8_b Q/K/V halves the reads; output cast back to bf16. Python only, no rebuild.

## A/B (running)
driver: blx01 /var/tmp/fasth3/t253/drv/driver253.sh AB 47e19132826 "def bf8:DIFFVAE_NA_BF8=1" "bf8" 330
marker: /var/tmp/fasth3/t253/drv/driver_AB.marker; out: /var/tmp/fasth3/t253/out_AB
(cmp_<arm>.json vs ref_dvx, stage_tree_bf8.txt). Accept if PCC >= 0.9999 and PSNR within ~0.5 dB of 55.
Next step on wake: read marker + driver.log + cmp jsons; if accepted flip default and land on t48.
