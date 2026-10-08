# t261 notes: R2 DiffVAE det stages on all 32 chips

## Code
- 3c4521837d5 `DIFFVAE_DET_S1_SPLIT=1` (opt-in): stage 1 on the W-sharded bricked executor with
  the axes swapped. W=60 is split over axis 0 (15 per chip, brick (t,h,1)). The 32 heads are split
  over axis 1 (4 per chip, colpar qkv). The band is gathered back over axis 0 before upsample 1,
  and stages 2-4 run unchanged. parameter_layout gets an `s1x-` prefix, so it uses its own
  weight-cache folder (about 808 MB on blx01: /var/tmp/fasth3/cache/dit-ltx25/.../diffvae/det-s1x-...).
- Test: test_decode_full_bricked_matches_replicated[tp4_s1split] was added but not run on the device.

## Findings (code read)
- 4-axis replication today:
  - Stage 1 runs fully replicated on 32 chips. Only the linear-order executor shards query tiles
    and all-gathers them.
  - In stages 2-4, only colpar qkv and the NA op split heads over axis 0. Norms, out-proj, MLP
    (tp_mlp off), upsample and head-unflatten run 4x replicated.
- The spec's S5_2D split (H/4 x W/8) for stages 2-4 does not work with 32-site bricks:
  - Stage 2 and 3: H/4=17 and W/8=15 are odd, so only a (32,1,1) brick fits. Stage 2 T=21<32 has
    no valid brick, and stage 3 would waste most of each brick.
  - Stage 1: 34/4 and 60/8 do not divide.

## Device A/B (blx01)
- Driver /var/tmp/fasth3/t261/drv/driver261.sh AB, broker job 931, -t 480.
- Arms (one process each): def, and s1x (DIFFVAE_DET_S1_SPLIT=1). Both are profiled with the deep tree.
- Log: /var/tmp/fasth3/t261/drv/driver.log. Marker: /var/tmp/fasth3/t261/drv/driver_AB.marker.
- Out: /var/tmp/fasth3/t261/out_AB/{run.log, stage_tree_*.txt, cmp_*.json}.
- Baseline (job 912): decode 3.379 s; det tree 1408.7 ms, of which stage 1 is 606.6 ms.

## Next
On wake: read driver.log and run.log. Check DECODE_MEAN_S per arm, the PCC/PSNR lines (cmp vs
refs) and stage 1 in stage_tree_s1x.txt.
Accept: PCC >= 0.9999, PSNR within 0.5 dB, decode -0.7 s (unlikely from stage 1 alone; expect
about -0.4 s).
If it is a clear gain with neutral quality: make it the default, land it via a cherry-pick branch
with `ttp push --detach`, and remove the old cache if it is unused.
