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

## Drops
- 2026-10-08 04:46:45 UTC, blx01, broker job 931 (smarton t261 AB), chips 24-31 (one tray) dropped; job killed by
  device recovery. Driver died waiting on recovery (no marker). Partial out: /var/tmp/fasth3/t261/out_AB_drop1.
- (not ours) 2026-10-08 09:45:03 UTC blx01 chips 16-23 dropped, no job; recovered 09:55.

## Resubmit (run 1080)
- 2026-10-08 10:49:57 UTC: blx01 broker job 000 (ID counter wrapped past 999), same command as 931, submitted
  directly with run-bg (no driver), -t 480. Out: /var/tmp/fasth3/t261/out_AB/run.log.
- Scoring is NOT automatic now: on wake run cmp241.py per arm vs $F/diffvae/ref, plus s1x vs def and md5s
  (see the score block in drv/driver261.sh).

## Job 000 result (run 1082)
- def arm: decode 3.377/3.374 s (mean 3.376), md5 13802b01.../4a47f7a9... Out moved to out_AB_000.
- s1x arm crashed: `H shard 34 is not whole 32-site bricks` (assert from the S5_2D commit 4ee505fc applied
  on the 1-D path too). Fix 6b75bdfde1a: the assert applies only with an H axis.
- Resubmitted 2026-10-08 10:54:58 UTC as blx01 broker job 002, src 6b75bdfde1a, -t 480, env drv/env.yaml.

## Job 002 result (blx01, 10:54-10:58 UTC, build t238/b 34a571c5f47 = pre-K/V-ring C++)
- def 3.372 s, s1x 2.971 s (-0.40 s, -12%). Deep tree: det 1476 -> 1028 ms, stage 1 613 -> 178 ms.
- vs host-noise refs: def PCC 0.99995/0.99995 PSNR 55.04/54.59; s1x PCC 0.99995/0.99995 PSNR 54.71/54.28
  (-0.33/-0.31 dB, within 0.5 dB). s1x vs def PSNR 55.1/54.7. md5s differ between arms (valid A/B).
- Spec's -0.7 s / det <= 600 ms not reached; stages 2-4 S5_2D split is infeasible (see Findings).
  Judgment: land the stage-1 split as default anyway (12% gain, quality neutral).
- Default flip 3bcd80dc1b0. Landing branch <this>-land = origin/t48 b1ca9870f09 + d16bfe2f1f5,
  9d65f8ea669, 5833f56096f (cherry-picks of 3c4521837d5, 6b75bdfde1a, 3bcd80dc1b0).
- t48 now has the K/V ring C++ default (#260), untested with the stage-1 split. Validation job: arms
  off (DIFFVAE_DET_S1_SPLIT=0) vs def on build t272/b cae4b52657d (t48 C++ + hash-only fix), src
  5833f56096f, runner drv/run261r.sh, out /var/tmp/fasth3/t261/out_R.

## Drops
- 2026-10-08 ~10:59 UTC blx01 chips 16-23 (tray), broker post-job gate right after our job 002 (smarton t261),
  which had completed exit 0; bridge-reset jobs 005/006 failed, broker recovering.

## Next (run 1082 hand-off)
1. When blx01 is ready for tenants, submit ONE job:
   ssh blx01 "tt-device-mcp run-bg \"bash /var/tmp/fasth3/t261/drv/run261r.sh 5833f56096f 'off:DIFFVAE_DET_S1_SPLIT=0 def' /var/tmp/fasth3/t261/out_R 'def'\" -w /var/tmp/fasth3/t261 -e /var/tmp/fasth3/t261/drv/env.yaml -t 480"
   (each arm ~105 s process wall; check t272/b is still at cae4b52657d first: run261r.sh refuses otherwise.)
2. Score: cmp241.py per arm vs /var/tmp/fasth3/diffvae/ref, def vs off, md5s differ (see job 002 scoring).
   Expect off ~3.11 s (ring default), def ~2.7 s.
3. If def is faster and PCC >= 0.9999, PSNR within 0.5 dB: in this worktree
   `git switch ttp/t261-r2-diffvae-det-stages-1-4-on-all-32-chip-land` (local branch, head 5833f56096f),
   `ttp push --detach`, then switch back. If t48 moved, ttp push rebases.
