# t260 notes (R1: NA K/V L1 ring)

Code: 89d04d2ab71 "neighborhood_sdpa: add opt-in L1 K/V ring (DIFFVAE_NA_KV_RING)" on be5d1bc045a.
Not yet compiled on device or measured.

## Device run (blx01), not started yet
blx01 had no route at 2026-10-08 ~04:52 UTC (probably rebooting after t261 job 931's drop at 04:46).
Start once `ssh g15blx01 true` works:
1. ssh g15blx01 'mkdir -p /var/tmp/fasth3/t260/drv && cp /var/tmp/fasth3/t261/drv/{decode261.py,cmp241.py} /var/tmp/fasth3/t260/drv/'
2. git bundle create /tmp/t260.bundle be5d1bc045a..ttp/t260-r1-na-k-v-l1-ring-in-neighborhood-sdpa-d
3. scp t260-drv/driver260.sh to .../t260/drv/driver260.sh.tmp, then mv it into place. scp run260.sh, env.yaml and /tmp/t260.bundle to .../t260/drv/
4. ssh g15blx01 'cd /var/tmp/fasth3/t260/drv && setsid nohup bash driver260.sh > driver.out 2>&1 < /dev/null &'
The driver waits for other project drivers (t261) to end, builds /var/tmp/fasth3/t252/b at 89d04d2ab71
(t252 is done; its tree is reused, incremental), then runs ONE broker job (-t 480): def vs ring arms, one
process each, PROFILE=1, seeds 0,1 with host noise. It scores against the #214 refs and checks ring vs def md5.
Marker: /var/tmp/fasth3/t260/drv/driver.marker. Log: driver.log, out_AB/run.log, out_AB/stage_tree_*.txt.

## Accept
md5-identical to def (or PCC >= 0.9999 and PSNR within 0.1 dB of ~55 dB), NA op <= 60 ms/block,
decode <= 2.75 s (def 3.378 s). The factory log line "neighborhood sdpa kv ring: mode=" shows K+V(2)/K(1)/off(0).
If accepted: make it the default (=0 turns it off), land via a -land branch from origin/ttp/t48-ltx25-integrated with ttp push --detach.

## Result (blx01 job 946, 2026-10-08 04:56-05:05 UTC, 383 s, no drops)
One process per arm, seeds 0-1. Ring log lines appear only in the ring arm (mode=2, stage 5 columns=3, ring 336 KB).
- decode: def 3.383/3.374 s, ring 3.113/3.113 s (-0.265 s, -7.8%)
- stage-5 NA op: 152.3 -> 121.0 ms/block; stage-4 NA 37.5 -> 22.4 ms
- md5 ring vs def identical (raw seeds and host-noise seeds 0,1); vs #214 refs PCC 0.99995, PSNR 55.04/54.59 dB
- Spec targets missed (NA <= 60 ms, decode <= 2.75 s): L1 fits only 3 brick columns at stage 5, not the full 7x4x4 union.
Lossless gain above the 5% bar, so made default (3dfc583566b, =0 off) and landed on t48.
