# t138: user-approved 4x8 LTX-2.5 1080p/145f e2e, two gens back to back (blx03)

Scripts: tmp/t138/run_e2e.sh (the broker job), tmp/t138/driver.sh (detached on blx03, from the t78 template).
Config: t48 DEFAULT (baseline arm of t104; no 'all' arm, to keep the one 4x8 job short on a flaky box).
Tree: blx03 ~/fasth3/t48, built at 64571a953b2; the driver checks out c4409b1fa24 (test-only delta, no rebuild).
Env: SEED=0 (noise_seed 10000), RUN_WARMUP=1, LTX_TRACED=1, LTX_FRESH_PROMPTS=1, LTX_TIME_STAGES=1,
LTX_CONV3D_BLOCKING_MESH=4,8, caches /var/tmp/fasth3/cache. One pytest process: pipeline warmup, then
gen#0 (first run, trace capture inside) and gen#1 (warm replay, new prompt -> encoder on path) = headline.
Same protocol as job 879 (8.761 s), so gen#0/gen#1 compare against tt-project/baselines/t20/ltx_av_fast_1920x1088_{0,1}.mp4.
Note: there is no separate pixel upscale stage; the "upscale" is the latent upsample between S1 (half res)
and S2 (1088x1920); the output is native 1920x1088.

Output on blx03: /var/tmp/fasth3/t138/{driver.log,job_id,broker_slice.log,journal_slice.log,out/}
Done marker: grep T138_DRIVER_DONE /var/tmp/fasth3/t138/driver.log  (rc 9 = drop/reboot during OUR job -> stop all)

## 2026-10-05 20:11 UTC: submitted
blx03 rebooted at 20:06 (none of our jobs ran; we had not submitted). Startup gate passed (32 chips, fabric OK 20:09:22).
Broker job 099 (timeout 2400 s), queued behind ltx-host job 098. The first driver submitted 099 just before I
killed it to fix its health grep; the second driver was killed before it submitted. A watch-only driver
(WATCH_JOB=099) now polls 099, runs the post-job gate and writes T138_DRIVER_DONE.
Next: when T138_DRIVER_DONE lands, read driver.log (rc 9 = drop during our job: stop all, report with
broker_slice.log/journal_slice.log), then pull run.log timings, copy mp4s + stills back to tt-project/t-e2e/,
and run ltx_eval video --vbench none against tt-project/baselines/t20/ltx_av_fast_1920x1088_{0,1}.mp4.

## 2026-10-06 02:15 UTC: result (job 099 completed 2026-10-05 20:31:58, exit 0)
The watch driver died in a later blx03 reboot (boots 22:50, 23:23, 23:38, 00:00, 00:43 UTC, all after our job),
so T138_DRIVER_DONE was never written. Broker log: job 099 started 20:23:06, finished 20:31:58 status=completed,
post-job health gate OK (32/32 chips, ARC heartbeat). No drop during our job. pytest: 1 passed in 516 s.
The broker restarted at 20:22 and its startup gate (fabric OK 20:23:06) passed before 099 started.

Commit c4409b1fa24 (built 64571a953b2), t48 default config, bh_4x8sp1tp0_ring, 1088x1920, 145f @24fps, traced,
SEED=0 (noise_seed 10000), LTX_FRESH_PROMPTS=1, LTX_CONV3D_BLOCKING_MESH=4,8.

| stage            | gen#0 cold (s) | gen#1 warm (s) |
|------------------|---------------:|---------------:|
| text encode      | 0.27           | 0.20           |
| S1 denoise 8 st. | 15.63          | 2.25 (~0.277/step) |
| latent upsample  | 0.15           | 0.14           |
| S2 denoise 3 st. | 14.93          | 2.50 (~0.823/step) |
| VAE decode       | 0.79           | 0.56           |
| audio decode     | 0.90           | 0.39           |
| video export     | 0.17           | 0.17           |
| host gaps        | ~0.03          | ~0.03          |
| **E2E_WALL_S**   | **35.357**     | **6.238**      |

gen#0 is cold: step 1 of each stage captures the trace (S1 13.7 s, S2 13.2 s).
gen#1 vs job 879 (8.761 s, t20): -2.52 s (-29%). Under the 7 s target.

Quality vs tt-project/baselines/t20 (job 879, same prompts and seed), ltx_eval video --vbench none:
- gen#0: PCC 0.99857 (min 0.99769), PSNR 40.77 dB (min 39.50 @ f115). OK
- gen#1: PCC 0.99902 (min 0.99881), PSNR 39.32 dB (min 38.50 @ f63). OK
Stills look the same as the reference by eye (paper boat, t=3 s).

Media on g15blx02: tt-project/t-e2e/t138/ (mp4s, t3s stills, eval_gen{0,1}/ stills + reports, run.log.gz).
On blx03: /var/tmp/fasth3/t138/out/ (26 MB).
