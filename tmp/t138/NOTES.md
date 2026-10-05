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
