# t185: S2 step cut 3->2 on LTX 2.5 (plan t-denoise task 1)

## Setup (blx01, all under /var/tmp/fasth3/t185)
- Python: t171 overlay tree (t48 f6b806516cc defaults), C++/JIT cache /var/tmp/fasth3/t48 @ bf7db12a149. Same as ref_t48_f6b8 (job 710).
- run_cfg.sh = t171's with T=t185 (output dir only). Config: LTX_S2_SIGMAS=0.909375,0.421875,0.0,
  LTX_FRESH_PROMPTS=0, LTX_E2E_SEEDS=0,1,2,3,4 (gen#N = seed N-1; gen#0 = cold capture of seed 0).
- Judgment: default warmup, NOT LTX_WARMUP_T2V_ONLY=1/LTX_WARMUP_ENCODERS=0 as the spec asked. That warmup
  config dropped tray 3 twice on blx01 (jobs 625, 640) and is skipped there per the drop rule; job 710
  used the default warmup, so the comparison is like for like. Job 710 took 160 s -> -t 240 (=+50%), PYTEST_S=220.

## Jobs
- blx01 broker job 715, submitted 2026-10-07 04:44:59 UTC, running at 04:46.
  Output: /var/tmp/fasth3/t185/res/s2x2/{run.log, ltx_av_fast_1920x1088_<g>.{mp4,json}}.

## Post (g15blx02, CPU)
- Detached waiter (run 744, t185post.{log,rc}): wait_post.sh 715 s2x2 -> post.sh s2x2 ->
  tt-project/data/g15/t185_s2x2/{seed<N>.mp4, sbs_seed<N>.mp4 (ref | cand), sbs_seed<N>_t3s.png,
  eval_vs_ref_t48_f6b8/ (PCC/PSNR + VBench 5 dims, ref scored too), POST.done}.
- Next: read per-seed E2E_WALL_S gen#1..5 from run.log, summary.json, look at the 5 sbs stills;
  if visuals degrade, second job with LTX_S2_SIGMAS=0.909375,0.725,0.0 (label s2x2b).
- DROP: blx01 job 715 killed by broker device recovery (exit -9, 72.6 s, ~04:46 UTC 2026-10-07; smarton's job; chips/tray: see /var/log/tt-device-broker/2026-10-07_044459_715.log). No outputs; fetch failed. Next: once blx01 health passes, resubmit s2x2 once (first drop of this config on blx01).

## Rerun (attempt 2, 2026-10-07 ~04:55 UTC)
- Drop detail (job 715): chips 16-23 (tray 3, broker calls it tray 4) left PCIe at 04:46:11 UTC; bridge-reset
  failed (jobs 717/718), glx_reset health gate failed (719); by 04:52 blx01 was unreachable by ssh
  (broker power cycle). Same tray as jobs 625/640.
- Driver: tt-project/t185/driver.sh = t186's driver with S2 sigmas, -e t159/env.yaml, DROPS0=1 (715 counted),
  so a second drop of s2x2 skips it on blx01. Waits for blx01 ready (2 passes), submits under
  `ttp lock blx01-device` (t186's S1 arms wait for the same lock, so jobs stay one at a time), then post.sh s2x2.
- Detached on g15blx02: run 750 t185drv.{log,rc}; driver log t185/driver.log; marker t185/DRIVER.done.
- Next on wake: DRIVER.done rc 0 -> read data/g15/t185_s2x2 (run.log E2E_WALL_S gen#1..5,
  eval_vs_ref_t48_f6b8/, sbs stills). rc 86 -> dropped twice on blx01: move to blx03 runner.
