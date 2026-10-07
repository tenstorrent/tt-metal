# t200: 6-step S1 that keeps the high-sigma head, stacked on the S2x2 default

## Setup (blx01, all under /var/tmp/fasth3/t200)
- Python: overlay tree at t48 b21f12b93a2 (S2 2-step default; t48 head ba4682afbd0 only changes a README), built by
  mkoverlay.sh from t195's files/. C++ build + JIT cache /var/tmp/fasth3/t48 (bf7db12a149). Same stack as ref_t48_s2x2 (job 764).
- Default S1 (8 steps): 1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0
- Arm s1x6b: LTX_S1_SIGMAS=1.0,0.99375,0.9875,0.975,0.909375,0.725,0.0 (drops 0.98125 and 0.421875).
  #186's rejected s1x6 dropped 0.99375 and 0.98125 instead.
- One broker job: run_cfg.sh s1x6b LTX_FRESH_PROMPTS=0 LTX_E2E_SEEDS=0,1,2,3,4 LTX_E2E_EXTRA_REPLAYS=0, -t 240,
  PYTEST_S=220, -e /var/tmp/fasth3/t159/env.yaml (job 764 took 156.7 s).
- driver.sh (t186's): waits for blx01 ready (broker RUNNING section, not HELD, no smarton job; 2 passes), submits
  under `ttp lock blx01-device`, reruns a dropped job once, skips after 2 drops, then post.sh.
- post.sh -> tt-project/data/g15/t200_s1x6b/: seed<N>.mp4, vis/ (seed0 t=2.6-3.4 s at 10 fps, ref and cand;
  1 fps ref|cand strips per seed), sbs/, eval_vs_ref_t48_s2x2/ (PCC/PSNR + VBench 5 dims), POST.done.
- Marker: t200/DRIVER.done = "<rc> <reason>".

## Attempt 1 (2026-10-07)
- Driver detached 06:39 UTC as run 802 t200drv (state/runs/802/t200drv.{log,rc}); blx01 was busy with ltx-host job 765.
- Next: when t200drv.rc exists, read driver.log (job id, drops), DRIVER.done, data/g15/t200_s1x6b/POST.done,
  E2E_WALL_S gen#1..5 from run.log, eval summary, then look at vis/ (seed0 t=2.6-3.4 first), write verdict.
- Light wake: job 766 completed exit=0 (06:40-06:43 UTC, no drops), but post.sh rc=1 with empty post_s1x6b.log. Needs debugging of post.sh (standard tier).
- Standard wake: post.sh did not break. ltx_eval exits 1 on BATCH FAIL and set -e turned that into POST.done "1 score",
  but every step finished and the scores were written. post.sh now records the eval rc and goes on.

## Verdict (2026-10-07): s1x6b REJECTED as a default, stays opt-in via LTX_S1_SIGMAS
- Job 766, blx01, 06:40-06:43 UTC, exit 0, no drops. Mean warm E2E over seeds 0-4 is 4.327 s
  (4.318/4.330/4.291/4.361/4.337). ref_t48_s2x2 (job 764, same stack) is 4.794 s, so -0.467 s.
- vs ref_t48_s2x2: PCC 0.605 mean (worst frame 0.204), PSNR 17.1 dB mean (min 13.6). VBench cand/ref: subject 0.895/0.888,
  background 0.927/0.924, imaging 0.550/0.553, aesthetic 0.555/0.590 (-0.036, worse than #186's -0.011/-0.029),
  motion 0.984/0.984.
- Visuals: seed 0 at t=3.0 s shows the same face smear #186 found (doubled mouth, smeared eyes):
  data/g15/t200_s1x6b/vis/seed0_cand_t3.0_face.png, 3x3 grid vis/seed0_cand_t2.6-3.4.png (ref: seed0_ref_t2.6-3.4.png).
  Seed 1 looks clean at 1 fps. The bar is all 5 seeds clean, and seed 0 fails it, so the other strips were not scored in detail.
- Reading: dropping 0.421875 (the last low-sigma step) seems to cause the t=3 s smear as well. #186 kept 0.421875 but dropped
  0.99375/0.98125 and smeared too. So that seed-0 smear shows up with any 6-step S1 tried so far.
  Running the 8-step S1 with S2x2 stays the default.
