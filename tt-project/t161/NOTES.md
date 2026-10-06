# t161: FastH3 (fl2v) numbers (2026-10-06)

## Step 1 (done, CPU only)
Project's only H3 e2e measurement (no 10 s run was ever made; nothing optimized was timed):
- fl2va, 1344x768 native + host bicubic upscale to 1920x1080, 158 frames @24 fps (6.58 s), dense, 50 steps (49 fwd), seed 0
- g15blx02 4x8 (tp4/sp8), 2026-09-30 ~09:55 UTC, commit 3712e859da9 (fasth3-opt), rerun of broker job 544, warm
- encoder 1.73, keyframe enc 0.55, denoise 75.10 (1.53 s/fwd), VAE 4.75, audio 8.19, upscale 4.74; compute 95.06; generate wall 120.7; call 125.4
- source: tt-project/baselines/fl2va_768p_6s_dense_50steps/seed0_timings.json (+ seed0_1920x1080.mp4, seed0_1920x1080_mid.png)

Org numbers (Glean, "H3 Video Generation Speed Tracker", Ben Hall, updated 10-02/10-06), all 1344x768, 1x BH Galaxy:
- H3 base 50-step FL2VA first kf: 5 s 50.5, 10 s 116.5, 15 s 229.9 s; first+last: 53.0 / 135.6 / 240.4 s
- lightx2v 4-step FL2VA first kf: 5 s 8.0, 10 s 15.9, 15 s 25.6 s; first+last: 9.1 / 16.0 / 26.6 s
- 4x BH Galaxy 4-step FL2VA first kf: 5 s 6.8, 10 s 9.6 s
- PR #57745 (4-step Turbo, t2va, 1x 4x8 BH): 5 s 4.57 s, 10 s 10.77 s, 15 s 19.36 s (denoise 3.49 / 9.00 / 16.81)

## Step 2 (waiting)
No state/ready/*.READY marker exists; blx03 broker HELD (degraded) at 2026-10-06 13:10 PT.
Blockers to plan around:
- H3 dit cache (131 GB) was deleted by #35/#116. g15blx02 /home cap (100 GB project) forbids rebuilding it there;
  on blx01/blx03 it must go to /var/tmp/fasth3/h3-cache (cold fill took >500 s on 09-30, several broker jobs).
- Weights: /mnt/MLPerf/tt-shield/persistent-volume/volume_id_tt_transformers-MiniMax-H3-v0.22.0/weights/MiniMax-H3 (present on blx03).
- Harness: models/tt_dit/tests/models/minimax_h3/test_fasth3_baseline_minimax_h3.py (BASE_SECONDS, BASE_HEIGHT/WIDTH,
  BASE_UPSCALE, BASE_STEPS, BASE_WARM_STEPS=0 for jobs that cannot fit warm+timed).
- 10 s native 1080p dense 50-step is ~3x the tokens of the 09-30 run; estimate 6-10 min denoise alone and possible OOM.
Next: once a box is ready, run 6 s (768p+upscale, as 09-30) first, then 10 s, each its own job through that box's queue.
