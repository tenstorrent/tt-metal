# t24 notes (untracked)

Branch ttp/t24-... = t13 tip (eee3baf7c0d) + f56467d4bbc (opt-in background mp4 export), pushed.
Unit tests: models/tt_dit/tests/unit/test_ltx_export_latency.py 8/8 pass host-only
  (T7=../t7; TT_METAL_HOME=$T7 PYTHONPATH=$PWD:$T7/ttnn:$T7/tools python -m pytest -q -p no:cacheprovider <file>)

Attempt 1: broker job 600 TIMED OUT (600s cap) still JIT-compiling kernels: kernel sources resolve from
the t24 worktree path, so the shared t7 kernel cache was cold for them. Post-job health gate then saw chip 20
off the PCIe bus; broker recovered it itself (hands off).

Attempt 2 (14:06 UTC): tmp/prewarm.log = nohup prewarm_and_submit.sh -e tmp/env.yaml -t 600 -c -- bash tmp/job.sh
  stage 1 capture = broker job 621 (queued pos 6), stage 2 off-device compile, stage 3 = run-bg real A/B job
  (its ID is the last "Job N queued" line in tmp/prewarm.log). tmp/check_done.sh exits 0 when all is finished.
  env: t7 build, t13 env + EXTRA_REPLAYS=5, LTX_ASYNC_EXPORT=4, TT_METAL_KERNEL_PREWARM=1.
  gens 0..6: gen0 capture, gens 1-3 sync export (t13 behaviour), gens 4-6 background export.
  Read from the stage-3 job log: grep REQUEST_S (period between consecutive generate() returns = steady-state
  per-request), "Video export" lines, and the === MD5 === block (all 7 mp4s must match -> byte-identical).
  mp4s land in tmp/out/.
  If stage 1 aborted ("no manifest growth") or stage 3 timed out again: fall back to running from cwd=t7
  (kernels then resolve to t7 paths, which the cache already holds).

Next step: compare periods gens 2-3 (sync) vs 5-6 (async), check md5s, still frame from one async mp4
(ffmpeg -ss 3 -frames:v 1), delete the other mp4s in tmp/out, write result.json.

If the wrapper died with the session (prewarm.log has no "stage 2/3"/"stage 3/3" line) but job 621 finished:
  run stage 2 by hand: env TT_METAL_KERNEL_PREWARM=1 TT_METAL_CACHE=$T7/tmp/tt-metal-cache TT_METAL_HOME=$T7 $T7/build_Release/tools/kernel_prewarm
  then: tt-device-mcp run-bg "bash tmp/job.sh" -w $PWD -t 600 -e tmp/env.yaml

## DONE (attempt 3, 14:57 UTC): job 634 completed (253s, device healthy after)
REQUEST_S period: sync gens 1-3 = 6.493/6.461/6.481 (mean 6.478); async gens 4-6 = 6.279/6.318/6.317
(steady 5-6 mean 6.318) -> -0.16s/request. All 7 mp4 md5 = 13b1a12cb7ffff5448b3bea3c37f90ef.
Kept tmp/out/ltx_av_fast_1920x1088_6.mp4 + async_gen6_t3s.png; deleted the rest.
