# t10 next steps (written 2026-09-30)
Build: tmp/build.log, ends with BUILD_EXIT=<code>. First try died on a half-downloaded .cpmcache/nlohmann_json (log in tmp/build_fail1.log).
Broker caps jobs at 600s (reservation_cap hook). Use kernel prewarm, once per build:
  tt_metal/tools/kernel_prewarm/prewarm_and_submit.sh -e tmp/ltx25_env.yaml -w $PWD -t 600 -- bash tmp/run25.sh dv145
Then submit each as its own broker job (timeout 600):
  bash tmp/run25.sh dv145_c211 DIFFVAE_NA_CHUNK_BRICKS=2,1,1 DIFFVAE_NA_UNSAFE_CHUNK=1
  bash tmp/run25.sh dv153 NUM_FRAMES=153 FPS=25
  bash tmp/run25.sh conv145 LTX25_DIFFVAE=0
Outputs: tt-project/baselines/ltx25_1080p_6s/<label>/{run.log,*.mp4}. Gen #1 is the traced steady-state replay with a fresh prompt.
Caches are on /tmp (root fs): /tmp/t10-tt-metal-cache, /tmp/t10-dit-cache-ltx25. Delete both when the task ends.
The first run fills the DiT cache and may run past 600s. If so, rerun: the second run loads from cache.
Attempt 2 (2026-09-30): build done (BUILD_EXIT=0). Launched prewarm+dv145 detached; log tmp/prewarm_dv145.log, broker capture job 580.
Next: once tmp/prewarm_dv145.log shows RUN_EXIT[dv145], read baselines/ltx25_1080p_6s/dv145/run.log, then submit dv145_c211, dv153, conv145 one at a time.
Attempt 2 result: dv145 run (job 580) got through gen+decode in 330s, then crashed in ping_pong_buffer_report (list entries, DIFFVAE_MEM_LOG path). Fixed in e15bb15e1ab.
Resubmitted dv145 as broker job 596 (log /var/log/tt-device-broker/2026-09-30_132151_596.log). Kernel + DiT caches now warm.
Next: read baselines/ltx25_1080p_6s/dv145/run.log (per-stage times, [dram] lines, export time), then submit dv145_c211, dv153, conv145 one at a time via tt_device_job_run_bg (env tmp/ltx25_env.yaml, timeout 600).
Attempt 2b (2026-09-30): job 596 timed out at the 600s cap (pytest-timeout 580s) during the warmup gen's stage 2, before gen #1.
Root cause: all 10 DiT weight_load entries report CACHE MISS ("TT_DIT_CACHE_DIR unset or blocking key changed") although TT_DIT_CACHE_DIR=/tmp/t10-dit-cache-ltx25 is set and holds 72G.
The cache dirs were rewritten 13:22-13:29 during job 596, so the cache key changes between runs (job 580 wrote, 596 missed). Load+convert ~7.5 min, text-encode ~1.5 min.
Next: find why the key differs run to run (grep "blocking key" / cache key builder in models/tt_dit/utils/cache*.py; compare the key file written by 580 vs 596), fix, then rerun dv145.
