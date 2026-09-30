# t10 next steps (written 2026-09-30)
Build: tmp/build.log, ends with BUILD_EXIT=<code>. First try died on a half-downloaded .cpmcache/nlohmann_json (log in tmp/build_fail1.log).
Broker caps jobs at 600s (reservation_cap hook). Use kernel prewarm, once per build:
  tt_metal/tools/kernel_prewarm/prewarm_and_submit.sh -e tmp/ltx25_env.yaml -w $PWD -t 600 -- bash tmp/run25.sh dv145
Then submit each as its own broker job (timeout 600):
  bash tmp/run25.sh dv145_c211 DIFFVAE_NA_CHUNK_BRICKS=2,1,1 DIFFVAE_NA_UNSAFE_CHUNK=1
  bash tmp/run25.sh dv153 NUM_FRAMES=153 FPS=25
  bash tmp/run25.sh conv145 LTX25_DIFFVAE=0
Outputs: tt-project/baselines/ltx25_1080p_6s/<label>/{run.log,*.mp4}. Gen #1 is the traced steady-state replay with a fresh prompt.
Caches are on /var/tmp (root fs; /tmp is wiped at boot): /var/tmp/t10-tt-metal-cache, /var/tmp/t10-dit-cache-ltx25. Delete both when the task ends.
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
Task t26 (2026-09-30): the DiT cache key is NOT unstable. Job 580 was the prewarm kernel-capture pass
(TT_METAL_KERNEL_CAPTURE_ONLY), which by design never writes the weight cache; 596 was the first real
write. Paths/keys are identical in both logs. All 10 caches under /tmp/t10-dit-cache-ltx25 now carry a
manifest that matches the tensorbins (checked offline). Commit f2ddefed262 makes each miss name its real
reason (capture pass / absent / rejected) instead of "blocking key changed".
Baselines: detached driver tmp/drive26.sh (log tmp/drive26.log, ends DRIVE_DONE) submits dv145 (job 605),
then dv145_c211, dv153, conv145 one after another. It stops after dv145 if the transformer still misses.
Next: when DRIVE_DONE, read baselines/ltx25_1080p_6s/<label>/run.log for per-stage times (LTX_TIME_STAGES),
the walltime ledger (expect 10 HITs), the mp4 path; grab a still with ffmpeg; build the table.
Task t26 attempt 2 (2026-09-30): job 605 (dv145) timed out again, every DiT cache "absent". Real cause: the box
rebooted at 13:51 (also 12:19, 07:17) and tmpfiles 'D /tmp' empties /tmp at boot, wiping both caches.
Moved both caches to /var/tmp (same fs, survives reboot); tmp/ltx25_env.yaml and tmp/run25.sh updated.
605 re-published all 10 DiT caches before timing out, so the DiT cache is warm. Commit 3f159a235a5 adds a
"/tmp is wiped at every boot" hint to the miss reason (unit tests: 13 passed, 3 skipped).
Driver: tmp/drive26b.sh (log tmp/drive26b.log, ends DRIVE_DONE; prewarm log tmp/prewarm26b.log, capture job 633).
It runs kernel prewarm + dv145, then dv145_c211, dv153, conv145. A reboot kills it: if tmp/drive26b.log lacks
DRIVE_DONE and no drive26b.sh process exists, check which run.logs passed and resubmit the rest.
Next: when DRIVE_DONE, read baselines/ltx25_1080p_6s/<label>/run.log (LTX_TIME_STAGES lines, walltime ledger,
expect 10 CACHE HITs), mp4 in the same dir; still with ffmpeg -ss 3 -frames:v 1; build the table.
Task t26 attempt 3 (2026-09-30): drive26b stopped early by mistake. prewarm_and_submit.sh returns as soon as the real
run is queued (job 640), so the driver read the stale dv145/run.log from 605 and quit. Prewarm itself passed (3291 kernels).
New driver tmp/drive26c.sh 640 (log tmp/drive26c.log, ends DRIVE_DONE) waits for job 640, then runs dv145_c211, dv153,
conv145. At 14:52 job 640 was 3rd in the queue behind 634/638/639 (each up to 600s).
Next: same as above; when DRIVE_DONE, read the run.logs and build the table.
Task t26 attempt 4 (2026-09-30 15:43): job 640 (dv145) was killed by the broker at 175s when chips left PCIe (box rebooted ~15:23).
Before that it confirmed the fix: every model loaded from /var/tmp/t10-dit-cache-ltx25 (transformer load-cache 62s vs ~7.5 min convert, VAE/upsampler/audio 0-3s).
Both caches survived the reboot on /var/tmp. New resumable driver tmp/drive26d.sh (log tmp/drive26d.log, ends DRIVE_DONE)
skips labels whose run.log already shows " passed". dv145 = job 671 (queued behind 643/651/656).
If the driver dies (reboot): rerun `nohup bash tmp/drive26d.sh >> tmp/drive26d.log 2>&1 &` in the t10 worktree.
Next: when DRIVE_DONE, read baselines/ltx25_1080p_6s/<label>/run.log (LTX_TIME_STAGES lines), mp4 in the same dir; still with ffmpeg -ss 3 -frames:v 1; build the table.
