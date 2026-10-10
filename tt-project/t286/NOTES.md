# t286 FastH3 baseline (Turbo 4-step fl2va, 4x8)

- Box: blx01 (blx03 unreachable 2026-10-08 ~15:00 UTC: ssh "No route to host", no blx03.READY).
- Code on blx01: /var/tmp/fasth3/t284/b = ttp/fasth3-hyperflow e24a2b93d79 + test import fix
  (git apply fix_import.patch, uncommitted there). Same fix committed as acebc7d39e2 on
  ttp/t286-fasth3-baseline. The test imported align_num_frames from packing; it lives in policy.
  Upstream pshah/minimax-h3-hyperflow-turbo has the same bug.
- Test (unmodified apart from the import): models/tt_dit/tests/models/minimax_h3/test_pipeline_turbo_minimax_h3.py
  -k "<dur>s and not 4x32 and not WH" -> test_turbo_end_to_end[NOTSET-4x8-<dur>s]. 768p = 1344x768.
  No 6 s case in the test (DURATIONS_S = 5, 10, 15): baseline uses 5 s and 10 s.
- Caches: fresh /var/tmp/fasth3/cache/{dit-h3hf,tt-metal-cache-h3hf} (weight subfolder
  transformer_resident_adaln does not match older caches).
- Driver: /var/tmp/fasth3/t286/drv286.sh (pid 694174, started 15:15:44 UTC): fill1..3 (5 s, until
  fill.ok) -> t10 -> t5; each job -t 600 via blx01 broker. Log drv286.log, marker drv286.marker.
  Outputs out_<tag>/run.log and mp4.

## Jobs
- 064 (15:11 UTC): failed rc=2 at collection (ImportError above). No device time.
- Drop (not our t286 job): 2026-10-08 15:14 UTC, blx01, broker job 066 (smarton, t283 run283c bf16),
  chip 27 left PCIe; fabric-check 067 failed, bridge-reset 069 failed, broker HELD degraded.
  Driver waits for health (up to 4 h).

- 080 (fill1, 15:40:47-15:47:54 UTC, blx01): FAILED after 426 s, `No space left on device`
  (blx01 root fs filled by our /var/tmp/fasth3 data, see #289). Not a chip drop; no new incident.
  It got past weight conversion (205 s) and died writing the dit cache; the torn 38G staging dir
  (.TP4_0_SP8_1_mesh4x8_bf16.staging-*) was deleted 17:08 UTC. Old log kept as out_fill1_enospc080.
- drv286 is gone (died with the disk-full); not restarted. From now on one job per run via
  /var/tmp/fasth3/t286/sub286.sh <tag> <dur> (health gate + one run-bg, no waiting), polled by
  broker job status. Order: fill1 5 (repeat as fill2 if it ends without T286_EXIT=0) -> t10 10 -> t5 5.
- 17:08 UTC: blx01 healthy, root 52%, /var/tmp/fasth3 104G, but t283 job 083 (smarton) running:
  waiting for it (one project job per box). blx03 still "No route to host".

## Next step
On wake (marker exists): grep "PERFORMANCE RESULTS" blocks from out_t10/run.log and out_t5/run.log,
copy mp4s as reference, ffmpeg a still, copy stills+logs to tt-project/baselines/fasth3/.

## 2026-10-08 ~20:35 UTC (run 1206)
- Job 086 (fill1, run286.sh) incident: opened the 4x8 mesh, then hung in D state on a wekafs stat of
  MINIMAX_H3_MODEL_PATH (/mnt/MLPerf). Broker reaped it 17:21 UTC; its pytest survived and held the
  device until the blx01 reboot at ~19:05 UTC (took live LTX down). Now: blx01 up 12 min, no t286
  process left (checked ps), broker fsm "recovering"/HELD degraded, t301 job 126 running and t293
  job 145 queued (other project tasks).
- run286.sh fixed (here and on blx01): work runs under setsid, EXIT/TERM/INT trap kills the whole
  group; weights from T286_MODEL (default /var/tmp/fasth3/models/MiniMax-H3), refuses /mnt/*,
  needs a READ_OK file written after a full read-through outside the device job.
- Blocker: local weights needed = text_encoder 63G + transformer 62G + vae 9.8G + audio_vae 0.6G
  (~136G, transformer_ref not needed) + ~38G dit cache. blx01 /var/tmp/fasth3 is already 132G
  (cap 150G; t220 69G, cache/tt-metal-cache 28G, t301 11G, not ours). Does not fit on blx01.
  blx03 (597G free) is still "No route to host". Waiting for blx03.

## Next step (when blx03 answers ssh)
1. ssh g14blx03: df -h /, ls /var/tmp/fasth3 (H3 weights already there?), broker health.
2. Copy weights (no transformer_ref) to blx03 /var/tmp/fasth3/models/MiniMax-H3 outside any device
   job (detached setsid nohup rsync), then read-through (cat >/dev/null or sha256sum), touch READ_OK.
3. Code under ~/fasth3 on blx03: ttp/fasth3-hyperflow e24a2b93d79 + fix_import.patch (needs a build
   or reuse of an existing blx03 build); venv; lora + keyframe copied to /var/tmp/fasth3.
4. Through blx03 serial runner (blx03-enqueue.sh), one spec per job: fill 5 s, then t10, then t5,
   each -t 600. Delete the weight copy afterwards unless #288 needs it.

## 2026-10-09 04:42 UTC (run 1226)
- blx03: still "No route to host". Exabox tunnel: answered once (slurm-login-01, 04:41:56 UTC), then
  every command hung and ssh timed out at banner exchange: tunnel down again. No device time spent,
  nothing staged on exabox.
- Exabox path (when only exabox is up): need to find a fasth3 dir in user space, stage ~136G weights
  (no transformer_ref) + build ttp/fasth3-hyperflow e24a2b93d79 there, read-through + READ_OK, then
  one self-ending sbatch per clip on an idle (2 h+ LastBusyTime) dit node, full 4x8 mesh. This is
  several runs of work; budget for #286 is nearly spent ($8 cap).

## 2026-10-09 04:52 UTC (run 1227)
- Woken by probe pass (exabox tunnel flap). blx03: "No route to host". exabox-login: 3 tries,
  all "Connection timed out during banner exchange" (04:51 UTC). Nothing done, no device time.
- Probe tightened: exabox must answer sinfo twice, 30 s apart, before waking.

## 2026-10-10 02:25 UTC (run 1238): blx03 back, staging + build started
- blx03 answers ssh (up 2h19m, / 60%, 358G free). Our /var/tmp/fasth3 there is already 344G
  (older tasks' data); +136G weights +~38G dit cache will bring / to ~79% (blx03 lint cap 85%).
- Weight source (read only outside device jobs): tt-shield volume
  .../volume_id_tt_transformers-MiniMax-H3-FL2VA-LightX2V-v0.24.0/weights/MiniMax-H3-FL2VA-LightX2V
  (has transformer/ and lightx2v/ LoRA; the HF hub snapshot only has transformer_ref).
- stage286 (blx03 detached, /var/tmp/fasth3/t286/stage286.{sh,log,rc}): rsync (no .cache,
  transformer_ref, lightx2v) -> /var/tmp/fasth3/models/MiniMax-H3, LoRA -> models/lightx2v-h3-turbo,
  full cat read-through, then READ_OK.
- setup286 (blx03 detached, ~/fasth3/setup286.{sh,log,rc}): worktree ~/fasth3/t286 at
  origin/ttp/t286-fasth3-baseline acebc7d39e2 (= e24a2b93d79 + test import fix), build_metal --release.
  Venv: ~/fasth3/tt-metal/python_env (T286_VENV). Keyframe copied from blx01 to /var/tmp/fasth3/t209.
- run286.sh: T286_VENV knob, df / (T286_ROOT_MAX, 85 on blx03) and own-footprint (<=120G) guards.
  blx03 lint passes. Specs: tt-project/t286/specs/{fill1,t10,t5}.txt (NEEDS = READ_OK, LoRA,
  _ttnn.so, keyframe; TIMEOUT 600).

## Next step
1. Both rc files 0 (cat g14blx03:/var/tmp/fasth3/t286/stage286.rc, ~/fasth3/setup286.rc).
2. for s in fill1 t10 t5; do tt-project/harness/templates/blx03-runner/blx03-enqueue.sh tt-project/t286/specs/$s.txt; done
   wait on t286-t5's done marker.
3. Then: grep PERFORMANCE/timing blocks from /var/tmp/fasth3/t286/out_{t10,t5}/run.log, ffmpeg a
   still, copy stills+logs to tt-project/baselines/fasth3/, keep one mp4 under /var/tmp/fasth3/t286.
4. Cleanup: delete models/MiniMax-H3 (136G) + caches dit-h3hf/tt-metal-cache-h3hf unless #288
   needs them; remove ~/fasth3/t286 build after.

## 2026-10-10 02:45 UTC (run 1243): first enqueue failed at submit, re-queued as -r2
- t286-fill1/t10/t5 all failed before the device: broker run-bg looked for python_env under
  WORKDIR (/var/tmp/fasth3/t286/tt-metal/python_env). No device time used, no drop.
- Fix: specs now carry ENV=/home/smarton/fasth3/runner/env.yaml (PYTHON_ENV_DIR); run286.sh still
  sets TT_METAL_HOME/PYTHONPATH/caches itself. Re-queued as t286-{fill1,t10,t5}-r2.
- fill1-r2 submitted as blx03 broker job 794 (02:44:54 UTC).
## Next step
Wait on g14blx03:/var/tmp/fasth3/runner/done/t286-t5-r2.done, then steps 3-4 above.

## 2026-10-10 (run 1247): all three -r2 jobs failed, no timing yet
- fill1-r2 = blx03 broker job 794: TIMEOUT at 600 s (cold JIT/cache fill did not finish in the cap).
  Post-job health OK. Not a drop.
- t10-r2 = job 800, t5-r2 = job 803: exit 5 after ~2 min each. pytest exit 5 = no tests collected:
  the test selector in run286.sh / specs t10,t5 likely does not match a test id. Logs:
  /var/log/tt-device-broker/2026-10-10_030204_800.log, ..._030356_803.log (blx03).
- Weights (models/MiniMax-H3, lightx2v-h3-turbo) and the ~/fasth3/t286 build are still on blx03.
## Next step
1. `--collect-only` the test (no device) in ~/fasth3/t286 to fix the selector for t10/t5.
2. Split fill: the cold cache fill must fit 600 s (e.g. fill per stage, or fewer blocks per job).
3. Then rerun t10, t5; cleanup per steps 3-4 above.

## 2026-10-10 ~03:20 UTC (#314, run 1250)
- Correction: t10-r2 (800) and t5-r2 (803) exit 5 was run286.sh's own root-disk guard ("[t286] / at 88%",
  5 s runtime), not pytest. Selector was fine; tightened anyway to "<d>s and 4x8 and not 15s and not
  4x32 and not WH"; --collect-only on blx03 selects exactly test_turbo_end_to_end[NOTSET-4x8-5s] / [...-10s].
- Freed space: deleted /var/tmp/fasth3/cache/dit-ltx25/gemma4-12b-with-proj-ltx-2.5-bf16 (~50G, LTX 2.5
  text-encoder weight cache, project-created 09-30/10-01; LTX 2.5 work closed). blx03 / 88% -> 82% (164G free).
- fill1-r2 (794) left a complete dit cache (dit-h3hf 115G: transformer 63G, text_encoder 47G, vae 4.6G,
  audio 0.5G); it timed out in the construction warmup at VAE canvas 42/93 (~2.3 s per canvas, mostly readback).
  That warmup runs in every process, so a separate fill job would not help. t286_skipvaewarm.py
  (-p plugin, T286_SKIP_VAE_WARM=1) decodes only this test's canvases + the full-wave canvas at warmup.
  Measured call unchanged (the test's untimed priming call at the same shape runs first).
- blx03 log shows AICLK clamped at 900 MHz (expected 1350) on all chips during job 794 (same clamp as
  blx01 in #305). Timings taken under it are relative, not headline.
- Queued on blx03 runner: t286-t5-r3 then t286-t10-r3 (TIMEOUT 600, unmeasured).
## Next step
Wait on g14blx03:/var/tmp/fasth3/runner/done/t286-t10-r3.done (probe.sh t286-t10-r3). Then read both
markers, grep the log_pipeline_perf table from /var/tmp/fasth3/t286/out_{t5,t10}/run.log, check AICLK
lines, ffmpeg a still, copy stills+logs to tt-project/baselines/fasth3/. If t5 timed out: read where
the time went in out_t5/run.log and restructure.

## 2026-10-10 ~04:45 UTC (#314, run 1270): r3 jobs failed, restructured with a warm deadline
- t5-r3: two broker jobs, both killed at the 600 s cap inside the construction warmup, and the
  runner counted both as drops (config h3turbo-5s now skipped):
  - job 807 (03:13): VAE warm 6 s (plugin OK), then audio decode warm cold-compiled at ~60 s/length,
    9/12 done at the kill. Post-job: fabric UNHEALTHY ("mesh cannot run a program"), broker reset.
  - job 820 (03:55): audio done, prompt-encoder keyframe layouts 8/28 at ~25 s each (cold JIT) at the
    kill. Post-job: chip 30 (tray 3) fell off PCIe; broker walked per-tray BMC resets, chips 16-31
    off the bus at 04:09:47, recovered later (t10-r3 at 04:32 had a healthy post-job gate).
- t10-r3 = job 834: exit 5 in 5 s on run286.sh's own guard "[t286] footprint 124G" (limit 120:
  dit-h3hf 115G + JIT cache 9.5G).
- Cause: the 4x8 preset buckets (26 rungs) and traces audio, so construction warms every served
  shape (93 VAE canvases, 12 audio lengths, 28 prompt layouts, 26 denoise rungs). Cold JIT does not
  fit one job; JIT persists in /var/tmp/fasth3/cache/tt-metal-cache-h3hf, so each job gets further.
  Killing a job at the cap twice left the box unhealthy, so jobs must end before the cap.
- Fix: t286_skipvaewarm.py adds T286_WARM_DEADLINE_S (stop between warm items past it, RuntimeError
  -> test fails, mesh closes cleanly) and T286_WARM_LATE_S (fail if warmup ends later than this, so
  the generation never starts too late). run286.sh: T286_FOOT_MAX (140), T286_ONCE=1 (a clip with
  out_<tag>/PASS exits before the device). Prompt-encoder/audio/denoise warmups are NOT trimmed:
  4x8 captures audio traces, and compiling after capture can corrupt a replay.
- New config keys h3turbo-5s-dl / h3turbo-10s-dl (restructured job, not a retry of the dropped one).
- blx03 / at 72% (252G free) at 04:45 (someone else freed space).
- Queued: t286-t5-dl{a,b,c,d} (deadline 400, late 330) then t286-t10-dl{a,b} (400/300).
## Next step
Wait on probe.sh t286-t10-dlb. Then: done markers of all six; out_t5/PASS and out_t10/PASS; grep
log_pipeline_perf table + "[t286] construction warmup done" from out_{t5,t10}/run.log; AICLK lines;
ffmpeg still; copy stills+logs to tt-project/baselines/fasth3/. If no PASS: read how far warmup got
in the last job (it should advance each job) and queue more dl jobs, or trim warmups if a fully warm
warmup alone exceeds ~330 s.
