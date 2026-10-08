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
