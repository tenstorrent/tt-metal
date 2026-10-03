# t96 NOTES: conv VAE decode rebaseline, 2x4 submesh, LTX_CONV3D_BLOCKING_MESH=4,8

Branch ttp/t96-rebaseline-conv-vae-2x4 = t48 tip aa2b400c569 + t87 harness (7d877a1e498, test_vae_ltx_trace_ab.py)
+ t87 LTX_VIDEO_VAE_TRACE/LTX_TIME_STAGES hook (736a1c4d42d; t48 lacks it) + test_vae_ltx_prof_2x4.py + tmp/blx03/t96/.
Build on blx03: ~/fasth3/t48 @ b43f3ea63a (a613d669ee + ce356b8815a ttnn/ part), which closes its only C++ gap vs t48 tip.

## Attempt 1 (job 375, 2026-10-03 01:27 UTC): failed, code mismatch, not a drop
- Job 375 ran 01:32:55-01:33:36 and died on TT_FATAL conv3d_device_operation.cpp:219 padding_mode == "zeros":
  t48 tip defaults LTX_VAE_HALO_ONLY=1 with replicate pad, which needs ce356b8815a's C++. The NOTES claim
  "decoder runs zeros padding" was wrong.
- Chip 15 dropped at 01:29:06 during the broker's fabric-check, while 375 was still QUEUED (not ours). The old
  driver counted errors from submit time and so reported stage ab 9; fixed to count from job start.
- Fix: ce356b8815a's ttnn/ part applied to ~/fasth3/t48 on blx03 as local commit b43f3ea63a; driver now rebuilds
  it (build_metal.sh --release, incremental, log /var/tmp/fasth3/t96/build.log) before submitting.
- Old log kept as /var/tmp/fasth3/t96/run96.375.log.

## Attempt 2 (job 435, 2026-10-03 04:16 UTC): part 1 done, part 2 broke on a script bug
- Pre-submit check: blx03 reboots at 01:21, 02:16, 02:58 UTC; drops at 01:04, 01:29 (queued, not ours), 02:00, 02:27,
  02:32 all hit ltx-host jobs. Our only jobs: 354, 375 (01:32:51-01:33:32, all post-job gates OK). None lined up with ours.
- Job 435 ran 04:16:07-04:17:01, post-job gates healthy, no drop.
- Part 1 (2x4 submesh of full mesh, blocking 4,8, 544x960/145f, fused YUV): decode-only 445 ms eager and traced
  (VAE_DECODE_SPLIT decode=444.8-445.8 eager, 444.9-446.1 traced); total incl upload+output min 0.5169 s eager,
  0.5193 s traced. Trace capture 0.99 s, 2.7 MB. traced == eager bit-identical. Trace gives no gain.
  First (cold) eager decode 10.6 s (kernel compile). Trimmed log: tmp/blx03/t96/results/run96.435.trim.log
- Part 2 failed: conftest's pinning re-exec rebuilds argv as "python pytest", which breaks under tracy -m.
  Fix: LTX_PIN_CORES=0 for part 2; run96p2.sh runs part 2 only (log run96.p2.log); driver takes T96_RUN.

## Attempt 2b (job 436, 2026-10-03 04:19 UTC): profiler pass only, running
- Driver relaunched with T96_RUN=run96p2.sh; log /var/tmp/fasth3/t96/run96.p2.log; marker in driver.log.
- Next: read analysis.txt, compare with tt-project/t61/analysis_029.txt (828 ms/chip: conv3d 308, layout 256,
  halo 119, eltwise 96, norm 46) and the ~490 ms estimate. Copy analysis.txt into tmp/blx03/t96/results/.
  Note: eager wall decode is 445 ms, so the device op sum should land near or below that.

## Running on blx03 (attempt 2, see driver.log for launch time)
- overlay /var/tmp/fasth3/t96/src (stage96.sh), driver /var/tmp/fasth3/t96/driver.log, marker "T96_DRIVER_DONE <stage> <rc>"
- broker job 375: run96.sh, log /var/tmp/fasth3/t96/run96.log
  part 1: "AB arm=eager/traced decode_s=... min=", VAE_DECODE_SPLIT lines, AB_CMP (traced vs eager identical)
  part 2: tracy profile; analysis in /var/tmp/fasth3/t96/analysis.txt, csv copy ops_perf.csv
- stages: build !=0 = rebuild failed (read build.log). ab 9 = drop/ERROR/reboot during OUR job -> STOP ALL device work on every galaxy, report.
  ab 8 = broker never healthy (relaunch driver). analysis 0 = done.
- If driver gone without marker (reboot): relaunch (part 2 only: prefix T96_RUN=run96p2.sh)
  ssh g14blx03 'cd /var/tmp/fasth3/t96 && T96_RUN=run96p2.sh setsid nohup bash src/tmp/blx03/t96/driver96.sh > driver.out 2>&1 < /dev/null &' < /dev/null

## Next step
Read driver.log, run96.log (eager vs traced min, VAE_DECODE_SPLIT decode ms), analysis.txt (per-chip op table).
Compare with t61/job 029 (828 ms device, conv3d 308) and the ~490 ms estimate. Copy logs/analysis here.
Cleanup on blx03: rm -rf /var/tmp/fasth3/t96/{jit,prof,src,yuv_*.pt} (keep logs + analysis until copied).
