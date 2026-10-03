# t96 NOTES: conv VAE decode rebaseline, 2x4 submesh, LTX_CONV3D_BLOCKING_MESH=4,8

Branch ttp/t96-rebaseline-conv-vae-2x4 = t48 tip aa2b400c569 + t87 harness (7d877a1e498, test_vae_ltx_trace_ab.py)
+ t87 LTX_VIDEO_VAE_TRACE/LTX_TIME_STAGES hook (736a1c4d42d; t48 lacks it) + test_vae_ltx_prof_2x4.py + tmp/blx03/t96/.
Build on blx03: ~/fasth3/t48 @ a613d669ee. Its only C++ gap vs t48 tip is conv3d halo "replicate" support (ce356b8815a);
the decoder runs zeros padding, where old and new reader_vol2col behave the same.

## Running on blx03 (launched 2026-10-03 01:27 UTC)
- overlay /var/tmp/fasth3/t96/src (stage96.sh), driver /var/tmp/fasth3/t96/driver.log, marker "T96_DRIVER_DONE <stage> <rc>"
- broker job 375: run96.sh, log /var/tmp/fasth3/t96/run96.log
  part 1: "AB arm=eager/traced decode_s=... min=", VAE_DECODE_SPLIT lines, AB_CMP (traced vs eager identical)
  part 2: tracy profile; analysis in /var/tmp/fasth3/t96/analysis.txt, csv copy ops_perf.csv
- stages: ab 9 = drop/ERROR/reboot during OUR job -> STOP ALL device work on every galaxy, report.
  ab 8 = broker never healthy (relaunch driver). analysis 0 = done.
- If driver gone without marker (reboot): relaunch
  ssh g14blx03 'cd /var/tmp/fasth3/t96 && setsid nohup bash src/tmp/blx03/t96/driver96.sh > driver.out 2>&1 < /dev/null &' < /dev/null

## Next step
Read driver.log, run96.log (eager vs traced min, VAE_DECODE_SPLIT decode ms), analysis.txt (per-chip op table).
Compare with t61/job 029 (828 ms device, conv3d 308) and the ~490 ms estimate. Copy logs/analysis here.
Cleanup on blx03: rm -rf /var/tmp/fasth3/t96/{jit,prof,src,yuv_*.pt} (keep logs + analysis until copied).
