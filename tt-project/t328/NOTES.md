# t328: FastH3 baseline rerun at full AICLK on blx03 (follow-up of #314)

## Run 1 (2026-10-10 05:55 UTC): waiting, blx03 still clamped
- blx03 broker active, queue empty. Every device-opening job since the 00:05 UTC reboot logs
  "AICLK failed to settle" on 32/32 chips (900 MHz, expected 1350), up to ltx-host job 852 (05:54 UTC).
  No device job submitted.
- Probe: tt-project/t328/clkprobe.sh (exit 0 = newest finished device-opening blx03 job unclamped).
- 900 MHz reference (#314): t286-t5-rt job 850, 288 s wall, Total Pipeline 7.7421 s;
  t286-t10-rt job 851, 246 s wall, 15.9844 s. Org full clock: 4.57 s / 10.77 s.

## Next (on probe pass)
- run286.sh has T286_ONCE=1: out_t5/PASS and out_t10/PASS exist from #314, so the rerun skips unless they
  move. On blx03: mv /var/tmp/fasth3/t286/out_t5 out_t5_900mhz (same for t10) before enqueueing.
- Specs: copy specs/t5-rt.txt / t10-rt.txt (on ttp/t286-fasth3-baseline 0b22be8499c) to new IDs
  t328-t5-fc / t328-t10-fc (same CONFIG keys h3turbo-5s-rt / h3turbo-10s-rt), TIMEOUT measured +50%:
  t5 450, t10 380 (both <= 600). Enqueue t5, wait on its done marker, then t10 (one at a time).
- After each: verbatim log_pipeline_perf table from out_<tag>/run.log, AICLK clamp count must be 0,
  ffmpeg still, mp4 path; copy still + logs to tt-project/baselines/fasth3/ (suffix _fc).
