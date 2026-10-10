# t330: FastH3 swirl check, matched-input t2va fox run on blx03 (follow-up of #329)

Question: do the white swirls in the #314 baseline (blx03 jobs 850/851: fl2va, lake-house keyframe
/var/tmp/fasth3/t209/kf_first.png + CALIBRATED_FOX_PROMPT, seed 0) come from the mismatched inputs or
from the device?

## Setup (run 1, 2026-10-10 ~06:05 UTC)
- Job script: run330.sh (copy of t286/run286.sh with task/keyframe args; same build ~/fasth3/t286 @
  acebc7d39e, same t286_skipvaewarm plugin and rungs). Outputs: blx03 /var/tmp/fasth3/t330/out_<tag>/.
  Deployed to blx03 /var/tmp/fasth3/t330/run330.sh; lint --device passes (rc 0).
- Test working point: 768p, video shift 6 / audio shift 3, 4 forwards, seed 0 = the B300 Turbo t2va
  reference in tt-inference-server#5308 (comment "Added: lightx2v MiniMax-H3 Turbo"). That issue has
  no public output video or still, only timings.
- Expected t2va seq_len 39773 - 1008 (no keyframe) = 38765 -> rung 38912 (warmed by job 850).
- Job 1: specs/t2va.txt (ID t330-t2va-5s, TIMEOUT 450 = job 850's 288 s +50%).
- Job 2 (optional): specs/fl2va.txt, keyframe = frame 0 of the t2va clip, extracted to
  blx03 /var/tmp/fasth3/t330/kf_t2va_f0.png before enqueue.
- blx03 AICLK still clamped at 900 MHz (probe tt-project/t328/clkprobe.sh exits 1). Spec: wait.

## Next step
On clkprobe pass: tt-project/harness/templates/blx03-runner/blx03-enqueue.sh tt-project/t330/specs/t2va.txt,
hand off waiting on its probe. Then: grep "[t330]", "-> rung", Total Pipeline, AICLK count from
out_t2va/run.log; ffmpeg stills (same frame times as #314 stills) + frame 0 as kf_t2va_f0.png;
compare with tt-project/baselines/fasth3/still_t5.jpg.
