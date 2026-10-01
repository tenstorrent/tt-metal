# t40 notes: Gemma text encode for new prompts (LTX-2.5, 1080p/145f)

Branch ttp/t40-... fast-forwarded onto t20 (1eadde3ce6c), then:
- 4b99dfa47de test_gemma4_encode_timing.py: eager / capture / replay timing per phase, gemma48 vs FE vs
  connectors split, gemma-only replay at 1024 vs a 256 bucket, PCC traced-vs-eager and bucket-vs-1024.
- 9e336c44b71 fix: encoder trace is captured right after gen #0's export (capture_trace on both Gemma pairs),
  so gen #1 replays. Unit test models/tt_dit/tests/unit/test_gemma_encode_trace_capture.py (run with python
  directly; fails without the fix, passes with it).
- 02a2cf83fc6 tmp/blx03/t40_drive.sh + t40_prof.sh.

## Diagnosis from job 879 log (no device needed)
The 1.91s is not the encoder's steady cost. The pipeline defers the encode trace until gen #0 has captured every
other trace (open_trace_gate at the end of generate). gen #0 therefore encoded EAGER: 0.26s for a new prompt
(its prompt differed from the warmup "warmup" prompt; no disk cache on the static path). gen #1 was the first
traced encode: prep run (~0.24s, "capturing trace..." logged 0.24s after the request) + capture + execute = 1.91s.

## Device jobs (blx03, driver ~/fasth3/t40/tmp/blx03/t40_drive.sh, started 19:55, waits behind t16 job 886)
1. prof: bash ~/fasth3/t40/tmp/blx03/t40_prof.sh  -> ~/fasth3/out/t40/prof.log (grep GEMMA4_TIMING / GEMMA4_PCC)
2. e2e:  run25.sh conv145_t40 from W=~/fasth3/t40 -> ~/fasth3/out/t40/conv145_t40/{run.log,*.mp4}
Job ids/status: ~/fasth3/out/t40/jobs.txt ; both done when ~/fasth3/out/t40/DONE exists.

## Next
- prof.log: eager vs replay total, device vs host (prep/readback), gemma48 share, 256-bucket replay and PCC.
  Decide bucketing only if replay device time is a meaningful share (>~0.1s) and bucket PCC >= 0.9999.
- e2e: gen#1 Encoder row (expect ~0.1-0.3s vs 1.91), E2E_WALL_S vs 8.761; still at 3s from gen#1 mp4,
  PSNR vs tt-project/baselines/t20/ltx_av_fast_1920x1088_1.mp4 (same prompt, same seed).
- Clean up after: blx03 ~/fasth3/t40 worktree (git worktree remove), ~/fasth3/out/t40 mp4s once copied.

## 2026-09-30 20:27 (attempt 1, wake 2)
blx03 answers ping but refuses ssh (port 22 connection refused): it is most likely rebooting. The detached driver
t40_drive.sh probably died with it. On the next wake: ssh in, check ~/fasth3/out/t40/{jobs.txt,DONE} and the
broker status of the job ids there. If DONE is missing and t40_drive.sh is not running, check /var/tmp/fasth3
still holds Gemma + caches, then relaunch the driver (setsid nohup bash ~/fasth3/t40/tmp/blx03/t40_drive.sh
> ~/fasth3/out/t40/drive.log 2>&1 &), one device job at a time.

## 2026-10-01 10:00 UTC (attempt 1, wake 3)
The 2026-09-30 driver never submitted anything (no jobs.txt). blx03 rebooted ~08:09 UTC with none of our jobs
running (broker startup + fabric-check passed at 08:11). Rules now: full mesh + create_submesh(2,4) only, no 4x8.
- 49af172e201: timing test opens (4,8) and runs on create_submesh(2,4), TP=4 on axis 1. Driver is prof-only;
  the 4x8 e2e step is commented out until full-mesh runs are allowed.
- blx03 ~/fasth3/t40 checked out at 49af172e20; prof driver relaunched: broker job 027 (started 09:59 UTC).
Next: read ~/fasth3/out/t40/prof.log (GEMMA4_TIMING / GEMMA4_PCC). If blx03 dropped while 027 ran: stop all
device work, kill our queued jobs, report. Numbers are TP=4 on 2x4, not the TP=8 production layout.
