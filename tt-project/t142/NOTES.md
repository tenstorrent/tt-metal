# t142: 5-seed 4x8 e2e confirm of #138's 6.238 s (continues #141)

## State (2026-10-06 21:05 UTC): blx01 job staged, NOT submitted (reservation cap)
Updates moved the run to blx01 (job 621 clean, 6.031 s). Staged on g15blx01:/var/tmp/fasth3/t142/
(job.sh = local job_blx01.sh, t142_e2e_seeds.py = 9f2b28b7663 test through patch_test.py, md5 4b9442a81885;
blx01's t48 clone lacks 9f2b28b7663). Dry run OK (tree bf7db12a149, clean, build 20:28:59).
Submit was refused by the user's tt-workflows hook (reservation_cap.py): -t > 600 s is blocked, "fix the run".
Measured fresh-process 4x8 e2e (warmup + gen#0 + 1 warm gen): blx01 job 621 661 s (warmup 551 s, incl.
VAE dec/enc cache misses ~45 s, now written), blx03 t138 530 s (warmup 429 s). Biggest warmup piece:
eager audio decode 211 s (blx01) / 161 s (blx03). 5 seeds add ~4 x 8 s. Unsized (600 s default) would be
~6-15% headroom: a SIGKILL mid-run can wedge eth on a shared box, so not submitted.
T2V-safe trims: LTX_WARMUP_ENCODERS=0 (image-encoder warmup, ~45-63 s; Gemma still warmed);
i2v trace families s1_i2v/s2_i2v (~39 s, no knob). Neither gets near 400 s; the audio warmup is the lever.

## Submit (once allowed)
ssh g15blx01 "tt-device-mcp run-bg 'bash /var/tmp/fasth3/t142/job.sh' -w /var/tmp/fasth3/t48 -e /var/tmp/fasth3/t159/env.yaml -t <T>"
Check first: no smarton job RUNNING/QUEUED on blx01 (job 623 t161 was running at 21:03 UTC), fsm healthy,
no autoupdate process (`ps -eo args | grep [a]utoupdate`; pgrep -f over ssh false-positives on itself).
Outputs: /var/tmp/fasth3/t142/out/seed<N>/{ltx_av_fast_1920x1088_s<N>.mp4,_t3s.png,timing.json,done}; resumes.

## Earlier (blx03)
### State (2026-10-06 09:30 UTC): waiting on tray 2, NOTHING submitted
Preconditions failed at 09:17-09:27 UTC:
1. Broker health: FAIL. FSM `recovering` then `down`; tray 2 (chips 8-15) off the bus since 09:12:45,
   bridge resets fail, glx_reset escalation (host rebooted 09:08).
2. Clean full-mesh job since 03:10: literally yes (ltx-host 293 754 s, 04:54 UTC; 295 251 s, 05:03 UTC),
   but every full-mesh tenant job since then dropped tray 2 (9 in a row, ours and ltx-host). Judged not stable.
   #144 rule (no tray-2 incident in the last 30 min) also fails.

Gate used instead (`ready.sh`): broker active, not upgrading, FSM `healthy`; newest incident >= 30 min old;
a tenant job of >= 120 s completed exit 0 after that incident. Prints READY, or DEADLINE after 2026-10-07 10:00 UTC.

## Files (local copies here; deployed to g14blx03:/var/tmp/fasth3/t142/)
- `job.sh`: runner job. One pytest process, warmup + gen#0 + one warm replay per seed without `out/seed<N>/done`.
  Test = 9f2b28b7663's test (LTX_E2E_SEEDS) written untracked to `~/fasth3/t48/models/tt_dit/tests/models/ltx/t142_e2e_seeds.py`
  by `patch_test.py` (removed on exit, not on SIGKILL). No checkout of t48: tree and build stay at b4c2b9f9d6
  (= c4409b1fa2 + conv3d_program_factory.cpp unaligned fix; build 05:17:36 UTC). Dry run OK.
- `patch_test.py`: prompt keyed by seed (seed%3; same as gen-keyed for a full 0..4 run), per-seed save:
  `out/seed<N>/{ltx_av_fast_1920x1088_s<N>.mp4,timing.json,done}`. Logs per attempt: `out/attempt<k>/run.log`.
- `spec.txt`: runner spec, ID t142-confirm5-r1, CONFIG e2e-default-5seed (no configs/ entry yet).

## Next steps
1. Wake (light) when `ready.sh` exits 0. `ssh g14blx03 bash /var/tmp/fasth3/t142/ready.sh -v`:
   READY -> `tt-project/harness/templates/blx03-runner/blx03-enqueue.sh tt-project/t142/spec.txt`, hand off waiting on
   the printed retry_when (survives_reboot, wake_tier standard: eval follows). DEADLINE -> hand off blocked (tray 2 hardware).
2. On the marker `g14blx03:/var/tmp/fasth3/runner/done/t142-confirm5-r1.done`:
   - skipped/failed with seeds missing: the runner counts one broker job's drop twice (killed, then broker-kill after the
     power-cycle; t140-lofi was skipped on "jobs: 371 371"). If the reason lists one job id twice, that is ONE drop:
     delete `configs/e2e-default-5seed` and enqueue `-r2` (job.sh resumes). Two distinct job ids = real skip, record it.
   - done: scp `out/seed*/` here; e2e median/min/max from timing.json `e2e_wall_s`, stage medians from `stages`.
3. Eval (g15blx02, `tt-project/quality/README.md`): seed 0 (boat prompt) vs `baselines/t20/ltx_av_fast_1920x1088_1.mp4`
   and vs `t-e2e/t138/ltx_av_fast_1920x1088_1.mp4` with `video --vbench-ref`; seeds 1-4 have no reference: `video --cand`
   (VBench + stills). One still per seed. Remove the untracked t48 test file if a kill left it.

## Drops seen 03:10-09:27 UTC (all tray 2)
| UTC | job | chips | owner |
|---|---|---|---|
| 03:10:17 | 246 | 8-15 | ours t141 (already in #141) |
| 05:05:17 | 296 | 8-15 | ours t140 (legacy driver) |
| 05:22:39 | 311 | 8-15 | ours t140-baseline |
| 06:01:32 | 331 | 8-15 | ours t140-exact_shard |
| 06:42:22 | 351 | 8-15 | ours t140-exact_shard |
| 07:21:47 | 371 | 8,9,12,13 | ours t140-lofi |
| 08:02:09 | 389 | 8,9,12,13 | ours t140-gate |
| 08:44:12 | 407 | 9 | ltx-host |
| 08:49:33 | 410 | 8-15 | ltx-host |
| 09:12:45 | 431 | 8-15 | ltx-host |
Broker power-cycles (start UTC): 327 (before 05:58), 347 (before 06:39), 367 07:14, 385 07:53, 403 08:33, 427 09:08.

## 2026-10-06 21:56 UTC: g15blx02 run (supersedes the blx03 plan above)
- Driver `t142/driver_g15.sh`, detached as run 689 `t142g15` (pid 21089; rc file state/runs/689/t142g15.rc).
  It waits for 3 healthy broker checks in a row with no other smarton job running/queued, then submits ONE job:
  `t142/job_g15.sh` = t166 job_g15.sh (t166 tree, t158 build/JIT cache, LTX_WARMUP_T2V_ONLY=1, LTX_WARMUP_ENCODERS=0)
  with LTX_FRESH_PROMPTS=0, seeds 0-4, -t 300 (job 406: 152 s warm; page cache cold after the 21:42 reboot).
  Drop = rerun all 5 seeds; two drops in a row = skip. Then ltx_eval per seed vs ref_dv145/seed<N>.mp4
  (seed 0 also --vbench-ref). Outputs: data/g15/t142/{driver.log,jobs.txt,out/r<k>/,summary.txt,quality.txt,eval/,DONE}.
- Gen mapping: _0.mp4 = gen#0 (seed 0, cold), _<i+1>.mp4 = seed i (timed warm). All DEFAULT_LTX_PROMPT.
- Embed cache: pipeline only caches prompt embeds when dynamic_load is set; check encode ~0.2 s per gen in summary.txt.
- Broker state at start: #164's job 407 dropped chips 0-7 (left PCIe) 21:34 UTC; power-cycle 21:42:56;
  broker up 21:52:44; HOLD-DEADLINE-ESCALATE reset + health verified 21:55:00.
- Next: on DONE read driver.log/summary.txt/quality.txt, median/min/max of E2E_WALL_S for gens 1-5, stills per seed.
