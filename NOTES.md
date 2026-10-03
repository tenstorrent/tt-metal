# t85 notes

## State (2026-10-01 22:00 UTC)
- Code: 129f0081c59 (3 opt-in flags + CPU tests, 7 pass), aadf468a540 (device A/B files).
- Device A/B staged on blx03 at ~/fasth3/t85 (src = this branch @aadf468a540, run85.sh, test_denoise_trims_ab.py).
- Detached driver on blx03: ~/fasth3/t85/driver85.sh (copy of tmp/t85/driver.sh), pid 78652.
  It waits up to 8 h for a healthy broker (blx03 was HELD: tray 1 drop in another tenant's job 104,
  then a reset left 32/32 off-bus), submits ONE job (run85.sh, 2x4 submesh from full mesh), watches it,
  then checks the broker log for drops/reboot during our job.
- Log: /var/tmp/fasth3/t85/driver.log, done marker `T85_DRIVER_DONE ab <rc>`.
  rc 0 ok, 9 = drop/error/reboot during OUR job (=> stop all device work, report), 8 = never healthy, 7 = submit failed.
  Job output: /var/tmp/fasth3/t85/run85.log (also ~/fasth3/t85/run85.log).
- If blx03 reboots before the job, the driver dies without a marker: relaunch it
  (`ssh g14blx03`, then `setsid nohup bash ~/fasth3/t85/driver85.sh > /var/tmp/fasth3/t85/driver.out 2>&1 < /dev/null &`, run with ssh -f or it hangs).

## Attempt 2 (2026-10-02 18:38 UTC)
- blx03 rebooted 2026-10-02 18:23 UTC (driver died, no job had run). Broker healthy again.
- Relaunched driver85.sh (pid 41834); it submitted job 161 (queued behind ltx-host 160).
- Old driver log moved to /var/tmp/fasth3/t85/driver.prev.log.

## Next step on resume
1. `grep -E 'T85_(AB|BLOCK|STACK|FAIL|EXIT)' /var/tmp/fasth3/t85/run85.log` on blx03.
2. Per-step saving = 48 x (base - arm) ms_per_block, minus T85_STACK for the adaln arm.
   E2E = 8 x S1 per-step + 3 x S2 per-step.
3. Arms bit_identical=True and faster: flip default to "1" on ttp/t48-ltx25-integrated
   (origin @29a0e8dfdc8; local t48 is stale): cherry-pick 129f0081c59, change default, CPU tests, push.
4. LTX_AGMM_K2048 stays opt-in: only hits at TP=4 with Ring (4x8), not measurable under the 2x4 rule.
5. Remove ~/fasth3/t85 and /var/tmp/fasth3/t85 on blx03.

## Result (2026-10-03 00:45 UTC) — DONE
Job 161 ran 2026-10-02 19:04 (52.9 s, exit 0, post-job health OK). Later blx03 reboots were after our job.
All arms bit-identical (maxabs 0). ms/block, 2x4 submesh:
- S1 base 14.003 | v2a 14.007 | adaln 13.998 | all 14.013  (all within noise)
- S2 base 60.560 | v2a 60.269 (-0.48%) | adaln 60.543 | all 60.387
- T85_STACK (batched AdaLN table, per step) 1.390 ms vs <=0.24 ms saved in-block => net loss.
Decisions: V2A skip -> default on t48 (f793ec1c64a, pushed). AdaLN batch rejected (opt-in, default 0).
AGMM K2048 stays opt-in (only hits at TP=4 Ring on 4x8; untestable under 2x4 rule).
Projected e2e for V2A: <= 3 S2 steps x 48 x 0.29 ms ~ 42 ms upper bound; likely less (all-arm shows -0.17).
blx03 staging (~/fasth3/t85, /var/tmp/fasth3/t85) removed.
