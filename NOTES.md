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

## Next step on resume
1. `grep -E 'T85_(AB|BLOCK|STACK|FAIL|EXIT)' /var/tmp/fasth3/t85/run85.log` on blx03.
2. Per-step saving = 48 x (base - arm) ms_per_block, minus T85_STACK for the adaln arm.
   E2E = 8 x S1 per-step + 3 x S2 per-step.
3. Arms bit_identical=True and faster: flip default to "1" on ttp/t48-ltx25-integrated
   (origin @29a0e8dfdc8; local t48 is stale): cherry-pick 129f0081c59, change default, CPU tests, push.
4. LTX_AGMM_K2048 stays opt-in: only hits at TP=4 with Ring (4x8), not measurable under the 2x4 rule.
5. Remove ~/fasth3/t85 and /var/tmp/fasth3/t85 on blx03.
