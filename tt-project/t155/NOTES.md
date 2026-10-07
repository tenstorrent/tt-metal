# t155: device check of the t152 vol2col_rm CB-sizing fix

Scripts on blx03: ~/fasth3/t155/{build155.sh,job155.sh,fix.diff}. Logs: /var/tmp/fasth3/t155/{build,j1,j2}.log.

## State (2026-10-06 ~05:30 UTC)
- Build: runner job t155-build, broker 306, rc=0 (05:17:37). t48 tree: before bf7db12a14, after b4c2b9f9d6
  (fix be371d08a9f C++ part as one commit on top). Test source: /var/tmp/fasth3/t155/src @73347dfaec8 (t152).
- Attempt 1 jobs t155-j1 (broker 307) and t155-j2 (broker 309) exited 20 at the script's build gate
  without opening the device: the gate grep'd 'Unaligned blocks get exactly num_patches pages', which
  spans two source lines. Fixed to 'blocks get exactly num_patches pages' (job155.sh, build155.sh).
- Requeued as t155-j1-r2 and t155-j2-r2 (j2 gated on j1.log having '1 ok, 0 failed' and T155_PASS),
  behind 8 t140 jobs in the runner queue.

## Next
Wake: read g14blx03:/var/tmp/fasth3/runner/done/t155-j{1,2}-r2.done and /var/tmp/fasth3/t155/j{1,2}.log
(T155_PASS/T155_FAIL, 'Output check best vs table', T155_TIMES). No marker: run
`ssh g14blx03 bash ~/fasth3/runner/runner-start.sh` and wait again.
Production safety: aligned blocks unchanged (min(n,32)); unaligned n<=64 old min(n,64)=n, new n; unaligned
n>64 was rejected by the guard. So every blocking the guard accepted keeps its CB size.

## State (2026-10-07 01:25 UTC)
- t155-j1-r2 = broker job 929 (00:10:53-00:12:43 UTC) failed at mesh open after 0.52 s:
  `RuntimeError: Failed to pin pages for hugepage at virtual address 0x0 with size 0x0` (distributed.py:631).
  At 01:21 blx03 showed HugePages_Free 0/703. Host hugepage exhaustion: no conv3d ran, so it says nothing
  about the fix. The runner logged it as a drop (chips unknown, post-job fabric gate SKIPPED) and will rerun
  j1-r2 itself after 2 healthy checks. A second failure means the runner skips the config (dropped twice).
- Broker at 01:20 UTC: degraded hold loop (glx_reset, fabric check skipped rc 77, HELD). Same window:
  ltx-host job 978 failed in 3.6 s, job 958 abandoned. None of these are ours.
- #167 update applied: job155.sh is now `timeout 270` / `--timeout=250` / SWEEP_MAX_SECONDS=200
  (spec TIMEOUT 300), replaced atomically via mv.

## Next
Wait for g14blx03:/var/tmp/fasth3/runner/done/t155-j2-r2.done (j2 refuses at its gate if j1 did not pass).
Then read j1.log and j2.log. If j1 hits the hugepage error again, record it as an environment failure, not a
fix failure, and requeue once hugepages are free.

## State (2026-10-07 10:40 UTC): moved to blx01
- blx03: runner (pid 376797) was already dead. Parked t155-j1-r2 (.job+.state from running/) and t155-j2-r2
  (from queue/) into /var/tmp/fasth3/runner/parked. Nothing else touched there.
- blx01: scripts in tt-project/t155/blx01 (copies at g15blx01:/var/tmp/fasth3/t155). Driver started 10:36:51 UTC
  (pid 1497322): setup155.sh makes worktree /var/tmp/fasth3/t155/b of /var/tmp/fasth3/t48 at bf7db12a149 + fix.diff
  (commit 4c86a6f7, conv3d_program_factory.cpp aa4ade43a28..c0c02788344), fresh Release build (no ccache), test
  source = git archive c0c02788344 in t155/src. Then broker jobs j1 64,128,5,4,4 (-t 420, inner 390/370) and j2
  64,128,7,4,4 (only if j1 passes; -t = j1 +50%). Health gate waits for no other smarton job (so it queues behind
  #127's t127 driver). At exit it removes the B worktree/build and the JIT cache, keeps logs.
- t48 on blx01 untouched: bf7db12a149 before (the driver logs it again at cleanup).

## Next
Wake when `ssh g15blx01 test -e /var/tmp/fasth3/t155/driver.marker`. Read driver.marker, driver.log, j1.log, j2.log
(T155_PASS/FAIL, 'Output check best vs table', T155_TIMES). Confirm `ls /var/tmp/fasth3/t155/b` is gone.

## 2026-10-07 light wake (blx01)
- j1 (64,128,5,4,4) broker job 777: pytest PASSED, "1 ok, 0 failed", no hang, 14620 us/op (table 64,256,1,8,4 = 12355 us). Driver marked FAIL only because no "Output check best vs table" PCC line was printed (pcc=none). j2 not run. No drops.
- Cleanup done: build dir + JIT removed; t48 HEAD bf7db12a149 unchanged.
- Next (standard): decide how to get the PCC (test prints it only in some path?), then run j2 (64,128,7,4,4) -t ~240 (j1 took 34 s).

## 2026-10-07 standard wake (budget exhausted, $1 left)
- No new device work. A rebuild on blx01 is needed for j2 (build dir was removed), which does not fit this run.
- Hand-off: j1 hang fix confirmed (completed, no hang); PCC not checked; j2 not run. Follow-up filed.

## 2026-10-07 10:49 UTC (#204): rerun with the PCC check
- Cause of pcc=none: blx01 src was `git archive c0c02788344`, which predates 73347dfaec8 (SWEEP_CHECK_ALL). Without it
  run_sweep compares outputs only when the tested blocking beats the table one; j1 (14620 us) was slower than the
  table (12355 us), so no "Output check" line. SWEEP_CHECK_ALL=1 checks the last timed blocking vs the table and
  disables the speed gates.
- Fix: src.tar = git archive 73347dfaec8 (REV file), setup and run155 gate on SWEEP_CHECK_ALL being in the source.
  Timeouts resized to job 777's 34 s: j1 -t 60 (inner 52, pytest 47), j2 -t = j1 +50% (min 60).
- Attempt-1 logs moved to /var/tmp/fasth3/t155/attempt1/. Driver restarted 10:49:00 UTC (pid 1552794): rebuild B,
  then j1, then j2 (only if j1 passes), then cleanup of B + JIT.

## Next
Wake when `ssh g15blx01 test -e /var/tmp/fasth3/t155/driver.marker`. Read driver.marker, driver.log, j1.log, j2.log
(T155_PASS/FAIL pcc=, T155_TIMES). Confirm B and jit are gone and t48 HEAD is bf7db12a149.
- 10:50 UTC: build failed once: ld.lld aborted ("terminate called recursively", std::runtime_error) linking
  tracy-capture-daemon; same script built fine at 10:37. Logs in t155/attempt2/. Driver restarted 10:52 (pid 1568307).
  If the build fails the same way again: add `-j 32` / check `ulimit -u` and thread limits before retrying.
