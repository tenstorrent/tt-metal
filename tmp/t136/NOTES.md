# t136: #134 LTX_SDPA_MM_LOFI A/B, full 4x8 e2e on blx03 (per user update #111)

Scripts: tmp/t136/run_ab.sh (broker job), tmp/t136/driver.sh (detached on blx03), copies in /var/tmp/fasth3/t136/.
Build: blx03 ~/fasth3/t134 at e146b42f0e (SETUP134_DONE rc=0). One broker job, two pytest processes:
OFF (knob unset) then LOFI (LTX_SDPA_MM_LOFI=1). Each is #138's protocol (bh_4x8sp1tp0_ring, 1920x1088/145f,
SEED=0, warmup, gen#0 cold, gen#1 warm = headline, LTX_CONV3D_BLOCKING_MESH=4,8).
Baseline: #138 job 099 (t48 c4409b1fa24): gen#1 E2E 6.238 s, S1 2.25 s, S2 2.50 s; mp4s in
/var/tmp/fasth3/t138/out and tt-project/t-e2e/t138/.

## 2026-10-06 02:17 UTC: submitted
Broker job 212 (timeout 3600 s), started at once (queue empty, post-job gate 01:57 OK). blx03 boot 00:43:02.
Outputs: /var/tmp/fasth3/t136/{driver.log,job_ids,broker_slice_*.log,journal_slice_*.log,out/{off,lofi}/}.
Done: grep T136_DRIVER_DONE /var/tmp/fasth3/t136/driver.log. If blx03 rebooted (boot != 00:43:02) the driver
died: read the broker log for job 212's fate, wait for health, rerun once (WATCH_JOB=<id> WATCH_T0=... to
re-attach if the broker re-queued it).
Next: pull out/{off,lofi}/run.log timings (E2E_WALL_S, stage table), copy mp4s + t3s stills to
tt-project/t-e2e/t136/, run ltx_eval video --vbench none: OFF vs t138 (gate: identical or ~inf PSNR),
LOFI vs t138, LOFI vs OFF. Recommend LOFI for the eval pack only if S2 denoise drops and PSNR holds.

## 2026-10-06 02:37 UTC: drop 1 (run 2)
Job 212 killed at 02:19:44 UTC: tray 2 (chips 8-15, incl. chip 12) left the PCIe bus during OFF arm warmup
(Gemma encode, no timings). Our job; broker DEAD-CHIP kill. Post-job gate failed, broker power-cycled, blx03
rebooted 02:26:22, which killed our driver before its rerun. Then t141's job 224 (not ours) dropped tray 2 again
at ~02:31; broker in recovery at 02:36. t140 and t141 drivers are waiting on blx03 to submit.
driver.sh now has RERUN=1 (attempt 2; a second drop ends with T136_DRIVER_DONE e2e_drop2 = config skipped).
Next: once no other project driver/job is active and the broker is idle, relaunch on blx03:
  ssh g14blx03 'cd /var/tmp/fasth3/t136 && RERUN=1 setsid nohup bash /var/tmp/fasth3/t136/driver.sh >> driver.out 2>&1 < /dev/null &'
(driver.sh on blx03 is already the RERUN version). Then wait on grep T136_DRIVER_DONE in driver.log (after the relaunch line).
