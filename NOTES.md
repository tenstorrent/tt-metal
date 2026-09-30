# t22 notes
- Plan: tt-project/research/denoise_plan.md
- Win: seeded-noise prefetch, commit cc51d4a21bc (CPU test test_ltx_seeded_noise.py passes, bit-exact draws).
- E2E job 601 (submit log tmp/submit_noisepf.log): bash tmp/e2e.sh noisepf, env tmp/e2e_env.yaml
  (TT_METAL_HOME=t14 worktree for kernel-cache hits; python from t22). Outputs tmp/e2e/noisepf/.
- Next: when 601 is done: grep "denoise init\|Stage [12] denoise" from `tt-device-mcp logs -n 100000 601`;
  compare latents tmp/e2e/noisepf/latents.gen2.pt vs ../t14/tmp/e2e/safe/latents.gen2.pt (expect bit-identical);
  baseline init S1 ~85 ms, S2 ~135 ms, S1 2.18 s, S2 2.46 s (job 574).
- E2E job 602: same + LTX_DEVICE_PROMPT_HANDOFF=1 (plan item #2), outputs tmp/e2e/noisepf_ph/. Compare its latents to safe too
  (handoff may change nothing numerically — the prompt comes from the encoder straight into device buffers).
- 2026-09-30 14:10 UTC (attempt 2): 601/602 still queued (first in line); device HELD by broker for recovery
  (chip 20 left PCIe bus). Nothing run. On resume: `bash tmp/cmp.sh noisepf 601` and `bash tmp/cmp.sh noisepf_ph 602`.
- 2026-09-30 14:15 UTC (attempt 3): 601/602 FAILED in 3 s — t22 had no ttnn/ttnn/_ttnn.so (ImportError
  get_all_unsafe_tracked_ids). Fixed: symlink ttnn/ttnn/_ttnn.so -> t14's build (t22 has no ttnn diff vs t14; import checked OK).
  Resubmitted directly: job 630 (noisepf), 631 (noisepf_ph). On resume: `bash tmp/cmp.sh noisepf 630`, `bash tmp/cmp.sh noisepf_ph 631`.
- 2026-09-30 14:50 UTC (attempt 4): 630/631 FAILED at mesh open — t22 lacked the `runtime` symlink (firmware .ld not found).
  Fixed: `ln -s ../t7/runtime runtime` (same as t14). Resubmitted via prewarm_and_submit.sh (-t 450):
  job 638 (noisepf), 639 (noisepf_ph). On resume: `bash tmp/cmp.sh noisepf 638`, `bash tmp/cmp.sh noisepf_ph 639`.
- 2026-09-30 15:20 UTC (attempt 5): 638/639 FAILED on pytest's own 300 s timeout (pytest.ini) — kernel JIT cache
  missed (0/818 hits; prewarm skipped t14's manifest entries as foreign-tree), so in-process compile ate the budget.
  Fixed: e2e.sh passes --timeout=1500; resubmitted via prewarm_and_submit.sh with -c (capture for the t22 tree),
  then noisepf_ph without -c. Submitter log tmp/submit_attempt5.log; `bash tmp/done.sh` exits 0 when both finish.
  On resume: job IDs = `grep -oE 'Job [0-9]+ queued' tmp/submit_attempt5.log`, then `bash tmp/cmp.sh noisepf <id1>`,
  `bash tmp/cmp.sh noisepf_ph <id2>` (read the log via /var/log/tt-device-broker/*_<id>.log if `logs` says not found).
- 2026-09-30 15:35 UTC (attempt 6): submitter from attempt 5 had died with capture job 651 still queued (behind 641-643).
  Root cause of 0/818 hits confirmed: runs resolve kernels under the t22 tree (job 638 log paths), but the
  offline compile used TT_METAL_HOME=t14, so it skipped t22 recipes as foreign-tree. New driver tmp/drive6.sh
  (nohup, log tmp/drive6.log) waits for 651, compiles with TT_METAL_HOME=t22/, waits until no project job is
  queued/running (coordinator rule 15:27: one project device job at a time, short jobs), then run-bg's ONLY
  noisepf; ID lands in tmp/drive6.jobs. `bash tmp/done.sh` exits 0 when it finishes.
  On resume: `cat tmp/drive6.log`; ID = `grep -oE 'Job [0-9]+ queued' tmp/drive6.jobs`; then
  `bash tmp/cmp.sh noisepf <id>`. noisepf_ph (LTX_DEVICE_PROMPT_HANDOFF=1) is deferred: submit it as its own
  single job only after noisepf is done and the queue is clear. If drive6.log shows no DRIVE6_SUBMITTED
  and the driver is gone, rerun `nohup setsid bash tmp/drive6.sh > tmp/drive6.log 2>&1 &` (edit the wait if 651 is done).
- 2026-09-30 16:03 UTC (attempt 7): box rebooted 16:00 UTC and killed the drive6 driver (651 still queued, #2 behind 643;
  656 and 671 also queued). Restarted `nohup setsid bash tmp/drive6.sh` (PID 9190, log tmp/drive6.log). busy() already
  counts queued smarton jobs, so the e2e run goes in only after 643/651/656/671 all finish. Resume steps unchanged.
