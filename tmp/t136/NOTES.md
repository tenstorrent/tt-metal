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
