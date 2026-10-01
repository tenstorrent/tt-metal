# READY_27 — device A/B for task #27 (launch only after the user lifts the device pause)

One mode per broker job, one job at a time (~230 s each, cf. job 610). Run from this worktree.
Code: commit 0495a8e666c (encoder process, default) on top of 63902277007 (#18 thread export).

    cd /home/smarton/fasth3/tt-metal/tt-project/worktrees/t27
    tt-device-mcp run-bg -e tmp/env.yaml -t 600 "bash tmp/ab.sh process"   # new default
    # after it finishes:
    tt-device-mcp run-bg -e tmp/env.yaml -t 600 "bash tmp/ab.sh thread"    # #18 path, LTX_EXPORT_PROCESS=0

Read: grep -E "E2E_WALL_S gen#[23]|Video export|Audio decode:|VAE decode \(forward" <job log>
Baseline job 610 (thread): E2E 6.665 / 6.660 s, VAE 0.69, audio 0.41, export tail 0.3-0.4 s.
Bit-identity: md5 of demuxed video packets, tmp/out/process/*_2.mp4 vs tmp/out/thread/*_2.mp4
(snippet in tmp/bench_pin.py). If process is not faster than thread, set LTX_EXPORT_PROCESS default to 0.
Optional third job (only if process wins): "bash tmp/ab.sh siblings" (LTX_EXPORT_CPUS=32-63).

## Device stop procedure and drop guards (added 2026-10-01, #64)

- **At every device stop** (pause, chip drop, reboot, fabric failure): on each box, list our own queued
  broker jobs (`tt_device_queue_status` / `tt_device_recent_jobs`) and kill each with `tt_device_job_kill`.
  The broker re-queues jobs after a reboot (QUEUE-RESTORE): blx03 job 000 ran at 07:29:16 that way.
  Never kill other tenants' jobs. Skip any box whose broker is being upgraded.
- **`hostfmax.py 1150` is not a guard against drops.** ltx-host job 995 ran at aiclk 1150 and still dropped
  tray 1 (see tt-project/research/blx03_drop_0722.md). Rely on the rules instead: open the full mesh, then
  `create_submesh(2,4)`; one project job at a time per box; no 4x8 runs.
