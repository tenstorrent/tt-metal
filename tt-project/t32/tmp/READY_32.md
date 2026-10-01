# t32 ready-to-launch device jobs (DO NOT SUBMIT while the g15blx02 device pause holds)

One config per job, 5 trace replays, 600 s cap. Run ONE at a time, from this worktree
(/home/smarton/fasth3/tt-metal/tt-project/worktrees/t32). Kernel cache: /var/tmp/t32-tt-metal-cache (partly warm for S1).
Every job also re-times the shipped config first (the `ref` line) and compares against it.
Read results with: `tt-device-mcp logs -n 100000 <ID> | grep SWEEP | sed 's/.*SWEEP/SWEEP/'`

Order = expected value. Stop after 1-2 if the box is still unstable.

1. S2 V2A split-K vs shipped ring cross (t14 est. -0.5..-0.9 ms/block at S2):
   tt-device-mcp run-bg -e tmp/env.yaml -t 600 'export LTX_SWEEP_SELF="" LTX_SWEEP_CROSS_K="" LTX_SWEEP_OPS=5; bash tmp/sweep.sh stage_2'
2. S1 V2A split-K vs ring cross (est. -0.1..-0.2 ms/block):
   tt-device-mcp run-bg -e tmp/env.yaml -t 600 'export LTX_SWEEP_SELF="" LTX_SWEEP_CROSS_K="" LTX_SWEEP_OPS=5; bash tmp/sweep.sh stage_1'
3. S1 self (96,608): 19-tile K chunk divides the 38-tile shard, 2 K chunks instead of 5, no padded K compute:
   tt-device-mcp run-bg -e tmp/env.yaml -t 600 'export LTX_SWEEP_SELF="96,608 96,416" LTX_SWEEP_CROSS_K="" LTX_SWEEP_SPLITK=0 LTX_SWEEP_OPS=5; bash tmp/sweep.sh stage_1'
4. S2 self (384,256) / (192,448) / (128,608). (384,256) may fail L1 validation (logged as FAIL, no device impact):
   tt-device-mcp run-bg -e tmp/env.yaml -t 600 'export LTX_SWEEP_SELF="384,256 192,448 128,608" LTX_SWEEP_CROSS_K="" LTX_SWEEP_SPLITK=0 LTX_SWEEP_OPS=5; bash tmp/sweep.sh stage_2'
5. Only if split-K loses: ring cross k_chunk at S2:
   tt-device-mcp run-bg -e tmp/env.yaml -t 600 'export LTX_SWEEP_SELF="" LTX_SWEEP_CROSS_K="1024 2048" LTX_SWEEP_SPLITK=0 LTX_SWEEP_OPS=5; bash tmp/sweep.sh stage_2'

## How to judge
- Self configs: accept if us drops > ~3% vs the `ref` line and rel_l2 vs shipped < 1e-2 (not bit-identical:
  K chunking changes the online-softmax accumulation order). Then block A/B + e2e PCC before shipping.
- Split-K: accept if us beats the ring cross `ref` and its host_rel_l2 is <= ~1.5x the ring cross host_rel_l2.
  CPU bf16 emulation (test_v2a_split_k_reference.py) predicts split-K 0.0063/0.0066 vs 0.0061/0.0064 without
  the split (S1/S2), i.e. ~3% more error than dense bf16 attention; the math itself matches dense attention to 1e-5.
- Discard job 656's numbers: it was a capture-only prewarm (TT_METAL_KERNEL_CAPTURE_ONLY=1), kernels did not run.

## Device stop procedure and drop guards (added 2026-10-01, #64)

- **At every device stop** (pause, chip drop, reboot, fabric failure): on each box, list our own queued
  broker jobs (`tt_device_queue_status` / `tt_device_recent_jobs`) and kill each with `tt_device_job_kill`.
  The broker re-queues jobs after a reboot (QUEUE-RESTORE): blx03 job 000 ran at 07:29:16 that way.
  Never kill other tenants' jobs. Skip any box whose broker is being upgraded.
- **`hostfmax.py 1150` is not a guard against drops.** ltx-host job 995 ran at aiclk 1150 and still dropped
  tray 1 (see tt-project/research/blx03_drop_0722.md). Rely on the rules instead: open the full mesh, then
  `create_submesh(2,4)`; one project job at a time per box; no 4x8 runs.
