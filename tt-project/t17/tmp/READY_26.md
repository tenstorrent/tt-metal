# READY_26: LTX-2.5 1080p 6s baselines, ready to submit (DO NOT submit during the device pause)

Code: branch ttp/t26-fix-ltx-2-5-dit-cache-key-instability-th @ d079ee7cd11; t10 worktree fast-forwarded to it.
Caches (survive reboot, root fs): /var/tmp/t10-dit-cache-ltx25 (73G, all 10 DiT entries warm),
/var/tmp/t10-tt-metal-cache (6.2G). The kernel cache misses only the fixed neighborhood_sdpa kernels; they JIT on first use (~30s).
Rule: one job at a time. Submit the next one only after the previous job's status is final.
Outputs: tt-project/baselines/ltx25_1080p_6s/<label>/{run.log,ltx_av_fast_*.mp4}

Run from /home/smarton/fasth3/tt-metal/tt-project/worktrees/t10:

| # | label | command | expected runtime |
|---|-------|---------|------------------|
| 1 | dv145 | `tt-device-mcp run-bg "bash tmp/run25.sh dv145" -w $PWD -e tmp/ltx25_env.yaml -t 600` | ~4-5.5 min (load ~2 min, warmup ~1.5 min, warmup decode + NA JIT ~0.7 min, gen#1 ~0.5 min) |
| 2 | conv145 | `tt-device-mcp run-bg "bash tmp/run25.sh conv145 LTX25_DIFFVAE=0" -w $PWD -e tmp/ltx25_env.yaml -t 600` | ~4-5 min; first run may add ~1 min converting the conv decoder (no cache entry for it yet) |
| 3 | dv145_c211 | `tt-device-mcp run-bg "bash tmp/run25.sh dv145_c211 DIFFVAE_NA_CHUNK_BRICKS=2,1,1 DIFFVAE_NA_UNSAFE_CHUNK=1" -w $PWD -e tmp/ltx25_env.yaml -t 600` | ~4-5 min |
| 4 | dv153 | `tt-device-mcp run-bg "bash tmp/run25.sh dv153 NUM_FRAMES=153 FPS=25" -w $PWD -e tmp/ltx25_env.yaml -t 600` | ~5-6 min. SUSPECT: user says job 689 (this config) took the box down at 16:22. Submit only with the user's OK. |

Each job: pytest cap 580s, broker cap 600s, then ~1-1.5 min broker health gate.
Check: `tt-device-mcp status -j <id>`; pass = run.log ends " passed" and RUN_EXIT[<label>]=0.
Read: `grep -E "LTX_TIME|stage|decode|export|load-cache" run.log` for the per-stage table.
Still: `ffmpeg -ss 3 -i <mp4> -frames:v 1 <label>.png`.

Past failures, all fixed or external:
- 596/605: DiT cache misses (capture-only pass never writes; /tmp wiped at boot). Fixed: caches on /var/tmp, f2ddefed262/3f159a235a5.
- 671 dv145, 687 dv145_c211: neighborhood_sdpa.cpp JIT compile error (stale num_blocks arg). Fixed d079ee7cd11,
  verified off-device with tmp/na_kernel_compile.sh (job 671's exact riscv g++ commands: FAIL before, OK after).
- 640, 689: killed by broker device recovery (chips left PCIe). Box instability, not the job.

## Device stop procedure and drop guards (added 2026-10-01, #64)

- **At every device stop** (pause, chip drop, reboot, fabric failure): on each box, list our own queued
  broker jobs (`tt_device_queue_status` / `tt_device_recent_jobs`) and kill each with `tt_device_job_kill`.
  The broker re-queues jobs after a reboot (QUEUE-RESTORE): blx03 job 000 ran at 07:29:16 that way.
  Never kill other tenants' jobs. Skip any box whose broker is being upgraded.
- **`hostfmax.py 1150` is not a guard against drops.** ltx-host job 995 ran at aiclk 1150 and still dropped
  tray 1 (see tt-project/research/blx03_drop_0722.md). Rely on the rules instead: open the full mesh, then
  `create_submesh(2,4)`; one project job at a time per box; no 4x8 runs.
