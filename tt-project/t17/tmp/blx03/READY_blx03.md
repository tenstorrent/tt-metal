# READY (blx03): LTX-2.5 1080p 6s baselines on g14blx03

Adapted from tmp/READY_26.md (g15blx02). dv153 is left out (suspect: job 689 took g15blx02 down).

Tree: blx03:~/fasth3/tt-metal, branch ttp/t36-blx03-ltx25 (t10/t26 line 9287c953eb8 + t22 merged, NA fix d079ee7cd11 included).
Build: build_Release (Release only). Venv: ~/fasth3/tt-metal/python_env.
Caches (under $FASTH3_DATA=/var/tmp/fasth3): DiT /var/tmp/fasth3/cache/dit-ltx25, kernels /var/tmp/fasth3/cache/tt-metal-cache, Gemma /var/tmp/fasth3/models/gemma-3-12b-it-qat-q4_0-unquantized.
Weights: LTX-2.5 split files on /mnt/MLPerf (weka, shared with g15blx02); 2.3 monolith (VAE config) in ~/.cache/ltx-checkpoints.
Outputs: ~/fasth3/out/ltx25_1080p_6s/<label>/{run.log,ltx_av_fast_*.mp4}

Env: submit.sh passes `-e tmp/blx03/env.yaml` to the broker (venv, TT_METAL_HOME, kernel cache). Without it a caller
that has no venv active (nohup/setsid drivers, plain ssh) gets rejected: "Python env not found at .../tt-metal/tt-metal/python_env".
A worktree with its own env passes its own yaml (see ~/fasth3/t32/tmp/env.yaml).

Smoke (job 874, dv145, warm caches, seed 0): S1 2.27s, S2 2.50s, DiffVAE decode 11.77s, audio 0.39s, compute 19.02s,
E2E_WALL_S 19.81s; whole broker job 405s (load ~60s + warmup gen ~50s + timed gen). Budget ~600-900s per job.

Rules: one project device job at a time (submit.sh refuses while a smarton job is running/queued), short jobs, never reset.

From g15blx02 (one job, then wait for its final status before the next):

| # | label | command |
|---|-------|---------|
| 1 | dv145 | `ssh g14blx03 "~/fasth3/tt-metal/tmp/blx03/submit.sh 900 bash tmp/blx03/run25.sh dv145"` |
| 2 | conv145 | `ssh g14blx03 "~/fasth3/tt-metal/tmp/blx03/submit.sh 900 bash tmp/blx03/run25.sh conv145 LTX25_DIFFVAE=0"` |
| 3 | dv145_c211 | `ssh g14blx03 "~/fasth3/tt-metal/tmp/blx03/submit.sh 900 bash tmp/blx03/run25.sh dv145_c211 DIFFVAE_NA_CHUNK_BRICKS=2,1,1 DIFFVAE_NA_UNSAFE_CHUNK=1"` |

Check: `ssh g14blx03 tt-device-mcp status -j <id>`; pass = run.log ends " passed" and RUN_EXIT[<label>]=0.
Logs: `ssh g14blx03 tt-device-mcp logs <id>`.
Per-stage times: `grep -E "LTX_TIME|stage|decode|export|load-cache" ~/fasth3/out/ltx25_1080p_6s/<label>/run.log`.
Still: `ffmpeg -ss 3 -i <mp4> -frames:v 1 <label>.png`.
Other trees: `W=<tree with its own build> bash tmp/blx03/run25.sh ...`; a python-only worktree can symlink build_Release,
runtime and ttnn/ttnn/_ttnn.so from ~/fasth3/tt-metal (as t22 did on g15blx02).

## Device stop procedure and drop guards (added 2026-10-01, #64)

- **At every device stop** (pause, chip drop, reboot, fabric failure): on each box, list our own queued
  broker jobs (`tt_device_queue_status` / `tt_device_recent_jobs`) and kill each with `tt_device_job_kill`.
  The broker re-queues jobs after a reboot (QUEUE-RESTORE): blx03 job 000 ran at 07:29:16 that way.
  Never kill other tenants' jobs. Skip any box whose broker is being upgraded.
- **`hostfmax.py 1150` is not a guard against drops.** ltx-host job 995 ran at aiclk 1150 and still dropped
  tray 1 (see tt-project/research/blx03_drop_0722.md). Rely on the rules instead: open the full mesh, then
  `create_submesh(2,4)`; one project job at a time per box; no 4x8 runs.
