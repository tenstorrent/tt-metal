# t164 — t95 eval pack on g15blx02 (continues #140)

Box: g15blx02 (READY marker; ~/fasth3 93.8 GiB at start, 100 GB cap). Never blx03.
Code: Python from worktree tt-project/worktrees/t164 (branch ttp/t164-eval-pack-g15 = origin t48 b81bb403d86;
models/ identical to t140's overlay), C++ build + kernels from worktrees/t158 (bf7db12a149), caches data/g15
(job 399's). No knob writes a new weight cache (checked: gate-on-device folds after loading the plain
transformer/ cache; exact_shard/LoFi leave the VAE C_in_block hash alone; adaln/agmm/hostcopy are runtime only).
Only new JIT kernels land in data/g15/tt-metal-cache.

Per config: one broker job (run_cfg.sh), warmup + gen#0 (capture, default rapper prompt) + gen#1 (warm, paper
boat, headline) + gen#2 (warm, fisherman, spread). -t 330 baseline / 400 knobs (job 399 held ~180 s, 0 JIT compiles).
pytest runs with cwd = the config's output dir: tt-metal resolves relative kernel paths against the cwd before
TT_METAL_HOME, so running inside the t164 checkout (job 403) recompiled all 2172 kernels from t164 paths.
Driver: tmp/t164/driver.sh (CONFIGS, TAG), started with `ttp detach t164-<tag>`. Marker data/g15/t164/driver_<tag>/DRIVER.done.
Results: tt-project/data/g15/t164/<label>/; summary data/g15/t164/summary_configs.md (post.py, PCC/PSNR vs baseline same gen).

Next after the marker: read summary, pick 1-2 best, write configs5.txt (baseline5 + best5 with
LTX_E2E_SEEDS=0,1,2,3,4 LTX_FRESH_PROMPTS=0 LTX_E2E_EXTRA_REPLAYS=0, default prompt = ref_dv145's), run
TAG=seeds driver, then ltx_eval batch --vbench-ref vs ref_dv145 seeds, visual check, commit summary + stills
on ttp/t164-eval-pack-g15, ttp push.

## Run log
- 2026-10-06 21:15 UTC: pack driver started (ttp detach t164-pack, driver pid 3478924; log/rc in
  tt-project/state/runs/669/t164-pack.{log,rc}). Baseline = broker job 403, queued behind ltx-host 402.
  ~/fasth3 94.7 GiB at start. Wake probe: `ttp detach --check .../runs/669/t164-pack.rc`.
- If the driver dies while a job runs, wait for that job to end before relaunching (the skip check runs
  before the wait, so a relaunch during a live job would submit a duplicate).
- 2026-10-06 21:21 UTC: job 403 (baseline) failed, not a drop: pytest timeout 330 s hit at the end of warmup
  (audio-decode warmup not reached). Cause: cwd = t164 checkout -> 2172 JIT compiles (gemma encode 58 s vs 2 s,
  first S1 step 89 s vs 5.6 s), +3.6 GB in data/g15/tt-metal-cache (1646 new hash dirs, 1.75 GB by du, plus
  in-place rewrites of ~3200 ELFs in existing dirs). ~/fasth3 went 94.7 -> 98.1 GiB. Not deleted: the hash
  differs with the resolved path, but some new dirs may be t164-python variants the pack needs; list of the
  1646 dirs (all files 21:14-21:22) kept in state/runs/684/jit403_dirs.txt for cleanup after the pack.
  run.log archived as data/g15/t164/driver_pack/run_job403.log.gz.
- 2026-10-06 21:32 UTC: fixed run_cfg.sh (cwd = $OUT, absolute test path; dry-run collect 1/8 OK, models from
  t164). Pack driver relaunched as TAG=pack2 (ttp detach t164-pack2, pid 3798926; log/rc in
  state/runs/684/t164-pack2.{log,rc}). Baseline = job 407, started 21:32 right after #166's job 406 (152 s).
  Wake probe: `ttp detach --check .../runs/684/t164-pack2.rc`. Marker data/g15/t164/driver_pack2/DRIVER.done.
