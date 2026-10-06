# t164 — t95 eval pack on g15blx02 (continues #140)

Box: g15blx02 (READY marker; ~/fasth3 93.8 GiB at start, 100 GB cap). Never blx03.
Code: Python from worktree tt-project/worktrees/t164 (branch ttp/t164-eval-pack-g15 = origin t48 b81bb403d86;
models/ identical to t140's overlay), C++ build + kernels from worktrees/t158 (bf7db12a149), caches data/g15
(job 399's). No knob writes a new weight cache (checked: gate-on-device folds after loading the plain
transformer/ cache; exact_shard/LoFi leave the VAE C_in_block hash alone; adaln/agmm/hostcopy are runtime only).
Only new JIT kernels land in data/g15/tt-metal-cache.

Per config: one broker job (run_cfg.sh), warmup + gen#0 (capture, default rapper prompt) + gen#1 (warm, paper
boat, headline) + gen#2 (warm, fisherman, spread). -t 360 baseline / 400 knobs (job 399 held 162 s).
Driver: tmp/t164/driver.sh (CONFIGS, TAG), started with `ttp detach t164-pack`. Marker data/g15/t164/driver_pack/DRIVER.done.
Results: tt-project/data/g15/t164/<label>/; summary data/g15/t164/summary_configs.md (post.py, PCC/PSNR vs baseline same gen).

Next after the marker: read summary, pick 1-2 best, write configs5.txt (baseline5 + best5 with
LTX_E2E_SEEDS=0,1,2,3,4 LTX_FRESH_PROMPTS=0 LTX_E2E_EXTRA_REPLAYS=0, default prompt = ref_dv145's), run
TAG=seeds driver, then ltx_eval batch --vbench-ref vs ref_dv145 seeds, visual check, commit summary + stills
on ttp/t164-eval-pack-g15, ttp push.
