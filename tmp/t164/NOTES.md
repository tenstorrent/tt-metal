# t164 — t95 eval pack on g15blx02 (continues #140)

Box: g15blx02 (READY marker; ~/fasth3 93.8 GiB at start, 100 GB cap). Never blx03.
Code: Python from worktree tt-project/worktrees/t164 (branch ttp/t164-eval-pack-g15 = t48 abfd309e797 merged in;
models/ identical to t140's overlay), C++ build + kernels from worktrees/t158 (bf7db12a149), caches data/g15
(job 399's). No knob writes a new weight cache (checked: gate-on-device folds after loading the plain
transformer/ cache; exact_shard/LoFi leave the VAE C_in_block hash alone; adaln/agmm/hostcopy are runtime only).
Only new JIT kernels land in data/g15/tt-metal-cache.

Per config: one broker job (run_cfg.sh), warmup + gen#0 (capture, default rapper prompt) + gen#1 (warm, paper
boat, headline) + gen#2 (warm, fisherman, spread). -t 240 baseline / 400 knobs (job 406, warmup cuts, held 152 s). Warmup cuts on: LTX_WARMUP_T2V_ONLY=1 LTX_WARMUP_ENCODERS=0.
pytest runs with cwd = the config's output dir: tt-metal resolves relative kernel paths against the cwd before
TT_METAL_HOME, so running inside the t164 checkout (job 403) recompiled all 2172 kernels from t164 paths.
Driver: tmp/t164/driver.sh (CONFIGS, TAG), started with `ttp detach t164-<tag>`. Marker data/g15/t164/driver_<tag>/DRIVER.done.
Results: tt-project/data/g15/t164/<label>/; summary data/g15/t164/summary_configs.md (post.py, PCC/PSNR vs baseline same gen).

Phase 2 is automated by tmp/t164/after_pack.sh (ttp detach t164-phase2): it waits for the pack marker, picks
the 1-2 fastest configs (mean gen1/gen2 e2e < baseline - 20 ms, PCC >= 0.95 and PSNR >= 25 dB vs baseline on
gen1/gen2), writes data/g15/t164/phase2/configs5.txt (baseline5 + <cfg>5 with LTX_E2E_SEEDS=0,1,2,3,4
LTX_FRESH_PROMPTS=0 LTX_E2E_EXTRA_REPLAYS=0: default prompt, gen#N = seed N-1, the ref_dv145 protocol), runs the
TAG=seeds driver, then VBench per label vs ref_dv145 (<label>/vbench.log, <label>/vbench/). Marker
data/g15/t164/phase2/PHASE2.done ("<code> <reason>"; 20 = no config qualified, 11/12/13 = driver problem),
log phase2.log. 5-seed gens reuse one prompt, so their times are not headline numbers (encode may be cached);
the headline is the pack's fresh-prompt gen1/gen2.
Next run after PHASE2.done: read summary_configs.md, summary_configs5.md, the VBench BATCH lines, look at the
stills, pick the default set, commit the table + video/still paths on this branch, then `ttp push` to t48.

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
- Job 407 confirmed the cwd fix: 0 JIT compiles, gemma encode 2 s, first S1 step 5667 ms, gen#0 35.06 s.
- JIT cleanup: deleted the 1646 job-403-only hash dirs (state/runs/684/jit403_dirs.txt; rechecked per dir that
  every file was from 21:14-21:22). ~/fasth3 100662 -> 97151 MiB. In-place ELF rewrites in older dirs left.
- DROP 2026-10-06 21:34:29 UTC, g15blx02, broker job 407 (ours, t164 baseline), tray 1 = chips 0-7 off the
  PCIe bus after gen#0. The broker killed the holders and ran its TRAY_DOWN_NO_WINDOW recovery (incident
  /var/lib/tt-device-broker/health/incidents/20261006T213429Z_unhealthy_none). Driver: DROP #1, it reruns the
  baseline after the health gate passes. A second baseline drop stops the pack (marker 6).
- 2026-10-06 21:40 UTC: phase 2 started (ttp detach t164-phase2; log/rc in state/runs/684/t164-phase2.{log,rc}).
- 2026-10-06 ~21:40 UTC: host crashed (unclean: the old boot's journal ends 21:40:04, our files were written
  21:40:13, the box booted 21:42:57). This was during the broker's 60 s settle after its tray-1 reset sweep, before
  phase 2 launched. My host-side `pytest --collect-only` dry-run (outside the broker) was starting then.
  Collection only checks a file path in skipif, and conftest's cluster queries live in fixtures, so it most likely
  did not touch the chips, but its `import ttnn` cannot be excluded. Rule from now on: no host-side ttnn imports
  while a broker recovery runs, and no dry-runs outside the broker.
- 21:55 UTC: the broker finished its reset + verify (all 32 chips OK, fabric OK; logged at ERROR level as
  HOLD-DEADLINE-ESCALATE). driver.sh health() now accepts that line.
- 21:56 UTC: pack relaunched on t48 abfd309e797 + warmup cuts (TAG=pack3, ttp detach t164-pack4, run dir
  state/runs/690, PRIOR_DROP=baseline). Baseline = job 422 (-t 240), queued behind ltx-host 421. Phase 2: ttp
  detach t164-phase2b (PACK_TAG=pack3). Wake probe: `ttp detach --check state/runs/690/t164-phase2b.rc`.
- 2026-10-06 22:20 UTC: host rebooted. Detached pack4/phase2b drivers are gone; no PHASE2.done. Broker job 422
  (baseline) is still queued in the broker (survived), behind a broker glx_reset health gate (22:23 UTC). Next:
  relaunch the driver so it adopts job 422 (do NOT submit a duplicate baseline) and then runs the remaining configs.
- 2026-10-06 22:24 UTC: the host reboot at ~22:21 was the broker's own power-cycle (job 434, 22:20:55 UTC,
  "idle escalation: 32/32 chips off") after its tray-1 (chips 0-7) bridge-resets 427/429/431 and glx_reset gates
  426/428/430/432 failed from 21:58 on. Broker glx_reset gate 439 started 22:22:59. Job 422 still queued.
  driver.sh gained ADOPT=<label>:<job> (watch a job a killed driver left queued; if the broker never runs it,
  submit a fresh one without counting a drop). Relaunched as TAG=pack3 PRIOR_DROP=baseline ADOPT=baseline:422.
- DROP 2026-10-06 21:57:24 UTC, g15blx02, broker job 421 (ltx-host, live service; not ours), tray 1 = chips 0-7 off
  the PCIe bus again (2nd tray-1 drop in 23 min after our job 407). Recovery failed until the 22:20:55 power-cycle.
- 22:25 UTC: job 422 started running 22:25:10 (after gate 439 passed). Relaunched: ttp detach t164-pack5 (TAG=pack3,
  PRIOR_DROP=baseline, ADOPT=baseline:422, driver pid 16894) and t164-phase2c (PACK_TAG=pack3); run dir
  state/runs/695. Wake probe: `ttp detach --check state/runs/695/t164-phase2c.rc` (fires when phase 2 ends, or at
  once if the pack driver stops early). A third tray-1 drop on a baseline job stops the pack (marker 6): then move
  the pack to blx01 (/var/tmp/fasth3, t48 bf7db12a149 build from #159; its tray 3 config skip does not apply here).
