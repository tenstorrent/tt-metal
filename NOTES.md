# t158 — g15blx02 as a second 4x8 LTX-2.5 box (state at hand-off of run 645)

Done:
- Worktree tt-project/worktrees/t158 on ttp/t158-g15blx02-ltx25-4x8 at bf7db12a149 (same hash as blx03 t48: c4409b1fa24 + t149 TT_FATAL guard).
- Broker on g15blx02 healthy at 20:11 UTC (32 chips, HEALTH-GATE OK). Live ltx-host runs ~19 min jobs back to back; ours queues behind.
- VAE source: /mnt/MLPerf .../LTX-2.3/.../ltx-2.3-22b-dev.safetensors. Its 170 vae.* tensors (1.45 GB) and VAE config are
  byte-identical to blx03's ltx-2.3-22b-distilled-1.1 monolith (checked on blx03), so no local copy. The 2.5 test path uses
  LTX_CHECKPOINT only for the skipif; GEMMA_PATH is unused (Gemma-4 comes from LTX25_ROOT).
- Disk: freed build_Release of done tasks t60/t134 and removed clean, pushed worktrees t60, t81, t133, t149 (branches kept).
  ~/fasth3 was 20 GB before staging. Plan: +72 GB caches, +2.2 build, +~1.4 VAE caches, +JIT -> ~97 GB (< 100).
- Detached (ttp detach, run dir state/runs/645):
  - t158-stage: rsync of the 8 4x8 cache dirs from g14blx03:/var/tmp/fasth3/cache/dit-ltx25 (read-only there) to
    tt-project/data/g15/dit-ltx25. Script tt-project/data/g15/stage.sh. Done when t158-stage.rc exists (0 = ok).
  - t158-build2: tmp/t158/build.sh (submodule init + ./build_metal.sh with python_env). Done when t158-build2.rc exists.
  - (t158-build, the first attempt, failed: submodules missing. Ignore it.)

Next (one run):
1. Check both rc files are 0 and `du -sh ~/fasth3` < 100 GB.
2. A PreToolUse hook (reservation_cap.py) blocks broker timeouts > 600 s. The job's first run is cold (JIT compile of all
   kernels + VAE cache build), so prewarm kernels first: tt_metal/tools/kernel_prewarm/prewarm_and_submit.sh (README there),
   with TT_METAL_CACHE=tt-project/data/g15/tt-metal-cache. Then submit with -t 600 (blx03 job 099 used 2400; size from its
   measured device-held time).
3. Start tmp/t158/driver.sh with `ttp detach` (waits for build/stage, health-checks, submits ONE job via run-bg, watches it,
   checks broker log for drops, writes tt-project/data/g15/driver/DRIVER.done = "<code> <reason>").
4. After the job: stage table from tt-project/data/g15/out/run.log, ltx_eval PCC/PSNR of gen#0 vs
   tt-project/baselines/ltx25_1080p_6s/ref_dv145 seed0 (ref used DiffVAE) and vs tt-project/t-e2e/t138 job 099 outputs;
   still frames *_t3s.png. Only on success write tt-project/state/ready/g15blx02.READY.

## Light wake 2026-10-06
- Both rc files are 0 (runs/645/t158-stage.rc, t158-build2.rc). `du -sh ~/fasth3` = 96G (<100). /home 1.5T free.
- Not started: prewarm. prewarm_and_submit.sh needs a capture stage that is itself a device job (cold cache), so the
  prewarm + driver flow needs a standard run to wire up (driver.sh submits run_e2e.sh directly with -t 600).

## Standard wake 2026-10-06 (run 654)
- Job 099 compiled 2569 kernels in-window (process wall 530 s), so a cold cache does not fit the 600 s cap in one job.
- driver.sh now: capture-only job (TT_METAL_KERNEL_PREWARM=1 TT_METAL_KERNEL_CAPTURE_ONLY=1, OUT=data/g15/capture)
  -> off-device build_Release/tools/kernel_prewarm into data/g15/tt-metal-cache (log data/g15/driver/prewarm.log)
  -> disk check (<100 GB) -> real job (bash tmp/t158/run_e2e.sh, -t 600). Each job is followed by the broker/kernel
  drop check. Job ids in data/g15/driver/job_id_{capture,run}; marker data/g15/driver/DRIVER.done.
- Started with `ttp detach t158-driver` (run dir state/runs/654).
Next: on DRIVER.done "0 ...": stage table from data/g15/out/run.log, ltx_eval vs ref_dv145 seed 0 and vs t138 gen#1,
still frame, write state/ready/g15blx02.READY. On 9: log the drop (broker_slice/drop_evidence), rerun once.

## Standard wake 2026-10-06 (run 656)
- Capture job 397 rc 0 (20:36-20:40 UTC, no drops); offline prewarm built 3653 targets; driver then stopped rc 14 (~/fasth3 at 100 GiB).
- Freed ~/fasth3/tt-metal/.cpmcache (3 GB, project-created 09-30, re-downloadable). ~/fasth3 now 96 GiB.
- Old marker moved to data/g15/driver/DRIVER.done.rc14. Driver restarted: `ttp detach t158-driver2` (run dir state/runs/656);
  it reruns the (cached) prewarm, checks disk, then submits the warm e2e job (job id -> data/g15/driver/job_id_run).
Next: same as run 654's "Next" (stage table, ltx_eval vs ref_dv145 seed 0, still frame, READY marker).
