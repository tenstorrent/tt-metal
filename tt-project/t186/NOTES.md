# t186: S1 step cut 8->6 (and 8->5) on LTX 2.5 (plan t-denoise task 2)

## Setup (blx01, all under /var/tmp/fasth3/t186)
- Python: t171 overlay tree (t48 f6b806516cc; t48 head 87e4b0732df only deletes notes, so code is identical).
  C++/JIT cache /var/tmp/fasth3/t48. Same stack as ref_t48_f6b8 (job 710).
- run_cfg.sh = t185's with T=t186. Arms (one broker job each, -t 240, PYTEST_S=220):
  - s1x6: LTX_S1_SIGMAS=1.0,0.9875,0.975,0.909375,0.725,0.421875,0.0
  - s1x5: LTX_S1_SIGMAS=1.0,0.975,0.909375,0.725,0.421875,0.0 (optional arm of the spec)
  plus LTX_FRESH_PROMPTS=0 LTX_E2E_SEEDS=0,1,2,3,4 (gen#N = seed N-1), bf16, no LTX_QUANT.
- Judgment: default warmup, not LTX_WARMUP_T2V_ONLY=1/LTX_WARMUP_ENCODERS=0. That config dropped
  blx01 tray 3 twice (jobs 625, 640) and is skipped there; job 710 (the reference) used the default
  warmup (161.8 s), so 240 s fits and the comparison is like for like.

## Drops seen
- 2026-10-07 04:46:11 UTC, blx01, job 715 (smarton, t185 s2x2, not ours), chips 16-23 (tray 3) fell off
  PCIe; broker SIGKILLed it and started recovery (bridge-reset failed, glx_reset).

## Driver (g15blx02, detached: run dir t186drv.{log,rc})
- driver.sh s1x6=... s1x5=...: waits for blx01 ready (no broker job, not HELD, no smarton job; 2 passes),
  submits under `ttp lock blx01-device`, reruns a dropped arm once, skips it after 2 drops, then
  post.sh <label> in background -> tt-project/data/g15/t186_<label>/ (seed<N>.mp4, sbs_seed<N>.mp4
  ref|cand, sbs stills, eval_vs_ref_t48_f6b8/ PCC/PSNR + VBench 5 dims, POST.done).
- Log: t186/driver.log; marker t186/DRIVER.done = "<rc> <reason>".
- Next: read E2E_WALL_S gen#1..5 from data/g15/t186_<label>/run.log, eval summary, look at stills.

## Attempt 2 (2026-10-07)
- Attempt 1's driver failed both submits: tt-device-mcp looked for python_env under $WS/tt-metal. Fix: pass
  `-e /var/tmp/fasth3/t159/env.yaml` (PYTHON_ENV_DIR=/var/tmp/fasth3/t48/python_env, as t185 job 758).
- Rerun detached as run-dir t186drv2; s1x6 submitted as blx01 job 760 at 05:58:51 UTC; s1x5 follows.
