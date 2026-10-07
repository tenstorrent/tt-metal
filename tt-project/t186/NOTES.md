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

## Results so far (attempt 2, 2026-10-07)
- No drops. s1x6 = blx01 job 760, s1x5 = job 761. Warm gens (seeds 0-4):
  - s1x6: 5.209 5.181 5.180 5.175 5.167, mean 5.182 s (-0.489 s vs ref 5.671)
  - s1x5: 4.939 4.933 4.950 4.914 4.930, mean 4.933 s (-0.738 s)
- post.sh scoring failed: sbs_*.mp4 sat next to seed*.mp4 and ltx_eval has no reference for them.
  score.sh moves sbs_* into sbs/ and reruns ltx_eval (PCC/PSNR + VBench). Detached as run 783
  t186score (state/runs/783/t186score.{log,rc}); per-arm marker data/g15/t186_<label>/SCORE.done.
- Stills: data/g15/t186_<label>/stills_5seeds_t3s.png (rows = seeds 0-4, ref | cand at t=3 s).
  First look: same subject/setting, layout drifts from ref (more for s1x5). s1x6 seed0 face smeared at t=3 s.
  Check more frames (eval_vs_ref_t48_f6b8/seed*_cmp_f*.png) before a verdict.
- Next: read eval summary for both arms, compare VBench to ref, decide gate, write result.

## Verdict (attempt 3, 2026-10-07)
- Scores vs ref_t48_f6b8 (S2 still 3 steps in both arms and ref):
  - s1x6: PCC 0.447, PSNR 15.6 dB; VBench subj 0.894 (ref 0.888), bg 0.921 (0.921), imaging 0.562 (0.559),
    aesthetic 0.576 (0.587), motion 0.983 (0.984).
  - s1x5: PCC 0.350, PSNR 15.1 dB; aesthetic 0.558 (-0.029), others within noise.
  - Low PCC/PSNR = different trajectory (layout/pose drift), not noise or blur by itself.
- Visual: seeds 1-4 of s1x6 look as clean as ref over the whole clip (strip at 1.5 fps:
  data/g15/t186_s1x6/seeds1-4_ref_vs_s1x6_strip.png). Seed 0 has a visible face smear/ghost for about
  0.2 s around t=3.0 s in BOTH s1x6 and s1x5; ref is clean there
  (data/g15/t186_s1x6/seed0_ref_s1x6_s1x5_t2.6-3.4.png). s1x5 drifts further from ref and loses aesthetic.
- Call: s1x5 rejected. s1x6 is NOT made default: 1 of 5 seeds shows a visible transient artifact that the
  8-step run lacks, and the charter forbids visible degradation. It stays opt-in via LTX_S1_SIGMAS (env only,
  no code change). Worth one more try with a different 6-step schedule and stacked on S2x2 (current t48 default).
- Push: notes commits stay local. `ttp push` check runs models/tt_dit/tests/unit/test_ltx_*.py, which does not
  exist on fasth3-opt, so every push of this branch fails the check (harness config, not ours to edit).
