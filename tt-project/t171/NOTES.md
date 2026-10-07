# t171: 5-seed 4x8 1080p 6s e2e confirm of the new t48 defaults (continues #142)

Update 2026-10-07 00:26 (coordinator): run t48 f6b806516cc with NO knob env vars, seeds 0-4,
LTX_FRESH_PROMPTS=0, compare byte-for-byte with #170 fast5, store as new ltx_eval reference.

## Setup (blx01 = g15blx01, all under /var/tmp/fasth3/t171)
- C++ build, kernels, JIT cache: /var/tmp/fasth3/t48 @ bf7db12a149 (c4409b1fa24 + conv3d host guard TT_FATAL).
  f6b806516cc has no C++ diff vs c4409b1fa24, so only the guard differs (reject-only, not hit by the default config).
- Python: t171/tree = hardlinked t48 models/ with the 13 files changed bf7db12a149..f6b806516cc swapped in
  (mkoverlay.sh; tree/OVERLAY_COMMIT = f6b806516cc). Same scheme as #170's t170/tree.
- Job: run_cfg.sh (t170's, renamed markers). Default warmup (job 621/#170's), warm JIT cache.
- Encode: traced 4x8 path (dynamic_load off) runs Gemma every gen; no prompt-embedding cache hit.
- Sizing: job 708 (fast5, same shape) 162 s, job 707 205 s -> -t 310, PYTEST_S=290.

## Jobs
- blx01 broker job 710, queued 00:28:21 UTC behind ltx-host 709, started ~00:28:35.
  Output: /var/tmp/fasth3/t171/res/def5/{run.log, ltx_av_fast_1920x1088_<g>.{mp4,json}} (g = seed+1; g0 = cold seed 0).

## Post (g15blx02, CPU)
- Detached waiter (run 709, t171post.{log,rc}) runs post.sh after the job ends: copies seeds to
  tt-project/data/g15/ref_t48_f6b8/seed<N>.{mp4,json} + run.log, identity.txt (cmp vs t170 fast5),
  seed0_t3s.png; ltx_eval PCC/PSNR vs fast5 and ref_dv145 only if any seed differs.
- Next: read identity.txt, per-seed E2E_WALL_S (gen#1..5) + stage tables from run.log, register the reference
  (baselines/ltx25_1080p_6s/ref_t48_f6b8 symlink + meta.json), hand off.
