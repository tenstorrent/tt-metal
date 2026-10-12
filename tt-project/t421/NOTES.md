# t421: Turbo v1.0 host AdaLN precompute, off/on A/B seeds 0-4 (blx01)

## Code
- d4494e2c1a1 `minimax-h3 turbo: host AdaLN table for a fixed single-time schedule`.
  `MiniMaxH3TurboPipeline(adaln_num_inference_steps=N)` commits the pipeline to N steps: weight mode
  precomputed_adaln, single-time table (two_time None), cache key "single-time" + the sigma grids.
  Any other step count outside warmup raises. Env MINIMAX_H3_ADALN_PRECOMPUTE=0 turns it off.
  The e2e test passes N only when MINIMAX_H3_ADALN_PRECOMPUTE=1 (opt-in for Turbo).
- Host tests: models/tt_dit/tests/unit/test_minimax_h3_adaln_precompute.py, 22 pass (4 new).

## Device plan (blx01 broker)
- Tree: blx01 /var/tmp/fasth3/t421/b (worktree of t48's repo at d4494e2c1a1, t284 build hard-linked).
- Drivers: run421.sh (one arm per job), specs in tt-project/state/t421/specs, chain421.sh submits
  on-cache, then off-sN, on-sN for N=0..4, one job at a time via t288/blx01-submit.sh.
- Judgment call: both arms do not fit one job. A t400 arm took 272-326 s, so a pair is ~600 s and
  measured +50% would need ~900 s against the 600 s cap. The cap wins: arms run as adjacent jobs,
  same box and clock. Arm jobs -t 500 (py 470), cache job -t 300 (t400 measured 189 s).
- post421.sh (detached on g15blx02): after the chain, VBench batch (on vs off, --vbench-ref) via
  ltx_eval exported to tt-project/state/t421/evaltree. Marker tt-project/state/t421/post.marker.

## Status 2026-10-11 22:55 PDT
- Chain launched (run 1574, pid 1931564), log tt-project/state/t421/chain.log.
- blx01 broker HELD (degraded): chip 11 off PCIe bus since 22:42 PDT (05:42Z), after smarton t408
  job 839; fabric-check 840 failed, hold 841. Not a t421 job. Chain waits for the broker to clear.

## Next
- After post.marker: read state/t421/summary.md (s/step per arm, PSNR/PCC, stills) and vbench.log,
  look at stills, recommend default on/off. Then delete on blx01:
  /var/tmp/fasth3/cache/dit-h3hf/minimax-h3/transformer_precomputed_adaln (~38 GB), the t421 table in
  cache/dit-h3hf/minimax-h3-adaln, and /var/tmp/fasth3/t421.

## Status 2026-10-12 00:10 PDT (run 1603)
- on-cache: job 850 completed (194 s). off-s0: job 861 failed at the 400 s warm deadline in
  "Warming audio decode lengths" (4/12 done, ~60 s each). Cause: cold JIT kernel cache for the new
  tree path (JIT stats 178/1370 hits, 13%); kernel hashes depend on TT_METAL_HOME. Not a drop.
- The chain then hit the new device-lint setsid rule (harness a8268f2) for on-s0/off-s1 and stopped
  (3 skips). No device time used by those.
- Fixes: setsid re-exec block removed from run421.sh (pytest already under timeout --foreground -k 10);
  chain called analyze400.py, now analyze421.py; off-s0/on-s0 specs at -t 600 (unmeasured cold JIT),
  py 570, warm deadline 520. Other seeds stay at -t 500 / deadline 400.
- Relaunched chain (off-s0 .. on-s4) and post421.sh, both detached in run dir 1603.

## Status 2026-10-12 01:15 PDT (runs 1612, 1615)
- off-s0 job 868 (-t 600, deadline 520) failed again on cold JIT: deadline passed in "Warming prompt
  encoder keyframe layouts" (JIT hits 1370/3446, 40%; more kernels cached each try). Not a drop.
- Then every submit was refused by lint: blx01 /var/tmp/fasth3 at 455 GB, over its 450 GB cap.
  Chain stopped (T421_DONE 8), post.marker POST421_DONE 1 (no pairs).
- Drop (not ours to fix): 2026-10-12 07:30:54Z blx01, chip 24 left the PCIe bus right after our
  job 868 ended (hold 871, bridge-resets 872/873 failed, recovered 00:42 PDT, fabric-check 878 ok).
- Freed space: prunejit421.sh deletes JIT kernel cache entries whose source tree is gone (the hash
  includes the tree path, so they can never hit). Removed 22.6 GiB (t400/b, t288/b, t423/b, t427/b).
  blx01 / now 165 GB free.
- g15blx02 rebooted ~01:03 PDT and killed the run. Relaunched chain (off-s0..on-s4) and post421.sh
  detached in run 1615.

## Status 2026-10-12 05:05 PDT (run 1658)
- Results of the run-1615 chain: off-s1 (job 932) and off-s2 (job 938) completed; off-s0 failed twice
  on cold JIT (jobs 868, 907: T288_EXIT=124 at 571 s, still compiling). Every on arm failed after
  warmup (on-s0 924/929, on-s1 934/936, on-s2 940) with "ValueError: this pipeline precomputes AdaLN
  for num_inference_steps=5; num_inference_steps=3 cannot be honoured".
- Cause: run_warm_generation (common_av.py) warms with 3 steps unless the pipeline has a HyperFlow
  contract. Fixed in d943ff0dfe5: it also keeps the step count when adaln_num_inference_steps is set.
  New CPU test test_warm_generation_keeps_a_fixed_step_count (fails without the fix); host file 24/24 pass.
  Fixed common_av.py copied into blx01 /var/tmp/fasth3/t421/b (md5 b8d07600...).
- g15blx02 rebooted again 04:44 PDT. Relaunched chain: off-s0 on-s0 on-s1 on-s2 off-s3 on-s3 off-s4
  on-s4, plus post421.sh, detached in run 1658. off-s1/off-s2 outputs stay on blx01 and are reused.
