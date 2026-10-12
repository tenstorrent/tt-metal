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
