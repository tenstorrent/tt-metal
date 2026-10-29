# Stage review, round 4: the round-3 follow-up (commit 4ae4eef5bcc)

Independent review (fresh subagent, read-only, no device use), 2026-10-29. Scope: the resumed loop's commits
(iterations 4-8, the rejected trial, the TP sweeps) and the round-3 follow-up.

## Verdict: more-work-needed (doc / test items only)

The reviewer re-derived round 3's findings 1-6 from the raw profiler CSVs and logs and found them answered by
evidence: the layer-0 chain is gone (iter8 -> review3, 522 -> 515 ops per 4-layer step on chips 0 and 2, exactly
six rows changed: reduce-scatter 18 -> 17, all-gather 18 -> 17, residual adds 32 -> 30, norm backward 9 -> 8 and
partial 9 -> 8, qkv dX 4 -> 3; single chip 847 -> 844 ops, three rows changed), the gradient check equals
iteration 8's to the printed digit, the loss match is on the final code with the default flags (max |diff| 0.027,
TP trajectory within 0.004 of the baseline), the ledger clocks and temperatures match the files, the sweep policy
and the stacking cost are stated and supported.

Required work (all P3):
1. `optimization/README.md:9` still said "14 at the final path" (round-3 7c) — fixed: 15 (16 with the test below).
2. `work_log.md` commit table: iteration 8 still "(this commit)", no row for the follow-up — fixed: `4777ff7646a`,
   `4ae4eef5bcc` added (and this round's row).
3. "thermally bound box" survived in `README.md:38` — reworded as in `perf_analysis.md`.
4. `test_frozen_embedding_matches_the_ttml_op` asserted `out.get_node() is None or not weight.requires_grad`, which
   is vacuous once the weight is frozen — fixed: `assert not out.get_requires_grad()` and `out.get_node() is None`;
   a harness case added for the no-grad branch of the all-core norm wrapper
   (`test_rmsnorm_allcore_path_builds_no_node_for_a_no_grad_input`). Both run: `evidence/review4_tests.log`, 5
   passed (the frozen-embedding and the four rmsnorm cases).
5. Sweep paragraph numbers: 1624 distinct configs (1649 lines) not 1564, "ten" configs between 7.09 and 7.34 not
   five, and the trailing sorted list was appended by hand — fixed in `perf_analysis.md`.

Other concerns taken: the dX-sweep wording now says why each sub-block is excluded (4x1 by per-core M, 2x2 by
per-core N = 1); the 5b collective figures are given as a per-step range (126-162 + 58-59 us); the one cheaper
stacking variant (aligned, no unpad: 0.18 ms, needs a 32-row-padded `dA`) is listed as deferred. Not taken (noted):
the single-chip layer-0 slice / transpose of the fused dX for a no-grad input (~0.1 ms of 283 ms); the 6-10 %
run-to-run swing of `SwigluPackedBw` (not material); tt-smi's last ramp-down samples.

Residual risk recorded by the reviewer: the no-grad branches are now also covered by harness assertions (item 4);
the worktree's unrelated dirty files (`tt_metal/third_party/umd`, `models/demos/t3000/llama2_70b/`) predate the
phase and are outside its commits.
