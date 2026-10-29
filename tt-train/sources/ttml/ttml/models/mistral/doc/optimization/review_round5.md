# Stage review, round 5: close-out of the resumed loop (commit ddf3a77be15)

Independent review (fresh subagent, read-only, no device use), 2026-10-29.

## Verdict: clean-pass

Required work: none. All five round-4 items verified in the files, not in the commit message: README harness count
(15, 16 since the no-grad norm case; `review4_tests.log` collected 16 items, 5 selected, 5 passed); commit table
rows 4 .. review 4 match `git log` hash for hash; no "thermally bound" / "uniformly hotter" in the three docs; the
frozen-embedding test asserts `not out.get_requires_grad()` and `out.get_node() is None`, the new no-grad norm case
is non-vacuous (the wrapper routes a no-grad input to the C++ op, `RMSNormFrozenGamma.apply` would always build a
node); the sweep paragraph matches the log (1649 result lines, 1624 distinct configs, the 25-line hand-appended
list, 7.09 fastest + exactly ten between 7.14 and 7.34 at max |diff| 3.12e-2, fastest bit-exact 7.35, default 7.92);
the dX-sweep exclusion reasons match `matmul_sweep1d.py` (per-core M 162 / 258 has no divisor divisible by 4,
per-core N = 1 excludes 2x2); `review_round4.md` matches the diff of ddf3a77be15 one-to-one; every table row has
its header's cell count.

Non-blocking notes (P4), taken in the closing commit: three remaining "thermal noise" phrases (README:30,
perf_analysis.md:21 and :204) reworded to the effective-clock swing the ledger describes; a work-log bullet for
round 4; the round-4 "(this commit)" row filled with `ddf3a77be15`.

Residual risk as recorded: the worktree's unrelated dirty files (`tt_metal/third_party/umd`,
`models/demos/t3000/llama2_70b/`) predate the phase and are outside its commits.

## State of the phase at close

TP=4 full model 598.7 ms (composite baseline) -> 556.5 (loop exit after iteration 3) -> **545.7 ms / 29.73 % MFU**
(iteration 8 + follow-up), loss match on the final path max |diff| 0.027 vs torch, TP=4 real-weight trajectory as
the baseline's. Commits of the resumed loop: dc6ccf3db8e (4), 7c19e385351 (5), 758a7300a8d (5b), 190f3aad6fd (6),
d546567c67f (7, rejected), d03dfffa23c (TP sweeps), 4777ff7646a (8), 4ae4eef5bcc (review 3 follow-up),
ddf3a77be15 (review 4 items), plus this closing commit. The remaining ranking is in `../perf_analysis.md`, "Loop
resumed": the base matmuls at ttnn's optimum for the TP shapes, inherent TP collectives, the layout of the fused
route and of rank-16 adapters, and three kernel projects (SDPA, attention on the fused heads layout, a
residual-fused / all-core RMSNorm forward).
