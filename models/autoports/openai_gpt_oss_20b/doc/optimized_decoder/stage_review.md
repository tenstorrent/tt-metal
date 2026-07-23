# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The final JUnit XML records current-source pass/fail coverage but does not
  embed the printed PCC and wall-latency samples. Exact values are carried in
  the work log and compact profiler reports. This is an evidence-format
  limitation, not a contradiction: the final default performance gate passed,
  the canonical profile independently reconciles its device work, and the
  candidate JUnits identify the exercised variants.
- The watcher process reports nanobind shutdown leak diagnostics after all ten
  selected tests pass and the device closes cleanly. There is no watcher,
  kernel, NoC, timeout, stuck, uninitialized-memory, or fallback failure in
  the retained run, so this is classified as a host binding teardown warning
  rather than decoder corruption.

## Hard-Check Gaps

- The optimized-decoder runner validates advisor artifact presence, JSON
  parsing, and a matmul-bearing IR. It does not enforce recommendation
  disposition, precision propagation, or per-role geometry coverage; those
  were checked manually against the work log, current code, JUnit files, and
  profiler tables.
- The performance pytest enforces that the current optimized default beats
  fused, but does not enforce every candidate-versus-default delta. The
  retained O-candidate profile and the documented same-harness 500-replay
  matrix provide that comparison.

## Anomaly Ledger

- Observed anomaly: The former S=2048 full-layer value-cache PCC was below the
  functional bar.
  Evidence: `_fill_prefill_cache` now performs bounded 128-token
  `fill_cache(..., update_idx=start)` writes.
  `autofix_native_chunked_boundary_2048.junit.xml` records passing sliding and
  full cases with output, K/V, and following-decode assertions.
  Affected path: Long prefill cache population and following decode.
  Control or comparison: BF16-cache and precise-norm controls reproduced the
  old failure; bounded offset writes repaired it.
  Likely subsystem: Multi-head cache placement stride.
  Investigation performed: Isolated dtype/norm controls followed by bounded
  offset-write validation.
  Resolution: fixed.

- Observed anomaly: Primitive manual attention was previously selected for
  every decode position at or above 128.
  Evidence: The default now uses native
  `scaled_dot_product_attention_decode` from position 128 onward, with the
  manual FP32 guard limited to position 127. Repeated positions 127-131 pass
  for both layer kinds. Boundary reports show native device-plus-gap totals
  below manual for layer 12 (0.870 versus 1.326 ms) and layer 13
  (6.773 versus 7.402 ms).
  Affected path: Boundary and long-context decode.
  Control or comparison: Same-policy manual versus native traced windows.
  Likely subsystem: Sink-aware attention numerical policy.
  Investigation performed: Correctness isolation, repeated boundary checks,
  wall timing, and Tracy comparison.
  Resolution: fixed.

- Observed anomaly: The selected QKV projection retained
  `in0_block_w=2`.
  Evidence: Final-policy 30/3, 18/5, 15/6, and 9/10 input-core/block-width
  candidates measured 0.812395, 0.812845, 0.811533, and 0.812687 ms versus
  the adjacent 45/2 control at 0.811265 ms. The widest endpoint passed both
  real layer-kind checks.
  Affected path: Packed decode QKV projection.
  Control or comparison: Same BF16/LoFi traced default.
  Likely subsystem: QKV activation shard/program geometry.
  Investigation performed: Precision-locked wider-shard sweep plus
  real-weight correctness.
  Resolution: controlled; the fastest correct candidate remains selected.

- Observed anomaly: The selected O projection used 90 output cores,
  `per_core_N=1`, and a 1x1 output subblock without a measured wider-subblock
  candidate.
  Evidence: Final-policy 45-core/subblock-2 and 30-core/subblock-3 families
  were measured with the required return to the 90-core residual contract.
  The 500-replay means were 0.812616, 0.813712, 0.814129, and 0.816731 ms
  versus the adjacent selected control at 0.811668 ms. The fastest and extreme
  candidates passed real-weight prefill/decode and cache PCC for layers 12 and
  13. In Tracy, the 45-core O row plus return reshard was
  59.368 + 1.490 us versus 58.721 us for selected O with no O-specific
  reshard. Although unrelated row variance made the candidate's whole
  profiled window slightly lower, the changed O region and the uninstrumented
  500-replay target both regressed.
  Affected path: Decode output projection and residual-layout boundary.
  Control or comparison: Precision-locked selected 90/8/1 path versus
  45/8/2, 45/32/2, 30/8/3, and 30/32/3.
  Likely subsystem: O-projection output geometry and residual resharding.
  Investigation performed: Whole-layer traced sweep, real-weight endpoint
  correctness, and targeted profiler-row comparison.
  Resolution: controlled; 90/8/1 remains the measured winner.

- Observed anomaly: The advisor report formerly referenced an absent decision
  trace.
  Evidence: `report.json` now resolves the gzip-compressed trace and its
  documented hash. The current report has 39 modeled ops, 38 choices,
  24 reshards, zero spills, and zero unfixable ops; `final_ir.mlir` contains
  the rewritten attention plus dense MLP graph and five linear/matmul ops.
  Affected path: Mandatory OPT-015 provenance.
  Control or comparison: JSON/path/hash/gzip checks, authoritative IR, and the
  static stage runner.
  Likely subsystem: Artifact packaging.
  Investigation performed: Path resolution, JSON/IR inspection, hash and gzip
  validation.
  Resolution: fixed.

- Observed anomaly: Final candidate and selected profiler totals differ by
  normal run-to-run movement outside the changed O region.
  Evidence: Selected decode is 775.859 us device + 79.352 us gaps; the
  45-core candidate is 769.199 + 76.392 us. The candidate nevertheless makes
  the O region 2.137 us slower and makes the 500-replay uninstrumented target
  0.948 us slower.
  Affected path: O-geometry selection.
  Control or comparison: Row-local delta and uninstrumented whole-layer
  target, not cross-run whole-profile total alone.
  Likely subsystem: Profiler/process variance in unchanged operations.
  Investigation performed: Reconciled row-local and end-to-end measurements.
  Resolution: controlled; not a true performance regression in the selected
  path.

## Scope Inspected

- Goal/skill paths: optimized-decoder goal contract; complete `stage-review`,
  `optimize`, `tt-device-usage`, `shard-advise`, and shard-advisor setup
  instructions.
- Artifact paths: optimized README/work log, context contract, final and
  candidate JUnits, watcher log, final/boundary/O-geometry profiler reports,
  advisor `report.json`, `final_ir.mlir`, capture harness, and compressed
  decision trace.
- Code paths: current optimized decoder configuration/runtime methods and its
  tests, with focused inspection of optimized-path ownership, cache writes,
  layer-kind/window semantics, non-aligned lengths, trace replay, sparse
  experts, precision propagation, QKV/O geometry controls, and performance
  harnesses.
- Commands run: read-only source/diff/artifact inspection, Python AST and
  syntax checks, JSON/IR/hash/gzip checks, profiler recomputation, watcher-log
  scan, `git diff --check`, and
  `.agents/prompts/model_bringup_multigoal/02-optimized-decoder.check.sh`.
  No hardware or server command was run.

## Residual Risk

- S=131072 is a real-weight capacity and finite-edge-output qualification,
  while S=2048 is the largest full output/cache/following-decode PCC point.
  This distinction is explicit in the context contract and does not reduce
  the advertised capability.
- Batch two is correctness-qualified through an optimized-owned dense
  compatibility graph but is not a measured latency target. Batch-one traced
  latency is the declared optimization target.
- The stage-owned changes remain uncommitted at review time. A local
  scope-isolated checkpoint commit and work-log SHA entry are still the
  orchestrator's post-review bookkeeping step; this does not change the
  clean-pass verdict on the implementation and evidence.
