# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Complete the mandatory `shard-advise` seed and preserve its evidence.
  Evidence:
  The optimize skill at the stated functional base and at final HEAD has the
  same content hash and requires OPT-015: run `$shard-advise` on the rewritten
  dense attention+MLP block, save `report.json` and `final_ir.mlir` under
  `doc/optimized_decoder/shard_advise/`, apply its per-op layout/program
  recommendations as first candidates, and record measured rejections. There
  is no `shard_advise/` directory, `report.json`, `final_ir.mlir`, OPT-015
  result, or shard-advisor discussion anywhere in the optimized decoder,
  tests, README, or work log. The checklist instead says only that layouts
  were derived from shapes. This requirement was already present at
  `78dbd88bec7`; it is not a later skill change.
  Why this matters:
  The compiler seed is a mandatory optimization-stage check, not an optional
  artifact format. The hand sweep may still have selected the best path, but
  the stage has not compared it with the required compiler-validated L1
  residual-chain and 1D-mcast candidates.
  Required next step:
  Run `$shard-advise` once on the final rewritten dense block, retain its
  `report.json` and `final_ir.mlir` in the required directory, apply the
  emitted layouts/programs as candidates, and compare them with the selected
  DRAM-sharded path. Record exact before/after results and any per-op rejection
  or revert. Then rerun the affected correctness, traced decode, and profiler
  gates.

- P1: Retune and reconcile the final BFP4/LoFi dense-expert **prefill**
  geometry; Review 4 closes decode only.
  Evidence:
  The retained final profiles show that batch-32 layer-1 and layer-4 prefill
  spend 65.30% and 65.47% of device time in 18 DRAM-interleaved matmuls
  (`tracy/review3_selected/layer{1,4}_prefill_b32/human_report.txt`).
  Repeated `b={128} x 1024 x 2048 x 768` and
  `b={128} x 1024 x 768 x 2048` BFP4/LoFi rows are marked `SLOW`,
  use `in0_block_w=1`, and explicitly recommend `in0_block_w>=2`; representative
  layer-1 rows and advice are at lines 166-170 and 284-317, with the 65.30%
  aggregate at line 368. The runtime reaches this path through 1024-token
  chunks at `tt/optimized_decoder.py:1305-1309`, while its final dense-expert
  program fields are all zero/automatic at lines 83-95.

  Every file in `candidates/review4_dense_bfp4/` is `mode=decode`,
  batch 32, with explicit `per_core_M=1`; none measures the 1024-token
  batch-32 prefill matmuls. Thus README lines 151-153 overgeneralize when they
  say the profiler's dense-expert block-width suggestion was swept under the
  final policy. The Review 4 matrix validly closes the dominant decode rows,
  not these separately shaped prefill rows.

  There is also an unreconciled stronger earlier result:
  `candidates/dense_expert_chunk1024_prefill_b32.json` records 117.903 ms for
  the same sequence-128 batch-32 BFP4 expert/chunk family, while the final
  layer-1 result is 139.959 ms
  (`candidates/review3_final_runtime/layer1_prefill_b32.json`; layer 4 is
  139.855 ms). The work log neither reproduces nor explains this material
  difference. Current construction also materializes both BFP8 sparse and
  separate BFP4 dense expert weights at `tt/optimized_decoder.py:466-478`,
  although the batch-32 dispatch contract always enters the dense branch;
  this is an uninvestigated cumulative-policy storage difference that may be
  relevant to the regression.
  Why this matters:
  OPT-004/010/013/014 and the advice policy require a precision-locked search
  for each dominant shape, and the final-default rule requires the selected
  path to reproduce or explain the strongest prior correct candidate.
  A decode `M=32`/`per_core_M=1` sweep cannot reject prefill `M=1024`
  geometry, packing, subblocks, or input-placement changes.
  Required next step:
  On the final cumulative BFP8-sparse/BFP4-dense policy, sweep the batch-32
  sequence-128 prefill expert chain with prefill-appropriate `per_core_M`,
  legal larger `in0_block_w` divisors, core grids, subblocks, split versus
  packed gate/up, and L1/DRAM input/intermediate placement. Use the real
  sequence-33 matrix or equivalent real propagated activation coverage to
  preserve correctness, reproduce the winning final default with the normal
  warmed harness, and explain or recover the 117.903-ms prior result. Update
  the README so decode and prefill conclusions are stated separately.

## Other Concerns

- The focused Review 4 watcher log exists in the live tree but is ignored and
  not part of the model-only commit range. This does not independently block
  the stage because the tracked Review 4 JUnit records the focused watcher
  pass and the tracked Review 3 full-suite/capacity watcher logs cover the
  unchanged implementation. Preserve the focused log if it is intended to
  remain a named durable artifact.
- Generated profiler CSVs retain CRLF/trailing whitespace, so
  `git diff --check 78dbd88bec7..HEAD` is noisy. This is artifact hygiene, not
  a correctness or performance finding.

## Hard-Check Gaps

- No required `doc/optimized_decoder/shard_advise/report.json` or
  `final_ir.mlir` exists, and no compiler-seeded candidate was measured.
- No final-policy BFP4/LoFi geometry/packing table covers the dominant
  batch-32 prefill `M=1024` expert rows; all Review 4 geometry JSON is decode.
- The final batch-32 prefill default is materially slower than a retained
  earlier BFP4 chunk-1024 result without a same-harness explanation.
- Per the review request, no hardware, server, or vLLM execution was performed
  during this independent review.

## Anomaly Ledger

- Observed anomaly:
  The Review 4 dense-expert geometry closure is documented as closing the
  profiler advice generally, but its artifacts cover decode only while the
  retained prefill reports show the same advice on the dominant final-policy
  rows.
  Evidence:
  `candidates/review4_dense_bfp4/*.json`;
  `tracy/review3_selected/layer{1,4}_prefill_b32/human_report.txt`;
  README lines 139-153.
  Affected path:
  Layer-1/layer-4 batch-32 sequence-128 prefill.
  Control or comparison:
  The valid Review 4 batch-32 decode sweep and the earlier
  `dense_expert_chunk1024_prefill_b32.json` result.
  Likely subsystem:
  Shape-specific dense-expert program geometry, packing, intermediate
  placement, and cumulative expert-weight residency.
  Investigation performed:
  Cross-checked every Review 4 geometry JSON field against both final prefill
  human reports, the runtime dispatch/program construction, the earlier
  candidate matrix, and the final 20-sample runtime JSON.
  Resolution:
  more-work-needed; run a final-policy prefill-specific search and reconcile
  the default.

- Observed anomaly:
  The mandatory compiler layout seed is absent even though the work log marks
  the optimize checklist complete.
  Evidence:
  OPT-015 and the final evidence checklist in
  `.agents/skills/optimize/SKILL.md`; absence of `shard_advise/`,
  `report.json`, and `final_ir.mlir`; no matching work-log entry.
  Affected path:
  Dense attention+MLP residual layout and non-DRAM-sharded candidate search.
  Control or comparison:
  Manual 8/12/16/32-core and DRAM-sharded candidate evidence.
  Likely subsystem:
  Stage-process/evidence completeness rather than demonstrated runtime
  correctness.
  Investigation performed:
  Compared the optimize skill at the functional base and final HEAD, then
  searched all stage-owned code, tests, and documentation.
  Resolution:
  more-work-needed; run and record the mandatory seed and candidate
  comparison.

- Observed anomaly:
  Review 4 found that final BFP4 decode geometry was previously inferred from
  BFP8 evidence and that the compatible DRAM-sharded chain lacked packed
  gate/up.
  Evidence:
  `STAGE_REVIEW_4.md`;
  `candidates/review4_dense_bfp4/`;
  `candidates/review4_dram_full_chain_packed/`;
  `artifacts/review4_dram_packed.xml`.
  Affected path:
  Batch-32 dense-expert decode and the DRAM-sharded alternative.
  Control or comparison:
  Automatic/explicit/packed BFP4 decode rows and split/packed real-weight G8
  DRAM chains.
  Likely subsystem:
  Expert matmul precision/geometry and topology-family completeness.
  Investigation performed:
  Parsed all Review 4 candidate JSON and JUnit, traced the packed helper
  through router, gate/up, slices, activation, down, routing, reduction, and
  residual-layout boundary, and checked its real-weight source.
  Resolution:
  fixed for the Review 4 decode findings. Automatic split BFP4 remains the
  measured decode winner; packed G8 BFP4 is correct and slower.

- Observed anomaly:
  Review 4 found the selected precision matrix did not prove the runtime
  branch named by its documentation.
  Evidence:
  `tests/test_optimized_decoder.py:425-567`;
  `artifacts/review4_{selected_mixed,forced_dense_bfp4}_matrix.xml`.
  Affected path:
  Layer-1/layer-4 real-weight prefill/decode at batches 1/32.
  Control or comparison:
  Selected mixed topology versus explicitly forced dense BFP4.
  Likely subsystem:
  Test-policy construction and evidence labeling.
  Investigation performed:
  Traced the branch counters and parsed both 8/8 passing JUnit matrices.
  Resolution:
  fixed. The selected matrix proves two active-sparse prefill and six dense
  rows; the forced matrix proves eight dense BFP4 rows.

- Observed anomaly:
  The previous reviewed range contained unrelated `.agents/` history.
  Evidence:
  `git merge-base`, ancestry log, and base-to-final name audit.
  Affected path:
  Stage history isolation.
  Control or comparison:
  Requested base `78dbd88bec7` and final HEAD
  `74c95ddaf4f7f8eccf37f577e0a926afccbf52e9`.
  Likely subsystem:
  Commit ancestry.
  Investigation performed:
  Verified the base is the exact merge base and filtered every changed path.
  Resolution:
  fixed. The seven-commit range contains only
  `models/autoports/coherelabs_north_mini_code_1_0/{tt/optimized_decoder.py,tests/,doc/}`.

## Scope Inspected

- Goal/skill paths:
  `.agents/skills/stage-review/SKILL.md`,
  `.agents/skills/optimize/SKILL.md`, functional base `78dbd88bec7`, initial
  requested HEAD `770f70051f9bb1d71122788b6f16342d50e90f67`, and final doc-only
  review HEAD `74c95ddaf4f7f8eccf37f577e0a926afccbf52e9`.
- Artifact paths:
  `README.md`, complete `work_log.md`, `STAGE_REVIEW_1.md` through
  `STAGE_REVIEW_4.md`, AutoDebug/AutoFix/triage reports, context contract and
  capacity JSON, all Review 4 JUnit/candidate JSON, all 12 final runtime JSON
  and retained human-readable profiler reports, and final/focused/capacity
  watcher logs.
- Code paths:
  Complete `tt/optimized_decoder.py`,
  `tests/test_optimized_decoder.py`, `tests/optimized_decoder_perf.py`, and
  `tests/optimized_decoder_capacity.py`, with relevant functional helpers and
  base-to-final history.
- Commands run:
  Read-only `git status/log/show/diff/merge-base/rev-list/ls-files`,
  `rg`, `find`, `sed`, `nl`, `jq`, JSON/XML parsing, AST parsing, candidate
  matrix aggregation, and watcher signature scans. No hardware was opened and
  no test, profiler, server, or vLLM process was launched.

## Residual Risk

- Functional evidence is otherwise strong: the final tracked suite is 30
  passed with 16 opt-in skips; selected and forced-dense real-weight matrices
  are each 8/8; paged-cache slots, non-aligned lengths, representative layer
  kinds, trace replay, determinism, advertised-context prefill/decode, and
  watcher checks have passing artifacts.
- The final 12 profiler reports are current for the unchanged implementation,
  retain advice, show zero host ops, and verify the selected runtime dtypes.
  Review 4's BFP4 decode geometry, packed DRAM chain, branch-proof, readable
  report, and history-isolation findings are credibly closed.
- The stage cannot pass while a mandatory optimize-skill seed is absent and a
  65%-dominant final-policy prefill matmul family remains untuned with an
  unexplained materially faster retained candidate. These are optimization
  closure gaps, not evidence of a known functional correctness failure.
