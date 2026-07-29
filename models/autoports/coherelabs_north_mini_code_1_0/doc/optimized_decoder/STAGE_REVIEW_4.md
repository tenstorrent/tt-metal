# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Retune and reconcile the selected dense-expert family under the final
  BFP4/LoFi precision policy.
  Evidence:
  The final policy correctly selects dense-expert BFP4 at
  `tt/optimized_decoder.py:40-41`, and the final layer-1 batch-32 decode
  profile confirms three BFP4/LoFi expert matmuls. Those matmuls account for
  about 68% of the measured device window and all run with
  `in0_block_w=1`; `tt-perf-report` explicitly recommends trying at least 2
  (`tracy/review3_selected/layer1_decode_b32/filtered.csv`, operation IDs
  192, 193, and 196). The retained explicit 64/80/100-core and packed
  candidate artifacts instead use the earlier BFP8 expert policy:
  `dense_expert_explicit_64_b32.json`,
  `dense_expert_explicit_80_b32.json`,
  `dense_expert_explicit_adapted_b32.json`, and
  `packed_dense_experts_b32.json` record BFP8 and means of
  3.605/3.390/3.366/3.395 ms. The final BFP4 artifact
  `real_validated_bfp4_expert_decode_b32.json` is faster at 2.214 ms but
  leaves all explicit gate/up/down program fields at zero, so it exercises
  only automatic selection. The work log's claim that core grids and
  `in0_block_w` were swept therefore mixes BFP8 geometry evidence with the
  later BFP4 selection.
  Why this matters:
  Precision changes tile size, memory pressure, legal blocks, and kernel
  balance. OPT-004/010/014 require geometry and packing conclusions to be
  measured under the selected precision, especially when the final profiler
  identifies the dominant rows and an untried block-width opportunity.
  Required next step:
  On the final BFP4/LoFi path, sweep legal explicit 64/80/100-core (or
  equivalently justified) gate/up/down programs, including larger
  `in0_block_w`, subblocks, and packed versus split gate/up. Validate real
  target correctness and traced whole-layer batch-32 latency, retain the
  winning cumulative policy, and explicitly reconcile the profiler advice.

- P1: The compatible DRAM-sharded full-chain family still omits the required
  packed gate/up alternative.
  Evidence:
  The new harness is a substantial improvement: it uses propagated real
  activations and checkpoint weights, runs router, gate, up, fused
  SiLU/multiply, down, routing/reduction, residual, and trace, and establishes
  a correct but losing G8 BFP4 result at 2.818 ms versus the selected
  1.883-ms expert chain
  (`candidates/review3_dram_full_chain_final/`). However,
  `_dram_expert_candidate_operation()` constructs two separate gate and up
  matmuls at `tests/test_optimized_decoder.py:755-814`; neither its weights
  nor its program has a packed gate/up form. The only retained packed result,
  `packed_dense_experts_b32.json`, belongs to the ordinary interleaved BFP8
  family and is not a DRAM-sharded full-chain control. This leaves open the
  exact packed-versus-split comparison required by `STAGE_REVIEW_3.md` and
  diagnosed at `AUTODEBUG_REVIEW3.md:388-396`.
  Why this matters:
  Two same-input projections are a major dispatch and weight-stream
  opportunity. A slower split candidate does not reject a packed form of the
  same compatible DRAM-sharded topology.
  Required next step:
  Adapt the correct G8 BFP4/LoFi full chain to packed gate/up and measure
  real-target correctness plus traced whole-layer batch-32 latency. If the
  batched DRAM-sharded API cannot express or compose the packed form, retain
  the exact post-adaptation API, layout, or capacity blocker. Then update the
  family-rejection claim from the measured comparison.

- P1: Isolate the optimized-decoder stage from unrelated agent-framework
  changes in the requested base-to-stage range.
  Evidence:
  `git diff --name-status 78dbd88bec7..2151146368d` includes twelve
  `.agents/` paths in addition to the North Mini decoder, tests, and
  documentation. It adds `.agents/ARM.md`, changes agent prompts and the
  optimize skill, and deletes the graph-fusing and shard-advise skills and
  scripts. These changes enter the range through ancestor commit
  `51b17c3da34`; they are nevertheless part of the exact stage range being
  reviewed. The stage goal limits changes to the optimized decoder, tests,
  and documentation.
  Why this matters:
  Removing shared skills and changing orchestration prompts is materially
  outside model-stage scope and makes the stage neither review-isolated nor
  safely cherry-pickable from its stated base.
  Required next step:
  Rebase or otherwise isolate the North Mini stage commits onto the stated
  base so the resulting base-to-stage diff contains only the model-scoped
  implementation, tests, and evidence. If `78dbd88bec7` is not the intended
  base, record and use the correct stage base explicitly rather than
  silently excluding an in-range commit.

- P2: Correct the “eight-row dense BFP4 matrix” claim and make the test prove
  the topology it names.
  Evidence:
  `test_optimized_real_weight_moe_precision_matrix` sets
  `dense_expert_batch_threshold=1` and overrides only the dense-expert dtype
  at `tests/test_optimized_decoder.py:442-454`. But `_sparse_moe()` gives
  batch-1 prefill precedence via `active_prefill` at
  `tt/optimized_decoder.py:1305-1317`, so the two batch-1 prefill rows use
  sparse BFP8 experts regardless of that threshold. The test asserts selected
  weight allocation at lines 464-468 but does not assert which runtime branch
  executes. Consequently, README lines 21-24/70-72 and work-log lines
  354-361 overstate the evidence when they describe all eight rows as the
  dense BFP4 selection gate. Six rows do cover the natural dense path, while
  the two batch-1 prefill rows correctly cover the selected mixed policy.
  Why this matters:
  The selected mixed policy is defensible, but its evidence must distinguish
  sparse-BFP8 active prefill from dense-BFP4 serving workloads. Otherwise a
  future branch change can leave the nominal precision matrix passing without
  testing the precision/topology claimed.
  Required next step:
  Split the selected mixed-policy matrix from a forced-dense precision
  matrix, or add branch assertions and rename/document the current rows
  accurately. If all eight rows are to be claimed as dense BFP4 evidence,
  disable active prefill in that dedicated test and rerun it.

## Other Concerns

- `review3_grouped_active_l1_c128_warmed.json` records a 10.331-ms synthetic
  sequence-128 candidate, materially faster than the selected chunk-24
  candidate at 14.335 ms, but the work log does not record why chunk 128 was
  rejected. Chunk 128 may collapse route sparsity across the whole sequence
  and it lacks equivalent real-output correctness evidence; document the
  rejection explicitly rather than leaving a faster artifact unexplained.
- The final profile `report.txt` files contain command chatter from
  `tt-perf-report --csv`, not human-readable advice tables. The raw compressed
  CSVs, filtered CSVs, summaries, and README accounting are usable, but the
  artifact format does not fully satisfy the optimize skill's retained
  human-readable-report requirement.

## Hard-Check Gaps

- No final-policy BFP4 explicit-geometry or packed-versus-split dense-expert
  sweep closes the dominant `in0_block_w=1` profiler advice.
- No packed gate/up result exists for the real compatible DRAM-sharded
  full-chain family.
- The named authentic precision matrix does not prove its runtime branch for
  every row and cannot support the stated all-eight-dense interpretation.
- Per the review request, no hardware test was run during this independent
  review. Conclusions rely on tracked source, machine-readable results, raw
  profiles, JUnit, watcher logs, and capacity artifacts.

## Anomaly Ledger

- Observed anomaly:
  Final BFP4 expert rows dominate batch-32 decode at automatic
  `in0_block_w=1`, while all retained explicit geometry/packing comparisons
  are BFP8.
  Evidence:
  `tracy/review3_selected/layer1_decode_b32/filtered.csv`; explicit/packed
  candidate JSON; `real_validated_bfp4_expert_decode_b32.json`.
  Affected path:
  Layer-1/layer-4 batch-32 dense-expert MoE.
  Control or comparison:
  Final automatic BFP4 versus earlier explicit and packed BFP8 candidates.
  Likely subsystem:
  Dense-expert matmul program configuration and weight layout.
  Investigation performed:
  Source-policy audit, artifact-policy comparison, and final profiler-row
  inspection.
  Resolution:
  more-work-needed; rerun geometry and packing under final precision.

- Observed anomaly:
  The DRAM-sharded family is called a completed full-chain sweep although its
  gate and up projections are always split.
  Evidence:
  `tests/test_optimized_decoder.py:738-834`,
  `candidates/review3_dram_full_chain_final/`, and absence of a packed
  DRAM-sharded artifact.
  Affected path:
  Batch-32 serving MoE candidate topology.
  Control or comparison:
  Correct G8 BFP4 split full chain and ordinary packed BFP8 dense candidate.
  Likely subsystem:
  Batched DRAM-sharded weight/program composition.
  Investigation performed:
  Full helper/source trace and metadata audit across the candidate family.
  Resolution:
  more-work-needed; measure packed full chain or preserve an exact blocker.

- Observed anomaly:
  The authentic matrix's batch-1 prefill rows bypass the forced dense branch.
  Evidence:
  `tests/test_optimized_decoder.py:425-483` and
  `tt/optimized_decoder.py:1305-1317`.
  Affected path:
  Layer-1/layer-4 batch-1 prefill precision evidence.
  Control or comparison:
  Batch-32 prefill and all decode rows do enter the dense path where the
  threshold applies.
  Likely subsystem:
  Test-policy construction and evidence labeling.
  Investigation performed:
  Static dispatch derivation from the exact test configuration.
  Resolution:
  Runtime policy is intentional; evidence wording and branch proof need
  repair.

- Observed anomaly:
  Large-prefill programs and sharded QKV/O weights initially failed at the
  advertised context.
  Evidence:
  Work-log entry 25, final capacity JSON, and watcher capacity log.
  Affected path:
  Aligned 500,000-token and non-aligned 499,999-token optimized prefill, plus
  batch-32 context-500,000 decode allocation.
  Control or comparison:
  Automatic large-M programming and dedicated interleaved QKV/O weights.
  Likely subsystem:
  CB capacity, large-M program selection, and resident weight compatibility.
  Investigation performed:
  Exact CB failure adaptation, layer-0/1/4 capacity runs, final batch-32
  resident-set decode, and watcher repeat.
  Resolution:
  fixed; final artifacts are finite and the watcher log has no fatal/NoC/CB
  error signature.

## Scope Inspected

- Reviewed HEAD `2151146368d912b0df948c168d3416cbe7e9dbc6`, parent
  `c7e024e8faa54118c71771dc1629d8b75abeb022`, and the requested range from
  `78dbd88bec7`.
- Inspected the complete optimized decoder, correctness tests, performance and
  capacity harnesses, context contract, README, work log, AutoDebug/AutoFix
  reports, prior stage reviews, candidate JSON, JUnit XML, final Tracy
  raw/filtered/summary evidence, and final watcher/capacity logs.
- Verified statically that runtime methods are direct optimized overrides,
  default batch-1 prefill selects grouped active experts, dense experts select
  BFP4/LoFi, decode timing is trace replay, and runtime source contains no
  Torch/from-Torch/to-Torch fallback.
- Parsed the four Python implementation/test files successfully. Inspected
  final JUnit results (`30 passed, 16 opt-in skips` both normally and under
  watcher), the 12 three-warmup/20-sample runtime JSON files, the 12 fresh
  final profiles, and the advertised-context artifacts. No device execution
  was performed.

## Residual Risk

The correctness, watcher, final profiling, and advertised-context closures are
credible for the selected code, including capacity after the added resident
large-prefill weight family. The remaining risk is performance-selection and
audit integrity: the dominant final BFP4 expert path has not received a
precision-matched topology sweep, one explicitly required DRAM-sharded
alternative is absent, and the base-to-stage history is not model-scoped.
These gaps prevent a clean pass despite the repaired functional evidence.
