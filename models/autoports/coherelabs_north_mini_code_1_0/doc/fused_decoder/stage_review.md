# Stage Review

Verdict: more-work-needed

## Required Work

- P1: The exhaustive graph-fusing audit missed the score-weighted fused MoE
  reduction that directly matches the measured graph.
  Evidence:
  `FusedDecoder._packed_all_expert_moe` ends with
  `permute(routing) -> multiply(expert_output, routing) -> sum(dim=0)`
  (`tt/fused_decoder.py:318-319`). The retained filtered reports show that
  exact material sequence as `TransposeDeviceOperation ->
  BinaryNgDeviceOperation -> FastReduceNCDeviceOperation`: 2.021 + 378.405 +
  172.746 us in sliding prefill, 2.014 + 373.798 + 172.331 us in full-MoE
  prefill, 1.731 + 92.503 + 43.633 us in sliding batch-32 decode, and 1.670 +
  95.167 + 43.590 us in full-MoE batch-32 decode. The repository exposes
  `ttnn.experimental.deepseek_moe_fast_reduce_nc_fused`; its binding describes
  one kernel that combines `permute + tilize + mul(activation, expert_scores) +
  deepseek_moe_fast_reduce_nc`, accepts a tiled
  `[experts_k, 1, tokens, hidden]` activation and row-major expert scores, and
  eliminates the scaled-activation intermediate
  (`ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/deepseek_moe_fast_reduce_nc_fused_nanobind.cpp:17-64`).
  `graph_fusing_audit.md:73` assesses only replacing `sum` with
  `fast_reduce_nc`; it never assesses the fused score-multiply/reduce op, yet
  lines 94-96 conclude that no matching fused op remains.
  Why this matters:
  The omitted sequence consumes about 137 us at batch-32 and 548-553 us at
  prefill-128, so this is a material dedicated-op candidate, not a cosmetic
  dispatch reduction. Its omission violates graph-fusing Steps 1, 3, and 5 and
  invalidates the exhaustive-audit gate.
  Required next step:
  Adapt and test the registered fused reduction on the 1x1 mesh, including the
  required expert-output layout/sharding, static expert mapping, current top-k
  indices, and dense row-major routing scores. PCC-check it against the current
  path on real stage shapes and benchmark prefill b1 plus traced decode b32.
  Retain it if correct and faster. If the operator cannot express this
  all-expert contract, preserve a minimal repro and exact validation/program
  blocker after the required layout/shape adaptation; update the audit with
  that evidence.

- P1: The dense cross-branch shared-LHS projection family was not assessed.
  Evidence:
  The inherited prefill/decode forward computes one `normalized` tensor and
  sends it to both attention and MLP. The functional dense graph therefore had
  three same-LHS projections after the already-packed QKV projection:
  `QKV(normalized)`, `gate(normalized)`, and `up(normalized)`. The delivered
  rewrite packs only gate/up. Every final dense report still contains a
  separate QKV matmul and gate/up matmul: 59.730 + 69.461 us in prefill b1,
  53.893 + 61.659 us in decode b1, and 1641.553 + 1908.381 us in decode b32.
  `graph_fusing_audit.md:58` mentions the existing QKV pack and the new gate/up
  pack independently but does not assess packing all same-LHS outputs (or the
  resulting QKV + gate/up pair) into one projection with slices. The stage
  owner confirmed this candidate was not benchmarked.
  Why this matters:
  This is the graph-fusing skill's explicit shared-LHS rewrite, and the two
  remaining matmuls account for roughly 24% of fused dense prefill device time,
  38% of dense decode-b1 device time, and 63% of dense decode-b32 device time.
  The “no remaining subgraph” conclusion is therefore unsupported.
  Required next step:
  Implement and PCC/latency-test the setup-packed
  QKV+gate+up/QKV+gate-up candidate in all mandatory dense regimes, or retain
  an adapted measured rejection or exact op-contract blocker. Re-run the
  op-sequence audit after the result. While doing that audit, also classify the
  analogous MoE QKV+router same-LHS pair rather than leaving it implicit.

- P1: Mandatory dense batch-32 traced correctness is not exercised by the
  fused regression suite.
  Evidence:
  The accepted functional suite contains
  `test_serving_batch_32_paged_decode_trace_replay_matches_reference`
  (`tests/test_functional_decoder.py:520-557`). The fused wrapper invokes the
  dense batch-1 trace test, batch-4 cache/determinism test, and a sparse
  layer-1 batch-32 trace test, but never invokes the functional dense batch-32
  reference test (`tests/test_fused_decoder.py:55-109`). Both JUnit files
  confirm the resulting 18 cases and contain no dense batch-32 PCC case. The
  dense batch-32 latency and context probes check only finiteness; they do not
  compare PCC. This matters specifically because the measured dense path uses
  batched matmuls (`b={32}`), a 32-core Q/K layout, a disjoint 32-core V
  layout, the fused cache update, and the packed 6144-wide MLP.
  Why this matters:
  A passing sparse batch-32 path proves the common cache operation can run, but
  it does not prove the dense packed projection and complete dense b32 output
  preserve the accepted functional result. The goal requires traced decode
  b32 evidence and PCC at the functional acceptance threshold.
  Required next step:
  Add a fused wrapper for the existing functional dense batch-32 reference
  test, prove it constructs/dispatches `FusedDecoder`, and rerun it in both the
  normal and watcher suites with PCC >= 0.995 and no skip.

- P2: Material rejected-candidate claims are only prose and are not
  independently inspectable.
  Evidence:
  `graph_fusing_audit.md:49,73,79-89` and `work_log.md:23-34` report an exact
  sparse batch-32 result of 13.395 ms, a 0.325-to-0.320 ms cache-update
  experiment, a split JIT failure, and a 8.293376-versus-8.293310 ms
  reduction comparison. The retained candidate artifacts contain only the
  selected final paths and their functional controls; there is no candidate
  log/JSON/CSV from which those rejection numbers, PCC status, commands, or
  exact configuration can be re-derived.
  Why this matters:
  The exact-sparse rejection is central to selecting the all-expert batch-32
  topology. Under the optimization-stage review standard, a material rejected
  topology must be earned by inspectable correctness and same-regime
  performance evidence, not only a work-log assertion.
  Required next step:
  Preserve machine-readable correctness/timing evidence and the exact command
  and configuration for the exact-sparse batch-32 candidate. Preserve a
  minimal compile/error artifact for the split rejection. Tiny local
  micro-comparisons need not receive full Tracy reports, but their raw result
  must be inspectable if they remain quantitative claims in the final audit.

- P2: The implementation header overstates top-k-only expert evaluation.
  Evidence:
  `tt/fused_decoder.py:12` says sparse experts “evaluate only top-k routes,”
  but the sub-tile path constructs an all-expert sparsity tensor and calls the
  down projection with `nnz=self.num_experts`
  (`tt/fused_decoder.py:269-283`). Both batch-1 MoE reports confirm
  `SparseMatmulDeviceOperation active=128/128` for the down projection. The
  README and graph audit disclose this accurately, but the implementation
  docstring does not.
  Why this matters:
  The inaccurate statement obscures the dominant remaining batch-1 MoE
  operation and contradicts the measured topology.
  Required next step:
  Correct the source documentation to distinguish top-8 sparse gate/up from
  all-expert sparse down, and keep the audit wording aligned with the profiler.

## Other Concerns

- `doc/context_contract.json` is valid as the unchanged functional context
  contract, but its unqualified `completion_status: complete`,
  `pending_gates: []`, and `independent_stage_review.verdict: clean-pass`
  coexist with the fused README's “independent stage review pending.” Qualify
  that review field as the functional/context-stage review, or avoid using it
  as a global current-stage gate.
- The exact PCC decimals in the README/work log are not retained in
  `pytest_results.xml` or `watcher_pytest_results.xml`; those XMLs prove the
  threshold assertions passed but do not preserve stdout. This does not
  undermine the >=0.995 gate, but the decimals themselves remain prose.
- The module performs no explicit deallocation of several large temporary
  packed/all-expert tensors. This review found no runtime failure and the
  context probes are finite, so it is residual memory-pressure risk rather
  than a demonstrated blocker.

## Hard-Check Gaps

- Normal and watcher JUnit evidence is strong for the 18 delivered cases:
  18/18 passed, with zero error, failure, skip, or xfail. However, JUnit was
  configured without captured stdout, so exact PCC values and selected experts
  cannot be independently re-derived; only each test's hard >=0.995 assertion
  can.
- The nine final fused Tracy captures and all nine functional controls are
  internally sound: each raw CSV has exactly two matching signposts; every
  global-call ID and device-kernel duration in the signpost window matches its
  filtered CSV; every table reports zero host ops. Candidate/rejection runs do
  not have the same provenance coverage.
- Final wall JSONs use 20 samples after five fused warmups. Several functional
  MoE wall JSON controls contain 10 samples. The gains are large and the
  device-profiler comparisons are exact matched-regime controls, so this does
  not reverse any reported winner, but the wall sample counts are not
  identical.

## Anomaly Ledger

- Observed anomaly:
  The audit declares that no fused op remains while an existing registered op
  exactly names the retained route multiply/reduce sequence.
  Evidence:
  `graph_fusing_audit.md:73,94-96`; fused-reduce binding at
  `deepseek_moe_fast_reduce_nc_fused_nanobind.cpp:17-64`; final MoE profiler
  rows listed in Required Work.
  Affected path:
  Packed all-expert MoE, prefill b1 and decode b32.
  Control or comparison:
  Current final path is correct and faster than functional, but no fused-op
  candidate result exists.
  Likely subsystem:
  Graph-fusing audit/operator discovery.
  Investigation performed:
  Re-derived the final sequence from raw/filtered CSVs and searched the TTNN
  operation library and common MoE module usage.
  Resolution:
  more-work-needed.

- Observed anomaly:
  The dense parallel branches retain separate high-cost same-LHS projections.
  Evidence:
  Dense filtered CSV rows and `graph_fusing_audit.md:58`; stage-owner
  confirmation that the candidate was not benchmarked.
  Affected path:
  Dense prefill b1 and traced decode b1/b32.
  Control or comparison:
  Final packed gate/up is correct and faster than functional, but no
  cross-branch pack was tested.
  Likely subsystem:
  Shared-LHS graph rewrite.
  Investigation performed:
  Compared inherited forward dataflow with functional and fused op sequences.
  Resolution:
  more-work-needed.

- Observed anomaly:
  The initial paged fused-cache call used overlapping K/V core grids.
  Evidence:
  `work_log.md:17-21`, `graph_fusing_audit.md:51`, and the delivered disjoint
  grid code at `tt/fused_decoder.py:66-88,189-200`.
  Affected path:
  All decode kinds and batches.
  Control or comparison:
  Current batch-1/batch-4/batch-32 tests pass; final CSVs show one V reshard
  plus `PagedFusedUpdateCacheDeviceOperation`; watcher signature scan is clean.
  Likely subsystem:
  Fused cache NoC/core-grid contract.
  Investigation performed:
  Inspected code, semantic tests, final profiler rows, and watcher log.
  Resolution:
  fixed.

- Observed anomaly:
  Exact per-token sparse down hit an A-sparse mask-contract limitation and the
  retained sub-tile down projection is active for 128/128 experts.
  Evidence:
  `graph_fusing_audit.md:82-86`, `tt/fused_decoder.py:269-283`, and both
  batch-1 MoE CSVs.
  Affected path:
  MoE decode b1.
  Control or comparison:
  Gate/up remains active=8/128; final layer-1/layer-4 synthetic and official
  real-weight PCC tests pass, and device time is about 2.1 ms versus the
  approximately 9.45 ms functional control.
  Likely subsystem:
  Sparse-matmul A-mask granularity.
  Investigation performed:
  Compared code comments, profiler active counts, PCC suite, and functional
  control.
  Resolution:
  controlled, with inaccurate header documentation still requiring correction.

- Observed anomaly:
  `ttnn.split` produced a Blackhole JIT error.
  Evidence:
  `work_log.md:23-25` and `graph_fusing_audit.md:79-81`.
  Affected path:
  Packed gate/up output separation.
  Control or comparison:
  Two width slices implement the same split, are visible in every final
  relevant profiler, and pass the correctness/performance suite.
  Likely subsystem:
  Split kernel code generation.
  Investigation performed:
  Inspected the retained alternative and its current evidence.
  Resolution:
  controlled for this stage, subject to preserving the claimed minimal error
  evidence.

- Observed anomaly:
  `tt-perf-report` warns that several specialized operations are unclassified.
  Evidence:
  All nine fused `*_perf_report.txt` files.
  Affected path:
  Roofline category/advice only.
  Control or comparison:
  Raw kernel durations, filtered totals, op counts, and before/after gains all
  match exactly; no operation is omitted from the filtered window.
  Likely subsystem:
  Profiler operation-category metadata.
  Investigation performed:
  Matched every raw signpost-window global ID and kernel duration to each
  filtered CSV.
  Resolution:
  controlled/non-blocking.

- Observed anomaly:
  Final `tt-smi` warns that firmware bundle 19.9.0 is newer than the latest
  fully tested 19.5.0 bundle.
  Evidence:
  `final_tt_smi_list.log`.
  Affected path:
  Hardware environment.
  Control or comparison:
  Four Blackhole p300c boards enumerate normally, both test suites pass, and
  the watcher log has clean attach/dump/detach cycles with no fatal/assert/NoC
  signature.
  Likely subsystem:
  Environment qualification.
  Investigation performed:
  Inspected inventory, watcher log, XML, and profiler artifacts.
  Resolution:
  controlled/non-blocking.

## Scope Inspected

- Goal/skill paths:
  Original Stage-02 fused-decoder contract supplied to the reviewer;
  `.agents/skills/stage-review/SKILL.md`;
  `.agents/skills/graph-fusing/SKILL.md`.
- Artifact paths:
  All files under
  `models/autoports/coherelabs_north_mini_code_1_0/doc/fused_decoder/`,
  including README, work log, graph audit, 16 latency/capacity JSONs, both
  JUnit XMLs, final `tt-smi` log, the 20,721-line watcher log and generated
  watcher metadata, and all nine raw/filtered/stacked/table Tracy sets; all
  functional baseline latency/capacity/correctness artifacts and all nine
  functional raw/filtered/table Tracy controls under `doc/functional_decoder/`;
  `doc/context_contract.json`.
- Code paths:
  `tt/fused_decoder.py`, `tt/functional_decoder.py`,
  `tests/test_fused_decoder.py`, `tests/fused_decoder_perf.py`,
  `tests/fused_decoder_capacity.py`, the full functional test/perf/capacity
  harnesses, relevant TTNN fused MoE gate/reduction bindings and validation,
  and common MoE usage.
- Commands run:
  Read-only `sed`, `nl`, `rg`, `find`, `wc`, `git status`, `git diff`,
  `git ls-tree`, `git log`, `git rev-parse`, and small Python scripts for
  parsing XML/JSON/CSV, checking signposts, matching raw global IDs and kernel
  durations to filtered reports, summing device times/op gaps, comparing
  before/after rows, and enumerating test cases. No hardware, server, reset,
  model execution, profiler capture, or implementation mutation was run.

## Residual Risk

- The delivered final graph is genuinely fused relative to the functional
  baseline and all nine measured winners are real: wall/device gains are
  positive in every required regime, raw/filtered artifacts agree, current
  tests dispatch `FusedDecoder`, the official layer-1 real-weight test ran
  without skip, capacity remains 500000, and watcher evidence is clean.
  Nevertheless, the stage cannot pass while two concrete material fusion
  families are unassessed and dense batch-32 lacks its direct accepted PCC
  control.
- The current MoE correctness gate uses official real weights only for
  layer-1 batch-1 decode; other large-shape MoE paths use deterministic
  recorded-statistics synthetic weights. This is inherited accepted evidence,
  not a newly observed failure, but any future rewrite of routing/reduction
  should preserve the official-weight check and all dynamic trace checks.
- The nanobind leak diagnostic is described only in prose because fused pytest
  stdout was not retained. The functional-stage control reports the same
  shutdown-only diagnostic, and current devices close/watch cleanly, so it is
  not a demonstrated stage-critical leak.
