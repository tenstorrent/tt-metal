# Stage Review

Verdict: more-work-needed

Independent inspection of Stage 2 fused decoder for `Qwen/Qwen3.8-27B`,
live worktree on `mvasiljevic/qwen38-full-bringup`, base `41bdb276313`,
2026-09-11. Reviewed implementation SHA256:
`a362a3e1be350b1fdd2edfa50aaf724727bcd2d37822883c673632bad758ff04`.
This review covers the evidence available before the stage owner's queued
final benchmarks, batch-3 check, and untilize experiment. The initially
running full-context test finished during this review and is accepted below.
No concrete implementation defect was found. The outstanding items are
specific validation and completion work, not grounds to abandon the stage.

## Required Work

- P2: Measure the selected final default implementation's warmed latency.
  Evidence: both `final_linear_benchmark.source.sha256` and
  `final_full_benchmark.source.sha256` contain `227e09b629f36a1de7a542b99fb53b17fcf8484cca9de7fad3658be8f4a718c5`,
  which exactly matches `candidates/l1_rope.py`. That source precedes the
  padded-scan, logical-output-view and native-decode-RoPE rewrites. These
  files report linear prefill/decode medians 3.06362/2.50333 ms and full
  2.70799/2.29896 ms. Later candidate results include 2.45619 ms linear
  decode and 2.25999 ms full decode. Current-source profiler evidence exists,
  but the final wall-latency distribution has not yet been recorded.
  Why this matters: the stage explicitly requires the fastest correct traced
  decode candidate and reproduced before/after warmed timings; an older file
  named "final" cannot establish the selected runtime result.
  Required next step: after selecting or rejecting the remaining candidate,
  run the existing benchmark path for both layer kinds with the final default
  source, retaining all samples, direct functional/HF PCC and trace checks.
  Report those reproduced medians and compare them with the recorded baseline.
  Existing current-source profiler evidence need not be rerun if the measured
  implementation remains unchanged.

- P2: Cover the uneven batch group in the new flat, head-major linear scan.
  Evidence: `tt/fused_decoder.py` computes `scan_batch = grid.x * grid.y // hv`
  and slices/concatenates scan output and persistent state at lines 405-426.
  On the recorded 110-core grid with 48 value heads, this is a group of two.
  Current fused tests exercise batch 1, 2 and 32. Batch 32 covers repeated
  groups of two, but no current fused artifact exercises a final group of
  one after a group of two. The earlier functional stage's batch-3 evidence
  uses the previous scan layout and recurrence path.
  Why this matters: head-major outputs now have leading dimension `B*48`,
  while recurrent state retains a separate batch dimension; uneven grouping
  crosses the new output/state assembly boundary. The installed stage-review
  skill explicitly requires affected allocation/sharding branches and tails.
  Required next step: one real-weight linear batch-3 case with an awkward
  length such as 257, continuation, and traced decode/replay is sufficient.
  A full new boundary sweep or long soak is not requested.

- P2: Complete the outstanding untilize fusion assessment.
  Evidence: `tests/untilize_candidate.py` implements a separate decode-only
  candidate that folds TILE-to-ROW_MAJOR output conversion into the QKV
  projection, while separating the remaining packed projection columns.
  Its dispatch exists in `run_fused_decoder.py`, but there was no recorded
  candidate result at the review cutoff. The selected linear profiler still
  contains the conversion this candidate addresses.
  Why this matters: this is a concrete applicable op-merging experiment under
  the stage's "try every applicable pattern" contract, and splitting the
  projection may either offset or exceed the saved conversion cost.
  Required next step: record the candidate's correctness and comparable
  traced-decode measurement, or an exact demonstrated operation-contract
  blocker. Adapt an initial API/layout/padding failure as required by
  graph-fusing. Preserve the fastest correct resulting default and its final
  evidence. No broader matmul geometry/precision optimization is requested.

- P2: Finish the stage handoff documentation against the final evidence.
  Evidence: `README.md` currently contains the baseline graph and says final
  graph/performance conclusions will be recorded later. `patterns.md` still
  labels dedicated decode concat and matmul activation assessments pending,
  although raw results and the updated work log already establish those
  experiments. The work log's final paragraph still says full context is
  running, although its four rows and clean close are now present.
  Why this matters: the goal requires reproducible commands, final configs,
  PCC, performance conclusions, rejected candidates, limitations and artifact
  links. The current files are explicitly an unfinished handoff.
  Required next step: update README, pattern decisions and work log with the
  final selected graph, reproduced timing table, context results, replay and
  watcher evidence, candidate rejections, relevant command/config details,
  and classified anomalies. Clearly distinguish measured direct HF PCC from
  the mathematically derived long-context HF lower bounds. Retain the actual
  supported-context contract and disclose separate batch/context coverage.
  Obtain a later independent clean review before stage-owned local commits;
  the absence of those pre-review commits is not itself a defect.

## Other Concerns

- No later-stage or native implementation edits were present in the inspected
  status. The new implementation is a separate `FusedDecoder`, not a wrapper
  that dispatches to `FunctionalDecoder`; the regression test makes functional
  construction fail, and both synthetic layer tests pass.
- Existing state/trace evidence restores the prefix between repeated decode
  replays. Continuation checks exercise accumulated state, but this review
  does not infer continuous full-model generation accuracy or serving safety.
  Those are later stages, not extra gates for this decoder review.
- The paged-cache sentinel check inspects unowned pages after prefill. It is
  not an explicit post-decode sentinel check. Source inspection of disjoint
  K/V core allocation, direct output parity, changed per-user positions,
  refreshed page-table routing and clean watcher evidence did not reveal a
  concrete corruption issue requiring an additional test here.
- Benchmark precision remains BF16 weights/activations and HiFi4 projections,
  with FP32 linear state and BF16 full-attention KV. No datatype-stage sweep
  or subsequent optimized-decoder work is required by this review.

## Hard-Check Gaps

- This was source and artifact inspection only. The reviewer did not import
  TTNN, open/reset devices, launch tests/profilers/servers, or install software.
  Source files parse with Python's AST parser; no native build was required
  for the inspected Python/docs-only scope. No pre-commit run is claimed.
- Source hashes exist for current watcher/profile/context evidence; historical
  candidate snapshots and hashes distinguish earlier graphs. A different or
  stronger provenance format is not necessary where these records already
  establish the measured path.
- PCC is aggregate over logical output elements. Long-context HF bounds reuse
  frozen functional-stage HF correlations with the same source, checkpoint,
  seeded inputs and page permutation. This is a correlation-angle bound,
  not a newly executed HF run. The runner checks all outputs and hashes its
  baseline evidence; the approach is valid for this fusion equivalence task.
- The reviewed context manifest points to functional-stage capability evidence.
  Current fused context results now independently support retaining its
  262144 capability. Stage documentation should link those results; a new
  unrelated manifest schema is not requested.

## Anomaly Ledger

- Observed anomaly: benchmark files called final belong to an earlier graph.
  Evidence: matching `227e09b6...` source hashes and `candidates/l1_rope.py`;
  later candidate JSONs and current source/profile hashes.
  Affected path: reported final warmed performance and candidate selection.
  Control or comparison: all current raw profiler windows are valid; later
  candidate decode timings are faster than the earlier benchmark files.
  Likely subsystem: evidence finalization/provenance, not a demonstrated runtime
  regression.
  Investigation performed: matched SHA256 against snapshots, compared changed
  methods, and independently recomputed medians from recorded samples.
  Resolution: more-work-needed; first required item.

- Observed anomaly: initial RMSNorm sharding probes failed, including the
  first larger-batch boundary run.
  Evidence: `sharded_norm_full.log`, `boundaries_full_initial.log` report
  unsupported HEIGHT_SHARDED norm and too many shard rows, respectively.
  Affected path: full-attention Q/K normalization.
  Control or comparison: `sharded_norm_full_adapted.json`, full boundary
  sweeps, current synthetic tests and batch-1/batch-32 watcher results pass.
  Likely subsystem: native normalization/shard-layout restrictions.
  Investigation performed: inspected the batch-1 block-sharded branch and
  larger-batch interleaved branch and their actual recorded coverage.
  Resolution: fixed and controlled for the claimed full-attention batches.

- Observed anomaly: dedicated decode concatenation failed twice before the
  adapted padded-batch form completed.
  Evidence: `decode_concat_full.log` rejects GQA sharded SDPA output;
  `decode_concat_full_adapted.log` reports reshape volume mismatch;
  `decode_concat_full_padded.json` passes with decode median 2.31216 ms.
  Affected path: full-attention decode output layout.
  Control or comparison: direct flatten remains faster; subsequent native
  decode-layout result is 2.25999 ms, with passing full boundary checks.
  Likely subsystem: SDPA output contract and concat's padded batch geometry.
  Investigation performed: inspected retained adapted candidates and successful
  result rather than treating either first API failure as rejection evidence.
  Resolution: controlled; earned slower-candidate rejection, pending final
  pattern-documentation update.

- Observed anomaly: initial causal-convolution candidate used an invalid Python
  configuration namespace.
  Evidence: `conv_linear.log` AttributeError and `conv_linear_adapted.json`.
  Affected path: native convolution configuration construction.
  Control or comparison: corrected top-level config export passes real-weight
  PCC, and the selected convolution appears in current profiler rows.
  Likely subsystem: binding namespace.
  Investigation performed: checked corrected invocation and subsequent results.
  Resolution: fixed.

- Observed anomaly: full-context HF agreement is lower than short-context
  agreement.
  Evidence: functional full prefill PCC 0.9967552062 at 262143; measured
  fused/functional PCC 0.9999614744 gives lower bound 0.9960102579. At
  262144, the lower bound is 0.9960646314; final-context decode is bounded
  below by 0.9974338615.
  Affected path: long full-attention prefill and subsequent decode.
  Control or comparison: all bounds and direct equivalence pass the unchanged
  0.995 bar; both full context runs finish all four cases including short reuse.
  Likely subsystem: inherited long-context numerical accumulation; this review
  does not claim a component-level error attribution.
  Investigation performed: checked seed/page construction, baseline/source
  hashes, exact-limit handling, every result row and matching console JSON.
  Resolution: controlled within the required accuracy bar, without a context
  cap or threshold relaxation.

- Observed anomaly: clock-settling and profiler tooling warnings appear in
  successful runs.
  Evidence: 1343 MHz versus requested 1350 in `padded_scan_linear.log`,
  1337 MHz in `synthetic_pytest.log`; optional viewer-copy and pandas
  mixed-column warnings in profiler logs.
  Affected path: measurement environment and optional viewer export/parser.
  Control or comparison: original host Tracy captures exist for all four
  profiles; copied ops CSVs match originals exactly; all eight measured windows
  have complete device timing, and raw/filtered totals agree. Watcher logs
  contain no suspicious fault messages.
  Likely subsystem: clock settling and host tooling.
  Investigation performed: inspected warning signatures, retained capture
  sizes, raw signpost windows, trace IDs, and CSV totals. No numerical failure
  or incomplete profile is implied by these warnings.
  Resolution: controlled for existing evidence; document measurement variation
  when deciding among closely timed candidates and report final reproduced
  results instead of combining minima.

## Scope Inspected

- Goal/skill paths: supplied Stage 2 contract and AGENTS.md instructions;
  `.agents/skills/{stage-review,graph-fusing,tt-device-usage}/SKILL.md`;
  installed `tt-model-bringup/0.1.4/skills/stage-review/SKILL.md`, including
  additional changed-boundary coverage requirements, and its startup reference.
- Code paths: complete `tt/fused_decoder.py`; `tests/run_fused_decoder.py`,
  `test_fused_decoder.py`, `run_fused_context.py`, `untilize_candidate.py`,
  both stage shell runners; retained candidate diffs; functional baseline
  contract/review; native delta-rule adapter, KDA operation contracts and
  selected normalization API details.
- Artifact paths: `doc/context_contract.json`; functional-stage README and
  `stage_review_2.md`; fused README/work log/patterns, candidate JSONs and
  metrics CSV, boundary/synthetic/watcher/context JSON/logs, source hashes,
  current profiler JSON/logs, all eight filtered CSV/tables, original and
  copied ops CSVs and retained host traces.
- Commands run: read-only `cat`, `sed`, `nl`, `rg`, `diff`, `git status`;
  standard-library Python JSON/CSV/XML/AST, SHA256, statistics and log analysis.
  The sole reviewer mutation was writing this report. No subagent was spawned.

Independently rederived evidence:

| Requirement | Result |
|---|---|
| Separate fused runtime | Dedicated packed projections, convolution, flat scan, gated norm, native head creation/RoPE and dual paged update are directly wired. Current profile rows reflect those operations; functional fallback guard passes. |
| Correctness | All inspected successful candidate/runtime JSON records pass >=0.995; corresponding console JSON records agree. Current real watcher cases pass both kinds at batches 1 and 32. |
| Synthetic boundaries | JUnit: 2 tests, zero errors/failures/skips, 16.125 seconds. Each kind covers 1,2,3,31,32,33,127,128,129,257,31 at batch 2, continuation and refreshed trace inputs. Minimum prefill/decode PCC is linear 0.9960617423/0.9960767031 and full 0.9983770814/0.9975596666. |
| Full capability | Both current-source context runs contain 4097,262143,262144,31 with clean teardown and matching baseline hashes. No context reduction. Exact maximum prefill and final valid decode are covered; no out-of-contract extra decode at maximum prefill. |
| Trace/device-only runtime | Runtime Torch dispatch and host-conversion guards surround forward/capture; native scan constants and initial state are caller-supplied. Profiler decode rows all carry trace ID 0, and every measured window contains only device rows on device 3. |
| Watcher | Four current-source watcher logs exist with expected sizes and zero fatal/exception/invalid/overflow/out-of-bounds/error/corrupt/sanitize matches. Separate profiler runs retain complete evidence. |
| Baseline profiler | Linear prefill/decode: 88/100 ops, kernel totals 4604.284/3011.936 us. Full: 52/57 ops, 3416.256/2376.279 us. |
| Current fused profiler | Linear prefill/decode: 32/38 ops, 2828.365/2429.051 us. Full: 29/32 ops, 2587.935/2227.059 us. Raw nanosecond sums exactly reconcile with filtered CSV microseconds. These are kernel sums, not wall-latency medians. |
| Precision/movement | Selected projection rows show BF16 x BF16 => BF16 with HiFi4; no selected matmul is marked SLOW. Native conversion/state operations are visible, with no host-op row hidden in measured windows. |

## Residual Risk

This review concerns one-device representative decoder layers and the recorded
pinned environment. It does not establish later-layer/full-model accumulated
accuracy, generation quality, multi-device behavior, serving integration,
every context/batch combination, or long-run firmware stability. Resolving the
four required items should be followed by another independent review of the
final artifacts; existing unaffected correctness/profiler/context evidence can
be reused rather than rerun wholesale.
