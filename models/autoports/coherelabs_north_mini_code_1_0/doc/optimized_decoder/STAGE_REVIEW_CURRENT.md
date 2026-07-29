# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Re-evaluate BFP4 attention on authentic inputs and the final attention
  topology.
  Evidence:
  The selected default keeps attention weights at BFP8/LoFi
  (`tt/optimized_decoder.py:33,46`). The only retained all-attention BFP4
  timing artifacts, `candidates/all_bfp4_lofi_b{1,32}.json`, are screening
  runs at 0.195010/0.257612 ms using a 32-core output projection with
  `in0_block_w=4`. The final lowest-op default instead uses the packed-QKV,
  16-core output-projection topology with `in0_block_w=8` and measures
  0.186997/0.252220 ms
  (`candidates/review3_final_runtime/layer0_decode_b{1,32}.json`). BFP4 was
  not remeasured on that final topology.

  More importantly, `work_log.md:56-64` says BFP4 attention was rejected by a
  preliminary **real-weight/random-activation** layer-0 PCC of 0.988737.
  Random activation is synthetic for the current optimize guardrail. The
  current real-weight test can select `all_bfp4_lofi` through
  `NORTH_MINI_PRECISION_POLICY`
  (`tests/test_optimized_decoder.py:1378-1448`), but both
  `artifacts/current_full.xml` and `artifacts/current_watcher.xml` ran its
  default `selected` policy. No retained XML/log contains
  `optimized-real-layer0-{decode,prefill}-all_bfp4_lofi`; the only repository
  references to `all_bfp4_lofi` are the two timing JSONs and the test switch.
  `README.md:151` therefore overstates the evidence when it says BFP4
  attention failed authentic PCC.

  The current optimize skill explicitly requires real weights or recorded
  real activations for a BFP4 attention veto, separate attention precision
  from MLP/cache precision, and a rerun on the lowest-op compatible topology
  (`.agents/skills/optimize/SKILL.md:166-172,470`). The current stage-review
  standard likewise makes a synthetic-only precision veto required work
  (`.agents/skills/stage-review/SKILL.md:98-106,229-252`).
  Why this matters:
  The retained failure neither represents target activations nor isolates
  QKV from output projection, and its timing does not use the final topology.
  It cannot establish that BFP8/LoFi is the fastest correct attention policy.
  Required next step:
  On the final packed-QKV/16-core-output topology, test BFP4 QKV and BFP4
  output projection separately and cumulatively with each plausible
  fidelity. Use checkpoint weights and propagated/recorded target
  activations, cover b1 and b32 traced decode, and include a cache-consuming
  replay or prefill-to-decode transition. Re-run prefill precision separately
  rather than changing attention and MLP precision together. Select the
  fastest passing policy or retain BFP8 with an authentic model-visible
  failure, unacceptable final-topology latency, or exact op-contract blocker.

- P1: Sweep output subblocks separately for the dominant sparse gate, up, and
  down projections.
  Evidence:
  `_sparse_program()` hard-codes `out_subblock_h/w=1/1` and
  `out_block_h/w=1/1` (`tt/optimized_decoder.py:201-215`). Sparse grid and
  `in0_block_w` are configurable, but sparse output block/subblock fields do
  not exist in `OptimizationConfig`, the candidate harness, or the retained
  search tables. The current profiles confirm every selected sparse row still
  uses 1x1:

  | Path | Sparse rows | Sparse device time | Share of device window | Subblock |
  |---|---:|---:|---:|---:|
  | layer 1 decode b1 | 3 | 438.546 us | 57.32% | 1x1 |
  | layer 4 decode b1 | 3 | 443.395 us | 57.98% | 1x1 |
  | layer 1 prefill b1 | 18 | 11,697.596 us | 85.31% | 1x1 |
  | layer 4 prefill b1 | 18 | 11,740.782 us | 85.61% | 1x1 |

  These figures are re-derived from
  `tracy/review3_selected/layer{1,4}_{decode,prefill}_b1/filtered.csv`.
  Their advice-enabled `human_report.txt` files repeatedly recommend an
  output subblock area of at least two. The source and artifacts contain no
  attempted larger sparse subblock and no L1/divisibility/API blocker.
  Nevertheless `work_log.md:255-259` marks output blocks/subblocks as swept
  separately by dominant role.

  The current optimize skill specifically requires independent
  `in0_block_w` and output-subblock searches for each dominant sparse
  projection role, and says one failing or hanging candidate does not end the
  legal search (`.agents/skills/optimize/SKILL.md:246-249,468,477`).
  Why this matters:
  These are the majority of selected b1 MoE device time, including the
  primary serving batch. The retained grid/block-width sweep does not answer
  the profiler's separate output-subblock opportunity.
  Required next step:
  Expose legal sparse output-block/subblock candidates and sweep gate, up, and
  down independently under final BFP8/LoFi, including 2x1, 1x2, and larger
  legal areas where divisibility/register limits permit. Measure selected
  b1 traced decode and representative aligned/non-aligned b1 prefill with
  target weights and propagated/recorded routing activations. Keep the winner
  or retain the exact per-role L1, register, divisibility, PCC, hang, or
  op-contract blocker, then reproduce and reprofile the final default under
  watcher.

- P1: The batch-32 MoE runtime still uses the dense all-expert debug
  baseline, so the current optimize checklist is not complete.
  Evidence:
  The selected threshold is 32 tokens
  (`tt/optimized_decoder.py:83`). Except for b1 prefill, `_sparse_moe()` sends
  `total_tokens >= 32` directly to `_dense_expert_moe_chunk()`
  (`tt/optimized_decoder.py:1367-1372`). Thus both b32 decode and b32 prefill
  execute all 128 experts. `AUTOFIX.md:1-152` explicitly acknowledges that
  this does not satisfy the active-expert requirement. It establishes strong
  model-local blocker evidence: dynamic sparse is 20.535-21.896 ms,
  exact-static-`nnz` sparse is correct at 17.831 ms, packed static sparse is
  correct at 19.584 ms, and the fast single-card `moe_compute` path cannot
  return all routed contributions while full combine requires unavailable
  fabric handshakes. The selected dense path is correct and substantially
  faster, but that investigation proves a current TTNN output/API limitation;
  it does not turn the required implementation path into a completed
  checklist item.

  The current optimize skill says dense all-expert execution is a debug
  baseline rather than the optimized target and its final checklist requires
  “no dense all-expert runtime path”
  (`.agents/skills/optimize/SKILL.md:96,249,477`). The original stage contract
  requires the current optimize checklist to be complete.
  Why this matters:
  The final b32 performance is strong (2.214/2.219-ms decode and
  96.750/96.440-ms prefill for layers 1/4), but it is achieved by omitting a
  required routed-MoE implementation path, not by optimizing that path.
  Required next step:
  Provide a single-device compact routed-output or fabric-free combine
  contract in TTNN, wire it into the decoder, and prove real-weight
  correctness plus traced b32 no-regression; then add a branch guard that
  forbids the dense fallback on the selected optimized policy. If shared TTNN
  work is intentionally excluded, the goal/skill contract must be explicitly
  changed by the stage owner; a model-local scope statement or prose
  acknowledgement cannot make the current checklist complete.

## Other Concerns

- All 12 final runtime JSONs, including the route-sensitive b1 MoE rows, label
  their weights as `deterministic_synthetic_full_shape_recorded_marginals`
  and activations as `deterministic_synthetic`. The authentic correctness
  matrix is strong, but the headline performance and sparse route-union
  behavior have not been reproduced with recorded target activations. The
  remediation searches above should use target values so program and
  route-sensitive decisions are not made solely from synthetic distributions.
- `artifacts/current_watcher.xml`, the fresh watcher log, and the accompanying
  README/work-log updates were live uncommitted evidence when inspected. This
  is expected while review is in progress, but they and this review report
  still need the normal stage-owned local checkpoint commit. No push is
  required.
- The raw post-run `tt-smi` output is not retained as a separate artifact.
  The stage owner reports live heartbeat, healthy DRAM, zero GDDR errors, and
  zero thermal trips on all four p300c boards. This is a durability gap, not a
  reason to dispute the fresh clean watcher run.

## Hard-Check Gaps

- No authentic final-topology BFP4 attention PCC/trace/latency artifact exists;
  this is required work above rather than a preferred evidence format.
- No sparse gate/up/down output-subblock candidate table or exact blocker
  exists; this is required work above.
- No selected batch-32 routed active-expert full-output path exists; AutoFix
  identifies the shared TTNN/API gap but the required path remains absent.
- Current JUnit proves test outcomes but does not retain stdout PCC values.
  Earlier exact logs and the test's normal PCC assertions make this an
  evidence-format limitation rather than an additional blocker.
- This independent review did not open hardware, run tests/profiling, start
  servers, reset devices, or run vLLM. It used current machine-readable
  artifacts plus the separately completed fresh watcher run.

## Anomaly Ledger

- Observed anomaly:
  README calls the BFP4 attention failure authentic, while the work log names
  random activation and no authentic all-BFP4 attention result is retained.
  Evidence:
  `README.md:151`, `work_log.md:56-64`, the two all-BFP4 timing JSONs,
  the policy-selectable test, and absence of a matching XML/log result.
  Affected path:
  Layer-0 attention decode and prefill precision selection.
  Control or comparison:
  Current selected BFP8/LoFi real-weight test and final BFP8 runtime/profile.
  Likely subsystem:
  Precision-policy evidence and final-topology crossing.
  Investigation performed:
  Traced the test's environment policy, compared candidate policy/geometry
  fields, searched all retained artifacts, and compared the claim with the
  current optimize guardrail.
  Resolution:
  more-work-needed.

- Observed anomaly:
  The checklist says output blocks/subblocks were swept separately by
  dominant role, but every sparse program is hard-coded to 1x1 and no larger
  sparse subblock result exists.
  Evidence:
  `_sparse_program()`, `OptimizationConfig`, all current sparse profiler rows,
  advice-enabled reports, candidate search, and `work_log.md:255-257`.
  Affected path:
  Layer-1/layer-4 active-expert b1 prefill and decode.
  Control or comparison:
  The valid existing sparse grid and `in0_block_w` sweep.
  Likely subsystem:
  Sparse matmul program geometry.
  Investigation performed:
  Recomputed sparse device-time shares from filtered CSVs and searched
  source/tests/docs/candidates for role-specific output-subblock controls or
  blockers.
  Resolution:
  more-work-needed.

- Observed anomaly:
  Batch 32 is described as checklist-complete despite selecting dense
  all-expert execution.
  Evidence:
  Threshold/dispatch source, `AUTODEBUG.md`, `AUTOFIX.md`, final profiler
  topology, and current optimize checklist language.
  Affected path:
  Layer-1/layer-4 b32 decode and prefill.
  Control or comparison:
  Correct active-expert b1 path and the measured dynamic/static/packed/fused
  b32 candidate families.
  Likely subsystem:
  Single-device routed MoE output/combine API.
  Investigation performed:
  Traced current dispatch and reviewed the complete retained AutoDebug/AutoFix
  candidate/blocker chain.
  Resolution:
  more-work-needed; the limitation is well investigated but the required
  implementation remains absent.

- Observed anomaly:
  Prior Review 5 required OPT-015 shard-advisor evidence that no longer exists
  in the current arm.
  Evidence:
  Current `.agents/skills/optimize/SKILL.md`, arm commit `617056c`, and the
  README/work-log reconciliation of retained historical reviews.
  Affected path:
  Historical stage-process evidence only.
  Control or comparison:
  The current manual topology/program search requirements and retained
  candidate matrices.
  Likely subsystem:
  Skill evolution/history.
  Investigation performed:
  Compared the current contract with the retained Review 5 history and
  current arm boundary.
  Resolution:
  controlled. OPT-015 is intentionally removed and is not a current finding.

- Observed anomaly:
  Earlier final-policy batch-32 MoE prefill regressed to about 140 ms and broad
  watcher evidence predated the eventual packed-prefill promotion.
  Evidence:
  Review 5 history, `PREFILL_GEOMETRY_AUTOFIX.md`,
  `candidates/review5_prefill/`, current full-suite JUnit, and fresh
  current-arm watcher evidence.
  Affected path:
  Layer-1/layer-4 b32 MoE prefill and current-arm device safety.
  Control or comparison:
  Promoted packed 80-core default at 96.750/96.440 ms, authentic PCC
  0.99923857/0.99993403, and current watcher run.
  Likely subsystem:
  Phase-specific packed dense-expert prefill geometry.
  Investigation performed:
  Recomputed current/default versus candidate timing, inspected the focused
  watcher/profile rows, and parsed the fresh current-arm watcher XML/log.
  Resolution:
  fixed. The fresh run records 54 tests, 38 passes, 16 opt-in DRAM-candidate
  skips, zero failures/errors, and no fatal/NoC/CB/overflow/sanitizer/
  timeout/hang/tripped/kernel/watcher/assert signature in 3,247 log lines.

## Scope Inspected

- Goal/skill paths:
  functional checkpoint `78dbd88bec7`, current-arm boundary `617056c`, HEAD
  `8a1a5011e9d`, `.agents/skills/stage-review/SKILL.md`,
  `.agents/skills/optimize/SKILL.md`, and
  `.agents/skills/tt-device-usage/SKILL.md`.
- Artifact paths:
  current README/work log, `STAGE_REVIEW_1.md` through
  `STAGE_REVIEW_5.md`, AutoDebug/AutoFix and prefill-geometry reports, all
  current/final/candidate JSON and JUnit XML, context-contract/capacity
  evidence, raw compressed/filtered/human-readable Tracy reports, focused and
  broad watcher logs, and the fresh `current_watcher.xml`/`current_full`
  watcher tree.
- Code paths:
  complete `tt/optimized_decoder.py`, `tests/test_optimized_decoder.py`,
  `tests/test_optimized_decoder_prefill_geometry.py`,
  `tests/optimized_decoder_perf.py`, and
  `tests/optimized_decoder_capacity.py`.
- Commands run:
  read-only `git status/log/show/diff`, `rg`, `find`, `sed`, `nl`, `wc`,
  gzip integrity checks, AST parsing, JSON/XML parsing, profiler CSV
  aggregation, policy-field comparison, and watcher-signature scans. No
  device or server process was launched by the reviewer.

## Residual Risk

- Functional evidence is otherwise strong. Current selected-path suites pass
  38 tests normally and under watcher with 16 explicitly opt-in DRAM-family
  skips. Real-weight representative layer kinds, non-aligned prefill, paged
  cache slots, traced decode, ten-replay determinism, and the advertised
  500,000-token capacity have current evidence.
- Final warmed performance beats the functional baseline at primary b1 and
  preserves/improves b32. Current profiles are parseable, advice-enabled,
  dtype-consistent, zero-host-op, and accompanied by valid raw gzip CSVs.
- The remaining stage risk is optimization-contract completeness rather than
  a demonstrated selected-default correctness or watcher failure. Authentic
  final-topology attention precision and dominant sparse subblock searches
  are still missing, and b32 depends on a known shared TTNN limitation that
  leaves the required routed active-expert path unimplemented.
