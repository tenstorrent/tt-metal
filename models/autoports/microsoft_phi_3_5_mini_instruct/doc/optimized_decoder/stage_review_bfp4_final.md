# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Sweep dominant decode geometry under the shipped BFP4/LoFi policy.
  Evidence: `optimized_decoder.py` ships BFP4 attention and BFP4 down, but
  `AUTOFIX.md` records the 16/32-core, `in0_block_w`, and HiFi2/LoFi geometry
  matrices under BFP8 for QKV/output/down (gate/up was already BFP4). The final
  BFP4 frontier measures attention through TTNN automatic matmuls and measures
  down only through the already-selected 16-shard, `in0_block_w=8` topology.
  `tracy/decode_bfp4_final.txt` confirms the shipped dtype, but is not a
  BFP4/LoFi geometry matrix.
  Why this matters: The optimize contract and review gate explicitly prohibit
  carrying a dominant geometry conclusion across precision policies. QKV,
  gate/up, and down are the material decode rows, and the user specifically
  required BFP4/LoFi across geometries where MLP matmuls dominate.
  Required next step: Re-run the legal material geometry candidates at both B1
  and B32 with the shipped BFP4 policy (including larger legal K blocks and
  core/shard alternatives), gate with real-weight whole-layer PCC and traced
  whole-layer timing, then retain the best non-regressing default or record an
  exact blocker for each unmeasurable family.

- P1: Repair and preserve the precision-frontier reproducer and raw evidence.
  Evidence: The documented matrix in
  `AUTOFIX_BFP4_PRECISION_FRONTIER.md` distinguishes shipped BFP8,
  attention-BFP4, down-BFP4, and combined-BFP4 candidates. The current
  `tests/optimized_bfp4_frontier_experiment.py` constructs all four policies
  from `OptimizationPolicy` defaults that are now BFP4, so every named
  candidate is the same combined-BFP4 policy. `/tmp/phi_bfp4_frontier.txt`
  demonstrates this failure: all four candidates have identical PCC and
  essentially identical timing. The corrected run exists only in the
  untracked `/tmp/phi_bfp4_cumulative_fixed.txt`; the source that produced it
  and its raw output are not preserved in the stage evidence.
  Why this matters: The selected precision policy depends on a candidate
  comparison that the checked-in stage-owned reproducer cannot reproduce.
  Rejected/selected precision evidence must remain attributable to explicit
  policies rather than mutable defaults.
  Required next step: Make every frontier policy explicit (BFP8/BFP4 for all
  relevant groups), rerun at B1/B32, and preserve the raw command output under
  `doc/optimized_decoder`.

- P1: Preserve an actual clean final correctness/watcher runner artifact.
  Evidence: The nearest raw final BFP4 correctness artifact,
  `/tmp/phi_bfp4_full_correctness.txt`, reports `1 failed, 9 passed`; its
  runtime-dispatch assertion failed because a source comment contained
  `FunctionalDecoder`. The source comment was corrected afterward.
  `watcher_bfp4_final.txt` is a hand-written 451-byte summary claiming ten
  passes, not raw pytest/watcher output. The older
  `correctness_cache_trace_final.txt` and `watcher_cache_trace_final.txt`
  predate the final BFP4 default and have different PCC values.
  Why this matters: The goal requires a watcher-clean optimized correctness
  run, and the stage-review skill treats README/work-log summaries as claims,
  not runner evidence. The only preserved raw final-BFP4 run is failing.
  Required next step: Rerun the complete final suite with watcher separately
  from profiling and preserve the raw successful output in a tracked
  stage-owned artifact.

- P2: Complete the prefill program-config rejection evidence.
  Evidence: `tracy/prefill_program_config_final.txt` profiles only TTNN's
  automatic prefill choice. Its material QKV/output/gate-up/down rows all use
  `in0_block_w=2`; no explicit before/after large-program candidates are
  recorded. The conclusion that explicit configuration would not remove the
  B1 gap is asserted without a measured explicit configuration, and the
  README does not call out the required attempts to enable larger legal block
  widths.
  Why this matters: The user required large prefill program-config evidence,
  and the optimize checklist requires explicit treatment of material
  `in0_block_w<=2` rows rather than accepting one automatic config as a sweep.
  Required next step: Measure representative larger legal block/program
  candidates for the material prefill projections at B1 and B32, or preserve
  exact validation/L1/divisibility blockers after adapting the program
  contract. Record before/after device and warmed latency.

## Other Concerns

- `README.md` reports the final optimized B1 result as 0.646986 ms in the
  headline table but later says “Final B1 e2e is 0.668 ms.” The latter is the
  superseded BFP8 checkpoint and should be corrected.
- `README.md` calls `tracy/ops_final.csv` primary final evidence even though
  `work_log.md` correctly labels it an intermediate BFP8 profile. The final
  BFP4 source CSVs remain under ignored `generated/profiler/reports`; preserve
  or clearly identify the final stage-owned reports so the final policy is not
  conflated with the intermediate profile.
- The context contract itself is preserved: the optimized test directly
  exercises logical context 131072, and KV-cache dtype/layout remain BF16 and
  inherited. No capability reduction was found.

## Hard-Check Gaps

- Stage-owned remediation remains uncommitted on branch
  `skillexp-cell/fuse-noadvise/phi`; the latest local checkpoint is
  `774d6b2f32a`. A clean review and isolated local commit are still required
  after the findings above are closed.

## Anomaly Ledger

- Observed anomaly: All four policies in the current BFP4 frontier reproducer
  resolve to the same policy.
  Evidence: Current dataclass defaults plus policy construction in
  `optimized_bfp4_frontier_experiment.py`; identical PCC rows in
  `/tmp/phi_bfp4_frontier.txt`.
  Affected path: Precision selection evidence.
  Control or comparison: `/tmp/phi_bfp4_cumulative_fixed.txt` shows distinct
  results from a corrected but unpreserved source state.
  Likely subsystem: Mutable experiment meaning after promoting BFP4 defaults.
  Investigation performed: Static policy expansion and direct raw-log
  comparison.
  Resolution: more-work-needed.

- Observed anomaly: The raw final BFP4 correctness run failed while the final
  summary claims ten passes.
  Evidence: `/tmp/phi_bfp4_full_correctness.txt` versus
  `watcher_bfp4_final.txt`.
  Affected path: Final optimized correctness and watcher gate.
  Control or comparison: Failure was source-contract-only and the current
  comment is corrected, but no raw successful final run is preserved.
  Likely subsystem: Evidence ordering/preservation.
  Investigation performed: Read the failure traceback and current source.
  Resolution: more-work-needed.

- Observed anomaly: Initial profiler marker overflow.
  Evidence: Work log and bounded final profiler summaries.
  Affected path: Tracy evidence.
  Control or comparison: Bounded final source CSVs exist and report zero host
  ops.
  Likely subsystem: Profiler capacity.
  Investigation performed: Verified referenced generated reports exist.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths:
  - `.agents/skills/stage-review/SKILL.md`
  - `.agents/skills/optimize/SKILL.md`
  - `.agents/skills/tt-device-usage/SKILL.md`
  - Supplied optimized-decoder goal contract
- Artifact paths:
  - `doc/optimized_decoder/README.md`, `work_log.md`, both AutoFix reports,
    correctness/watcher artifacts, profiler summaries and referenced generated
    CSVs, prior review, and `doc/context_contract.json`
  - Raw `/tmp/phi_bfp4_*` outputs referenced by the stage report
- Code paths:
  - `tt/optimized_decoder.py`
  - `tests/test_optimized_decoder.py`
  - `tests/optimized_decoder_perf.py`
  - `tests/optimized_bfp4_frontier_experiment.py`
  - inherited functional/fused construction contract
- Commands run:
  - Read-only `sed`, `grep`, `find`, `git status`, `git diff`, and artifact
    existence/size checks
  - No hardware, tests, servers, or implementation changes

## Residual Risk

- Direct cache transition, non-aligned sequence lengths, B1/B32 decode,
  five-replay determinism, long-context decode, and final profiler dtype rows
  are represented in code/evidence and appear sound.
- Runtime source contains no Torch conversion or functional fallback, and the
  bounded BFP4 profiler summaries report zero host ops.
- A later clean-pass requires remediation plus a new independent rereview; this
  report does not authorize stage completion.
