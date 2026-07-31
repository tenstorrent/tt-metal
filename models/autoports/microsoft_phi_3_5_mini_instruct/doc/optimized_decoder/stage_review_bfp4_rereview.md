# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Reprofile the final block-16 decode and explicit-prefill-down default.
  Evidence: The final implementation now uses decode down
  `in0_block_w=16` and a 64-core prefill down with `in0_block_w=8`.
  `tracy/decode_bfp4_final.txt`, `decode_b1_bfp4_final.csv`,
  `decode_b32_bfp4_final.csv`, `prefill_b1_bfp4_final.csv`, and
  `prefill_b32_bfp4_final.csv` were collected before those promotions: the
  decode summary describes the former block-8 down path, while the prefill
  report describes automatic down with `in0_block_w=2`. README nevertheless
  labels those reports primary shipped-policy evidence and says the final
  reports show the optimized down at 51–52 us.
  Why this matters: The goal requires final measured prefill/decode runtime
  without host fallback and tt-perf-report conclusions. The final default was
  correctly reproduced by the timing harness, but the profiler artifacts used
  for the runtime-row and zero-host-op claims describe the superseded program
  configs.
  Required next step: Run bounded Tracy/tt-perf-report separately from watcher
  on the final default at B1 and B32 for decode and prefill. Preserve the final
  CSV/reports under the stage evidence, verify block-16 decode and explicit
  block-8 prefill-down rows, zero host ops/fallbacks, and update README/work log
  row timings and conclusions.

## Other Concerns

- README's optimize checklist still emphasizes the old automatic prefill
  configuration and `tracy/prefill_program_config_final.txt`; revise it to
  state that automatic configs are retained for the roles that rejected or
  lost explicit candidates, while down now uses the adaptive explicit
  64-core/block-8 program.
- The final code computes prefill `per_core_M` from logical padded shape and
  has direct non-aligned S31/S33/S65 evidence. The implementation and tests do
  support the documented adaptive prefill state.

## Hard-Check Gaps

- Stage-owned changes are still uncommitted. After a clean final profiler
  rerun and rereview, create the isolated local checkpoint and record its SHA;
  do not include unrelated GPT-OSS/prompt artifacts.

## Anomaly Ledger

- Observed anomaly: Final timing/correctness artifacts describe block-16
  decode and explicit prefill down, while the final profiler artifacts
  describe block-8 decode and automatic prefill down.
  Evidence: Current `optimized_decoder.py`,
  `perf_bfp4_block16_prefill_down_runner.txt`, and
  `correctness_prefill_down_explicit_runner.txt` versus
  `tracy/decode_bfp4_final.txt` and
  `tracy/prefill_program_config_final.txt`.
  Affected path: Final runtime profiler evidence.
  Control or comparison: Final same-harness timing passes at B1/B32 and the
  complete correctness/watcher suite passes.
  Likely subsystem: Evidence ordering after late program-config promotion.
  Investigation performed: Compared source configs, raw runners, profiler
  summaries, and README claims.
  Resolution: more-work-needed.

- Observed anomaly: The former precision-frontier reproducer inherited mutable
  BFP4 defaults.
  Evidence: Explicit policies now appear in
  `optimized_bfp4_frontier_experiment.py`; tracked
  `bfp4_precision_frontier_runner.txt` has distinct BFP8, attention-BFP4,
  down-BFP4, and combined-BFP4 rows at B1/B32.
  Affected path: Precision selection.
  Control or comparison: Combined BFP4 wins and passes whole-layer PCC.
  Likely subsystem: Experiment policy propagation.
  Investigation performed: Static source inspection and raw runner review.
  Resolution: fixed.

- Observed anomaly: Geometry evidence previously crossed precision policies.
  Evidence: `bfp4_lofi_decode_geometry_runner.txt` now sweeps shipped BFP4/LoFi
  at B1/B32 over 16/32 cores and legal block widths; block-16 down wins.
  Affected path: Dominant decode geometry.
  Control or comparison: Full-layer block-16 correctness and performance
  runners pass.
  Likely subsystem: Precision-locked program tuning.
  Investigation performed: Reviewed full raw matrix and promoted default.
  Resolution: fixed.

- Observed anomaly: The previous final raw BFP4 suite had one source-contract
  assertion failure and only a prose watcher summary.
  Evidence: `correctness_prefill_down_explicit_runner.txt` records ten passes;
  pytest-generated `watcher_bfp4_final.xml` records tests=10, failures=0,
  errors=0 for the final block-16/explicit-prefill default.
  Affected path: Correctness/watcher gate.
  Control or comparison: Raw PCC rows cover non-aligned prefill, B1/B32 decode,
  paged transition, context 131072, and trace replay.
  Likely subsystem: Evidence preservation.
  Investigation performed: Reviewed raw runner and JUnit cases.
  Resolution: fixed.

- Observed anomaly: Material prefill rows previously retained
  `in0_block_w=2` without explicit alternatives.
  Evidence: `prefill_explicit_config_runner.txt` measures block 4/8 at B1/B32,
  records exact B32 L1 blockers for QKV/gate-up, and shows legal output/down
  candidates. Final down promotion passes correctness and improves warmed
  prefill to 1.351156/24.148464 ms.
  Affected path: Prefill program configs.
  Control or comparison: Automatic rows and explicit candidates use the same
  real weights/shapes in the runner.
  Likely subsystem: Program config/L1 capacity.
  Investigation performed: Reviewed candidate matrix, errors, final code, and
  final timing/correctness runners.
  Resolution: fixed.

## Scope Inspected

- Goal/skill paths:
  - `.agents/skills/stage-review/SKILL.md`
  - `.agents/skills/optimize/SKILL.md`
  - `.agents/skills/tt-device-usage/SKILL.md`
  - Supplied optimized-decoder contract
- Artifact paths:
  - README, work log, AutoFix reports, prior review
  - BFP4 precision and geometry raw runners
  - Explicit prefill candidate runner
  - Final correctness/performance runners
  - Final watcher JUnit XML and summary
  - Existing decode/prefill Tracy CSVs and reports
  - Context contract
- Code paths:
  - `tt/optimized_decoder.py`
  - optimized correctness/performance/precision/geometry/prefill tests
- Commands run:
  - Read-only `sed`, `grep`, `find`, `git status`, and `git diff`
  - No hardware, tests, servers, or implementation edits

## Residual Risk

- Precision policy propagation, real-weight PCC, B1/B32 timing, direct paged
  cache transition, non-aligned sequence lengths, context 131072, five-replay
  determinism, and watcher cleanliness are now supported by raw artifacts.
- Runtime source contains no Torch conversion or functional fallback.
- The only remaining gate is collecting and documenting profiler evidence for
  the exact final program-config state, followed by another independent
  rereview and isolated local commit.
