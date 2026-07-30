# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- None.

## Hard-Check Gaps

- None within the optimized-decoder stage. The main agent must now create the
  isolated local checkpoint and record branch/SHA in the work log; never push
  and do not include unrelated dirty files.

## Anomaly Ledger

- Observed anomaly: Precision-frontier candidates inherited mutable defaults.
  Evidence: The final reproducer makes all group dtypes explicit, and
  `bfp4_precision_frontier_runner.txt` contains distinct BFP8, attention-BFP4,
  down-BFP4, and combined-BFP4 B1/B32 rows.
  Affected path: Precision selection.
  Control or comparison: Combined BFP4 wins at both batches and passes
  real-weight whole-layer PCC.
  Likely subsystem: Experiment policy propagation.
  Investigation performed: AutoFix, raw rerun, source inspection.
  Resolution: fixed.

- Observed anomaly: Dominant geometry was initially carried from BFP8 to BFP4.
  Evidence: `bfp4_lofi_decode_geometry_runner.txt` crosses shipped BFP4/LoFi
  over 16/32 cores and legal K blocks at B1/B32. Block-16 down wins and is
  promoted.
  Affected path: Decode matmul geometry.
  Control or comparison: Final full-layer correctness, watcher, timing, and
  profiler rows all exercise block 16.
  Likely subsystem: Precision-locked program tuning.
  Investigation performed: Full raw matrix plus final promotion reruns.
  Resolution: fixed.

- Observed anomaly: Material prefill rows used automatic block-2 configs
  without explicit alternatives.
  Evidence: `prefill_explicit_config_runner.txt` measures block 4/8 at B1/B32,
  records exact QKV/gate-up B32 L1 blockers, and identifies explicit down as
  the integrable winner.
  Affected path: Prefill program configs.
  Control or comparison: Final explicit down passes non-aligned and complete
  correctness and improves warmed prefill at both batches.
  Likely subsystem: Program configuration/L1 capacity.
  Investigation performed: Candidate matrix, final implementation, and
  correctness/performance reruns.
  Resolution: fixed.

- Observed anomaly: Earlier final evidence lacked a raw clean BFP4 watcher run.
  Evidence: `watcher_bfp4_final.xml` is pytest-generated and records ten tests,
  zero failures/errors for the final block-16/explicit-prefill default.
  `correctness_prefill_down_explicit_runner.txt` preserves the raw ten-pass PCC
  run.
  Affected path: Correctness/watcher gate.
  Control or comparison: Cases include non-aligned prefill, paged transition,
  B1/B32 eager and traced decode, five replays, and context 131072.
  Likely subsystem: Evidence preservation.
  Investigation performed: Raw/JUnit artifact inspection.
  Resolution: fixed.

- Observed anomaly: Profiler evidence initially predated the final program
  promotions.
  Evidence: Final stage-owned decode CSVs show block-16 down at
  47.879/47.992 us, 12 active cores, BFP4/LoFi, and zero host ops. Final
  prefill CSVs show explicit block-8 down at 136.433 us on 32 cores (B1) and
  768.119 us on 64 cores (B32), with zero host ops.
  Affected path: Final runtime and tt-perf-report gate.
  Control or comparison: CSV configs match current source; updated summaries,
  README, and work log agree.
  Likely subsystem: Evidence ordering after late config promotion.
  Investigation performed: Final source/CSV/report/doc comparison.
  Resolution: fixed.

## Scope Inspected

- Goal/skill paths:
  - `.agents/skills/stage-review/SKILL.md`
  - `.agents/skills/optimize/SKILL.md`
  - `.agents/skills/tt-device-usage/SKILL.md`
  - Supplied optimized-decoder contract
- Artifact paths:
  - README, work log, AutoFix reports, prior reviews
  - Precision, geometry, explicit-prefill, correctness, performance, watcher,
    and final B1/B32 decode/prefill profiler artifacts
  - `doc/context_contract.json`
- Code paths:
  - `tt/optimized_decoder.py`
  - optimized correctness/performance/experiment tests
- Commands run:
  - Read-only `sed`, `grep`, `find`, `git status`, `git diff`, and
    `git diff --check`
  - No hardware, servers, or implementation edits

## Residual Risk

- No stage-blocking risk remains. The selected default preserves real-weight
  PCC above 0.995, paged-cache transition semantics, deterministic traced
  decode, non-aligned logical sequence lengths, batch 1/32 behavior, and the
  131072 context contract.
- Runtime source contains no Torch conversion, functional fallback, or host
  fallback. Final profiler regions report zero host ops.
- Multi-chip, full-model, generation, and vLLM behavior remain outside this
  stage by explicit scope.
