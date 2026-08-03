# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The traced-decode gain is small (about 2.4-2.8 microseconds), but 40
  independent processes, 40,000 paired samples, reversed candidate ordering,
  and strictly favorable confidence intervals support it.
- The remaining RoPE layout operations require a Phi-specific width-96,
  midpoint-48 fused kernel that is absent from the current TTNN operation set
  and outside this model-local stage scope.
- Unrelated untracked files elsewhere in the worktree do not overlap this
  stage or its commits.

## Hard-Check Gaps

- The fused suite directly tests non-aligned prefill lengths 31, 33, and 65.
  Full-context behavior is inherited from the functional decoder because the
  fused override changes only the sequence-generic MLP activation/multiply and
  does not alter context, cache, page tables, LongRoPE, or chunking.

## Anomaly Ledger

- Observed anomaly: the first current-branch review found stale checkpoint
  provenance in `work_log.md`.
  Evidence: the corrected log distinguishes source checkpoint `042cb2dade7`,
  current-branch implementation/evidence checkpoint `e7c887226e7`, metadata
  checkpoint `0e5ae20717c`, and provenance-fix commit
  `ab87d0fee6280402444593bb5d0f437d8bc19782`.
  Affected path: stage traceability.
  Control or comparison: git history, branch containment, and stage-path diff.
  Likely subsystem: cross-branch transplant bookkeeping.
  Investigation performed: the independent reviewer verified ancestry, commit
  objects, and identical stage paths.
  Resolution: fixed.
- Observed anomaly: individual traced-decode timing pairs contain outliers.
  Evidence: all 40 per-process means still favor fused.
  Affected path: batch-1 and batch-32 traced-decode selection.
  Control or comparison: ten processes per batch/order, 1,000 interleaved pairs
  per process, reversed order, and hierarchical bootstrap.
  Likely subsystem: host synchronization and timing jitter.
  Investigation performed: the reviewer parsed all 40 JSON files and logs.
  Resolution: controlled.
- Observed anomaly: successful Tracy output reports histogram overflow buckets
  and nanobind teardown leaks.
  Evidence: messages occur after successful tests, report generation, and
  device closure.
  Affected path: profiler evidence generation.
  Control or comparison: complete `ops.csv`, four report families, successful
  bounded profiling, and separate watcher-clean execution.
  Likely subsystem: profiler presentation and binding teardown.
  Investigation performed: the reviewer distinguished these messages from the
  rejected marker-buffer-overflow run.
  Resolution: controlled.

## Scope Inspected

- Goal/skills: Stage 02 contract, `graph-fusing`, `tt-device-usage`, and
  `stage-review`.
- Artifacts: README, work log, correctness, repeated A/B, Tracy, watcher, and
  context-contract evidence.
- Code: `tt/fused_decoder.py`, `tests/test_fused_decoder.py`,
  `tests/fused_decoder_perf.py`, and `tests/analyze_fused_decoder_ab.py`.
- Commands: read-only git and artifact inspection; no hardware or file
  modification by the reviewer.

## Residual Risk

- Low. The rewrite is isolated, fused-versus-functional PCC is 1.0, reference
  PCC exceeds 0.999997, batch-1/32 trace replay is deterministic, watcher
  evidence is clean, and the small performance gain is statistically robust.
