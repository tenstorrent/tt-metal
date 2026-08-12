# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The retained adapter cleanup is performance-neutral, not an optimization
  win.  Documentation reports 62.36 final versus 62.94 baseline TPOT-derived
  tokens/s/user without claiming a speedup.
- A one-request primary run is weak evidence for sub-percent differences.  It
  does support the required comparable-work result: 14.573 ms median vLLM ITL
  versus 14.732 ms caller-visible full-model trace time at serving shapes.
- Sampling ends with nanobind shutdown leak diagnostics after 72 passed and one
  skipped; no functional failure or serving corruption accompanies them.

## Hard-Check Gaps

- Page-table changed/unchanged coverage is primarily an adapter contract test.
  It is supplemented by selected-policy device evidence, the real async-overlap
  run, final sampling/qualitative gates, and reset/steady source contracts.
- Profiler/device-time evidence is intentionally absent for optimized vLLM;
  `perf_summary.json` correctly records null device time and roofline.
- The primary run should not be interpreted as resolving a sub-percent speed
  difference.

## Anomaly Ledger

- Observed anomaly: serving initially appeared much slower than the 110.38
  tokens/s full-model result.
  Evidence: `AUTODEBUG.md`, `AUTOFIX.md`, and
  `results/autofix/full_model_batch32_active1.json`.
  Affected path: performance comparison.
  Control or comparison: physical batch-32/active-1 model-plus-sampling traces
  measure 14.732 ms; final serving median ITL is 14.573 ms.
  Likely subsystem: incomparable physical batch shapes.
  Investigation performed: async timing, synchronous-read and payload-reuse
  controls, then a serving-shape full-model harness.
  Resolution: controlled.
- Observed anomaly: final mean TPOT is 0.9% slower than baseline.
  Evidence: 15.888 versus 16.037 ms; CI 16.860 versus 16.876 ms.
  Affected path: primary and CI benchmarks.
  Control or comparison: median ITL 14.566 versus 14.573 ms; p99 improved from
  15.121 to 14.832 ms.
  Likely subsystem: normal run variance and request-boundary weighting.
  Investigation performed: exact command/config and raw JSON comparison.
  Resolution: controlled and reported as flat.
- Observed anomaly: base-model completions sometimes continue into adjacent
  Q&A material.
  Evidence: final qualitative JSON.
  Affected path: qualitative serving.
  Control or comparison: exact-revision HF controls show the same pattern.
  Likely subsystem: base checkpoint continuation behavior.
  Investigation performed: direct review of all prompts and HF controls.
  Resolution: controlled.
- Observed anomaly: nanobind teardown leak diagnostics.
  Evidence: final sampling log.
  Affected path: Python binding shutdown.
  Control or comparison: 72 passed, one unsupported beam test skipped, and all
  subsequent serving gates completed.
  Likely subsystem: binding teardown.
  Investigation performed: pytest summary and surrounding log review.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: stage-review, vllm-integration, optimize, tt-device-usage,
  tt-enable-tracing, and qualitative-check skill contracts.
- Artifact paths: optimized README/work log/perf/AutoFix evidence, before/after
  and candidate results, context contract, prompt/HF controls, async overlap,
  and selected/full-model trace evidence.
- Code paths: adapter, generator/model trace and sampling code, contract tests,
  comparable timing harness, repository status and diffs.
- Commands: read-only file, search, JSON, diff, and Git inspection.  The
  reviewer did not start a server, access TT devices, profile, or modify files.

## Residual Risk

- Sub-percent differences remain below the resolution of the primary run.
- Steady page-table reuse relies on the TT async scheduler guarantee; live slot
  remapping is rejected and resets refresh state.
- The external vLLM tree is clean; pre-existing untracked `third_party/tt-metal`
  remains excluded from the stage checkpoint.
