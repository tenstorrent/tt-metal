# Stage Review

Verdict: clean-pass

Independent follow-up review, round 2 of 3, for stage 1 functional-decoder,
`Qwen/Qwen3.6-27B`. This is a diff-only review of the round-1 measurement
finding and its remediation, not a second review of the entire stage.
The live worktree is on `main` at base
`a0b0bd54e198d81676662b5a2d3a489e29bc9ab9`; the target autoport is untracked
pending its authorized local checkpoint.

## Required Work

- None. The previous P2, "Collect the final performance evidence with trace
  allocation tracking disabled," is **closed**.
  - Both `layer_0_profile_untracked.json` and
    `layer_3_profile_untracked.json`, their compressed console logs, and the
    accepted `perf/*/metrics.json` agree on tracking `0`, tracebacks `0`,
    program-cache skip `0`, Watcher unset, and profiler capacity `5000`.
    `TT_METAL_DEVICE_PROFILER=1` is the expected Tracy profiler setting.
  - Console result dictionaries equal the JSON results. Both real-weight
    batch-1, length-512 runs pass prefill, traced-decode and updated-input PCC
    above 0.995, repeated bit equality, and changed-input output checks.
    Their null allocation positive controls are expected because tracking is
    disabled. Original `layer_{0,3}_profile_final.json` results retain
    tracking `1` and passing allocation positive controls.
  - New compressed console hashes match `profiler_audit.json` and the local
    original logs. Both logs complete ops-CSV generation and contain zero
    dropped-marker warnings. New and preserved tracked compressed raw ops
    match their respective recorded SHA256 values.
  - The decoder hash remains
    `476f9d99fa90a1268f264964037dbc6837b320adc9f48d172dc1156b8b35962d`;
    the execution harness hash remains
    `750a5874e781ad51659b0b16fc9fb2ca4dad07c86ee5e6095eef7378edfb5008`.
    Both match the current files and the original tracked profiles.
    HF-reference and config hashes also agree between the two regimes.
  - Independently summed raw signposted kernel durations and inter-op gaps
    match all four filtered CSV totals and accepted metric values. Each
    measured raw operation has a finite positive duration, device ID 0 and
    `PROGRAM CACHE HIT=True`. Decode windows contain trace ID 0, replay
    session 3. Counts and op/shape/dtype/fidelity signatures match the
    preserved tracked windows. Drains remain outside measured signposts in
    the unchanged harness. Human table row counts match the filtered CSVs.
  - Updated README, summary document and work-log timings agree with the
    replacement evidence. No decoder or full-context rerun was necessary.

Recomputed timings, in microseconds:

| Layer kind | Window | Device ops | Kernel sum | Inter-op gaps | Kernel plus gaps |
|---|---|---:|---:|---:|---:|
| linear_attention | prefill | 1021 | 16050.762 | 2528.762 | 18579.524 |
| linear_attention | traced decode | 103 | 2734.896 | 62.803 | 2797.699 |
| full_attention | prefill | 57 | 6316.927 | 162.495 | 6479.422 |
| full_attention | traced decode | 63 | 2326.155 | 45.034 | 2371.189 |

All four filtered overlap sums are zero. These are warmed per-layer timings
for batch 1, prefill 512 and decode context 513.

## Other Concerns

- Previous selected-row long-context concern: **open as an unchanged,
  non-blocking handoff risk**. The updated README and work log still disclose
  selected boundary/final-row HF comparison, all-token TT execution, and all
  HF state updates. No full-model or generated-text accuracy claim was added.
- Previous full-attention reduction-geometry concern: **open as an unchanged,
  non-blocking handoff risk**. The minimum selected long-context contribution
  PCC of 0.9977730592 and the instruction to preserve that gate during later
  optimization remain documented. The decoder did not change.
- Previous serial mixed-request/unaligned-continuation latency concern:
  **open as an unchanged, non-blocking handoff risk**. The docs retain the
  functional-baseline description and do not extend the aligned length-512
  timing claim to these paths.
- No new correctness, measurement or unsupported-capability concern was
  identified in the changed artifacts.

## Hard-Check Gaps

- The previous environment-provenance gap is **closed for new evidence**:
  `tests/evidence.py` now records both
  `TT_METAL_TRACE_ALLOC_TRACEBACKS` and
  `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE`, and both replacement runs
  explicitly record zero. This host-only helper is called outside measured
  execution. Original artifacts remain unchanged rather than retroactively
  inventing missing fields.
- The prior note about literal trace-reuse summary counters is unchanged and
  was not re-reviewed; round 1 already accepted its inspected harness
  evidence. This remediation does not alter trace lifetime or reuse logic.
- A checkpoint packaging issue noticed during this review was resolved by
  the stage owner: checkout-wide `*.csv` ignores initially excluded the
  filtered reports. Scoped exceptions now make all eight accepted and
  historical `*_perf_report.csv` files visible to normal Git staging.
  Raw Tracy sessions remain ignored; compressed raw ops, tables, metrics and
  logs remain available for the checkpoint.

## Anomaly Ledger

- Observed anomaly: Accepted performance previously used allocation tracking.
  Evidence: Original profiles and `perf_tracked/*/metrics.json`, compared
  with the new untracked profiles and accepted `perf/*/metrics.json`.
  Affected path: Warmed prefill and traced-decode timing.
  Control or comparison: Same decoder, harness, config, real layers, shapes
  and measured op signatures; diagnostic tracking disabled in the new runs.
  Likely subsystem: Measurement instrumentation.
  Investigation performed: Verified environments, source hashes, raw-log and
  raw-ops provenance, signposts, cache hits, trace replay IDs, and arithmetic.
  Resolution: fixed; the round-1 P2 is closed.

- Observed anomaly: New profiling results contain a null allocation positive
  control.
  Evidence: Both `layer_*_profile_untracked.json` results and
  `tests/run_decoder.py`'s tracking-conditional control.
  Affected path: Diagnostic test execution, outside the measured windows.
  Control or comparison: Original tracked profile results retain passing
  controls and matching execution-source hashes.
  Likely subsystem: Intentional tracking-disabled measurement mode.
  Investigation performed: Matched console dictionaries to JSON and checked
  the control condition against the recorded environments.
  Resolution: controlled; not a missing tracked correctness gate.

- Observed anomaly: Both new logs contain pandas `DtypeWarning` messages
  during Tracy postprocessing, alongside already classified motherboard,
  subset-MMIO and optional WASM-copy warnings; the rendered reports retain
  an unclassified-reshape category warning.
  Evidence: Both compressed untracked console logs and human report files.
  Affected path: Host postprocessing, discovery labels and optional report
  visualization/category classification.
  Control or comparison: Both runs complete CSV generation; every signposted
  device row has timing data; raw and filtered sums agree; measured operation
  counts and signatures match the preserved tracked runs.
  Likely subsystem: Host reporting tools, not decoder arithmetic.
  Investigation performed: Read warning context and successful report tails,
  checked all selected rows, hashes and totals, and verified no dropped-marker
  warning. No missing or non-finite measurement was found.
  Resolution: controlled.

- Observed anomaly: The repository-wide CSV ignore rule initially hid the
  filtered reports from ordinary checkpoint staging.
  Evidence: `git check-ignore -v` identified root `.gitignore:8`; after the
  owner's scoped exceptions, `git ls-files --others --exclude-standard`
  lists all eight filtered CSVs under `perf/` and `perf_tracked/`.
  Affected path: Evidence packaging.
  Control or comparison: Files remained on disk throughout; numerical
  artifacts were unchanged.
  Likely subsystem: Git ignore rules.
  Investigation performed: Checked ignore provenance and post-fix staging
  eligibility.
  Resolution: fixed.

## Scope Inspected

- Goal/skill paths: Parent-provided original stage contract and diff-only
  brief; `stage_review_1.md`; installed
  `/home/ttuser/qwen36-27b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/`
  `stage-review/SKILL.md`, model-bringup startup, and relevant functional-decoder,
  tt-enable-tracing and tt-device-usage requirements. Startup plugin inventory
  confirms enabled tt-model-bringup 0.1.23 and tt-autodebug 0.1.8.
- Artifact paths: `layer_{0,3}_profile_untracked.json/.log.gz`, corresponding
  local raw logs and Tracy ops; complete accepted `perf/` reports and
  `perf_tracked/` controls; original tracked profile JSON/log archives;
  `profiler_audit.json`, `profiler_audit_tracked.json`, `acceptance.json`,
  `README.md`, `work_log.md`, `../functional_decoder.md`,
  `hooks_provenance.log` and `hooks_review2.log`. Both hook logs report passed
  applicable checks.
- Code paths: Changed `tests/evidence.py` and target `.gitignore`; unchanged
  `tests/render_perf.py` and relevant signpost, drain, replay and control
  sections of `tests/run_decoder.py`. Decoder content was hashed to establish
  unchanged source, not re-reviewed.
- Commands run: Read-only `cat`, `sed`, `rg`, `git status`, `git rev-parse`,
  `git check-ignore`, `git ls-files`, and Python standard-library analysis of
  JSON, gzip logs, CSVs and hashes. No TTNN import, device open, hardware test,
  reset, process termination, server launch or implementation edit occurred.
  Only this report was written by the reviewer.
- Push/checkpoint state: The run plan authorizes a local checkpoint after a
  passing review and prohibits pushing. The owner still needs to record the
  passing review and checkpoint; this review performed neither commit nor push.

## Residual Risk

- This accepts completed artifacts by inspection, not independent hardware
  reproduction. The broader correctness assessment remains round 1's scope.
- Removing allocation diagnostics does not remove the device profiler needed
  to measure kernels. The accepted values are signposted profiler kernel/gap
  sums, not full-model serving latency or a statistical performance study.
- The unchanged handoff risks above remain relevant to later optimization
  and full-model work. No later-stage gate is asserted by this pass.
