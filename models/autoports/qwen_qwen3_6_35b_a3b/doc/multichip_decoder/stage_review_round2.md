# Stage Review

Verdict: clean-pass

Round 2 of 3, independent diff-only follow-up for stage 4, multichip decoder,
Qwen/Qwen3.6-35B-A3B. The sole previous Required Work item is closed. No new
correctness or required-evidence defect was found in the changes. The live
worktree remains on `main` at `2901d8a506198dc69a9e99714db3a9c590b6ff95`.

## Required Work

- None.
- **Closed — P2: Supply the required trace-allocation positive control.**
  `trace_alloc_tracking/positive_control.log` records the actual RuntimeError
  identifying the deliberately unacknowledged buffer 30. The control runner
  creates that allocation after capture, verifies that the tracker lists it,
  and requires replay to raise. It then deallocates it, verifies its removal
  from the unsafe set, and checks exact recovered output on all four devices.
  Only capture-output buffer 23 is acknowledged; the input and output remain
  live until the single trace is released. There is no broad allocation scope.
  The startup provenance records tracking `1`, skip-program-cache `0`, watcher
  unset, and the same TTNN build `0.1.dev1+ga464944` as the accepted decoder
  runs. The log's provenance and PASS payload exactly equal their JSON files.
  The current control-runner SHA256 matches the recorded
  `7f855d14bf1a6ec0b01c0e1caa40d0d980f4929f08e51d8982d96aa54b42b94d`.
  The mandatory `trace_positive_control()` gate checks the source identity,
  environment, buffer-specific exception, log completion, recovery and all
  four replicas. I invoked that method read-only on CPU; it passes. The saved
  full acceptance result and console log agree on 27/27 passing gates.

## Other Concerns

- **Closed — stale constructor census.** `mesh_plan.md:139` now gives
  465,569,792 bytes for the linear layer and 450,185,216 for the full layer.
  These exactly match both `mixed_stack_selected.json` and
  `memory_capacity_plan.json`; the latter retains 4,135,717,888 bytes of
  planned per-device headroom.
- **Open as a nonblocking stage-5 performance hand-off.** Per-request B32
  attention, approximately 200 us of output splitting, B32 shared/grouped
  geometry and adapting tuned prefill programs to local mesh shapes remain
  unmeasured candidates. `work_log.md` now explicitly records all four under
  `Performance hand-off notes`. The changes do not claim that these candidates
  were measured or rejected, and round 1 did not require them for stage-4
  correctness acceptance.

## Hard-Check Gaps

- **Closed — missing positive-control gate.** Its three evidence paths are
  present and are included in the gate's evidence manifest. All 135 paths in
  the saved full gate summary exist.
- **Unchanged, already controlled — historical paired-runner identity.**
  Round 1 established that the recorded older runner differs only by the
  disabled optional prefill-state export. The remediation does not change
  that runner or the measured decoder path. The new control has its own
  mandatory current-runner hash check. No previous measurement is invalidated.
- **Unchanged, advisory — shared profiler summarization helpers.** The
  independent raw-CSV reconstruction from round 1 remains applicable;
  profiler evidence was not changed or re-reviewed in this round.
- Optional future checker hardening: the existing generic `off()` helper
  accepts watcher value `"0"`, although runtime parsing enables watcher for
  that value. This does not invalidate the current result: the new control
  gate specifically requires watcher to be unset, and the inspected accepted
  clean-timing and B1/B32 profiler sidecars all record watcher as `null`.
  Preserve the stricter unset check when reusing the checker in later stages.

## Anomaly Ledger

- Observed anomaly: the required tracker rejection control was absent in
  round 1 despite clean tracked decoder executions.
  Evidence: previous report; new control runner, JSON, provenance and log.
  Affected path: allocation-safety acceptance for mesh traces.
  Control or comparison: deliberate live buffer 30 rejects replay; release
  permits replay with exact output on four devices.
  Likely subsystem: validation coverage.
  Investigation performed: inspected control allocation and lifetime rules,
  compared log payloads to JSON, verified source/environment identity and ran
  only the new CPU evidence-check method.
  Resolution: fixed; previous P2 closed.

- Observed anomaly: the first control attempt stalled during mesh open with
  `TT_METAL_WATCHER=0` and was recorded as timing out after 180 seconds.
  Evidence: `failures/trace_control_watcher_zero/` provenance, console log,
  live Python stack and triage capture; `logs/reset_12{,_smoke}.log`.
  Affected path: control-launch initialization, before decoder execution.
  Control or comparison: after reset, the four-device trivial-op smoke and
  corrected control with watcher unset both complete and close the mesh.
  Likely subsystem: runtime initialization with zero-interval watcher enabled.
  Investigation performed: verified the failed environment and live
  `open_mesh_device` stack; inspected `tt_metal/llrt/rtoptions.cpp:1467`, which
  enables watcher whenever the variable is present; inspected recovery and
  successful-control logs. The new runner rejects any present watcher value
  before importing TTNN.
  Resolution: controlled by recovery and corrected launcher. Reset and the
  environment correction occurred together, so the evidence does not isolate
  zero-interval polling as the sole cause of the original stall. This does
  not affect the successful control or earlier decoder measurements.

## Scope Inspected

- Goal/skill paths: `review_round2_brief.md`, `review_contract.md`,
  `stage_review_round1.md`; installed v0.1.24 `stage-review`, relevant
  `tt-enable-tracing` and `tt-device-usage` requirements, model-bringup startup
  and multichip trace/evidence requirements.
- Artifact paths: the new positive-control log, JSON and provenance; updated
  gate summary and final gate log; changed README, work-log, manifest,
  AutoFix and mesh-plan sections; initial control-failure captures and reset12
  recovery logs. Existing capacity and selected-run provenance were read only
  to check the corrected census, unchanged source and matching environments.
- Code paths: `tests/run_multichip_trace_control.py`, the new gate and its
  supporting provenance checks in `tests/summarize_multichip_validation.py`,
  and the unchanged runtime watcher option parser.
- Commands run: read-only `cat`, `sed`, `rg`, `git status`, `git rev-parse`,
  Python JSON/path/source-hash comparisons, and a `runpy` load followed by
  `Gate.trace_positive_control()` only. No full gate rewrite, TTNN import,
  device open, reset, server, hardware test or long test was performed.
  Only this review report was written.

## Residual Risk

- The decoder SHA256 remains
  `97792b176b8fdc2f41d0b1c9c6ab7e98f08dd7bb94d87fac08b008c3e27fcaa3`,
  matching the measured implementation and new control provenance. This
  follow-up relies on round 1 for unchanged numerical, timing and profiler
  validation; it is not a whole-stage re-review.
- The small negation control proves that the allocation checker rejects an
  unsafe live allocation in this environment. The existing tracked decoder
  stress, natural-batch and mixed-stack runs remain the evidence for actual
  model trace behavior.
- Full-model resident memory, generation quality, serving and concurrent
  workspace use remain later-stage obligations as recorded in round 1.
- Per the supplied run plan, the stage owner must now record this passing
  review and create the local stage-owned checkpoint commit. No push is
  authorized, and push history was not inferred. This reviewer made no commit.
