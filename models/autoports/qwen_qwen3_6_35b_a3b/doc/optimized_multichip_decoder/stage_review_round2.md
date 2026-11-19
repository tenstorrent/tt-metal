# Stage Review

Verdict: **more-work-needed**

Independent diff-only round 2 of 3 for stage 5, `optimized-multichip-decoder`,
`Qwen/Qwen3.6-35B-A3B`, dated 2026-11-14. The two Required Work findings and
Other Concerns in [round 1](stage_review_round1.md) were read verbatim. This
review checks their disposition and the targeted watcher changes; it does not
re-review the entire stage. No hardware, reset, server, implementation edit,
long test, or additional reviewer was used. Only this report was written.

## Required Work

- **P1 — OPEN: The retained linear-attention default fails the mandatory FUNCTIONAL floor.**
  Evidence: Independently recomputed all 384 samples in
  `restored_functional_l0.json`: mean **0.719504473863708 ms/token**. The
  unchanged `tests/multichip_functional.py` byte model recomputes
  **0.0518283125 ms/token**, yielding **7.203334292%** against the required
  10%. The ceiling remains **0.518283125 ms/token**, requiring **27.97%** less
  latency. No production or performance repair was made after round 1.
  `gate_summary.json` and `logs/final_gate_check.log` now report **33/34**,
  failing only `functional_l0`; the previous **32/33** result is preserved
  in `gate_summary_before_targeted_watcher.json`. The restored full-attention
  control independently recomputes 0.3408368071783722 ms/token and
  10.105496%, which does not satisfy the linear row.
  Why this matters: This is an unchanged mandatory acceptance gate. The
  optimize Stop Rule and successful watcher remediation do not waive it.
  Required next step: Restore the linear floor while preserving the accepted
  numerical and trace contracts, then qualify the affected final default
  and obtain a passing independent review. The already documented bounded
  AutoFix failed to restore the contract; this round adds no successful
  repair. If work ends on that outcome, retain **incomplete** stage status.
  The failed experiments do not establish that all possible kernel designs
  are exhausted.

## Other Concerns

- **P2 — CLOSED: Watcher coverage omits changed non-aligned/resumed prefill paths.**
  New evidence is `watcher_mixed_non_aligned.json`, `watcher_l0_s1.json`,
  `watcher_l3_s1.json`, their sidecars, runner logs, saved tensors and watcher
  logs. The mixed fixture runs layers `[0, 3]`, prefixes `[33, 35]`, resumed
  lengths `[32, 34]`, and four decode steps at positions `[65, 69]` through
  `[68, 72]`. Source inspection confirms continuation plans use the prefix
  length as `start_pos`; this is real resumed execution.

  The mixed fixture also executes independent one-shot controls of lengths
  65 and 69. Re-evaluating the existing host plan method without TTNN gives
  linear chunks `[64, 1]` and `[64, 5]`, in addition to the 33/35/32/34-row
  prefix/resume groups. Full attention covers aligned prefix interiors,
  singleton tails, and singleton lead rows at the nonzero starts. The
  separate S1 fixtures cover the whole-input singleton branch in both kinds.
  This closes the selected changed-path subset requested by round 1.

  All three sidecars record `TT_METAL_WATCHER=10`, allocation tracking `1`,
  program-cache exclusion `0`, serial device execution and device profiler
  unset. The mixed command uses the runner's defaults; its current recorded
  runner hash matches, and those defaults are explicitly 33/35, 32/34, D4.
  The reproduction group now spells out these values. No nondefault mesh
  policy is selected. The new raw watcher logs contain **12/2/1 completed
  dumps**, respectively, include devices 0–3, and pass the existing failure
  signature scan. Each log path resolves to its sidecar's `TT_METAL_LOGS_PATH`.
  Runner logs show sequential jobs with clean device shutdowns.

  Existing numerical, saved-state, cache ownership, runtime-fallback and
  trace checks pass. Mixed minimum output PCC is **0.99968888227761**;
  minimum saved-state PCC is **0.9999777446359203**. The mesh one-shot/resumed
  minimum is **0.9999334580612755**. It retains one capture and 24 replays per
  implementation. Singleton prefill PCC is **0.9999948538817867** /
  **0.9999912156446874**, with minimum decode PCC **0.9999925539994733** /
  **0.99982006223667**, exact saved-state PCC and passing live-trace prefill
  controls. These watcher runs establish correctness evidence, not new
  performance numbers.

- No new blocking problem was found in the three changed validation files.
  `mixed()` retains its prior default fixture and watcher-disabled default;
  extracting `watcher_log()` preserves the existing dump/error checks.
  `watcher_paths()` adds enforced geometry, mesh, instrumentation and log
  provenance checks. Independent CPU calls pass the new targeted gate,
  the unchanged default mixed gate and an existing watcher gate. In-memory
  negative controls reject wrong lengths, disabled watcher, disabled tracking,
  program-cache exclusion, mixed profiling and an incorrect watcher log root.
  `bash -n` passes; `logs/watcher_paths_precommit.log` records passing relevant
  hooks for the changed validation files.

- The earlier optional optimization handoffs, disclosed B32 prefill tradeoff,
  rejected native precision family and deferred production formatting remain
  as recorded in round 1. They were not newly measured or promoted here.
  README/work-log/AutoFix status updates accurately retain the linear failure.
  The README's final reproduction-group list omits `watcher_paths`, although
  the work log and runner document it; this is a minor documentation note,
  not another required hardware run or a separate blocker.

- No accepted local stage checkpoint exists at review time. The run plan
  authorizes local stage-owned commits and prohibits autonomous pushes;
  push policy is not inferred from Git history. A later incomplete snapshot
  commit would not turn this verdict into stage acceptance.

## Hard-Check Gaps

- The round-1 changed-path enforcement gap is closed:
  `watcher_non_aligned_paths` is now a required strict-check row, with the
  same details independently reproduced in `watcher_paths_summary.json`.
- The existing FUNCTIONAL checker still references `functional_l*.json`
  rather than `restored_functional_l*.json`. The independently recomputed
  restored result preserves the same fail/pass classification. This is not
  a new blocker and does not weaken P1. After any future native/runtime
  repair, qualification must use the final loaded build, as round 1 states.
- The completed full strict check was inspected rather than rerun. Only the
  affected small CPU gates and artifact analyses were rerun for this review.

## Anomaly Ledger

- Observed anomaly: Linear FUNCTIONAL remains below 10%.
  Evidence: `restored_functional_l0.json`, `gate_summary.json`,
  `logs/final_gate_check.log`, `AUTOFIX_functional.md`.
  Affected path: Selected batch-one linear decoder on the true 1x4 mesh.
  Control or comparison: Passing restored full-attention row; prior failed
  complete-decoder repair controls, unchanged since round 1.
  Likely subsystem: The ordered projection/norm/recurrent computation and
  data movement identified previously; this review assigns no new root cause.
  Investigation performed: Recomputed sample means and byte-model fraction;
  confirmed unchanged selected source and no new performance repair.
  Resolution: **more-work-needed**, P1 remains open.

- Observed anomaly: Required changed prefill paths lacked watcher evidence.
  Evidence: Three new watcher artifacts, provenance, completed dumps,
  `watcher_paths_summary.json` and the enforced strict-check row.
  Affected path: Non-aligned/resumed and singleton prefill in both layer kinds.
  Control or comparison: Single-chip controls, one-shot/resumed comparisons,
  shared/independent cache checks, changing-input replay and singleton controls.
  Likely subsystem: Validation coverage, with no detected memory fault.
  Investigation performed: Checked host branch plans, recorded source/build
  identity, raw watcher logs and the new gate including negative controls.
  Resolution: **fixed**, P2 closed.

## Scope Inspected

- Goal: `/home/ttuser/qwen36-35b-a3b/multigoal_logs/05-05-optimized-multichip-decoder.prompt.txt`.
- Skills: Installed `tt-model-bringup/0.1.24/skills/{stage-review,model-bringup,optimize,multichip,tt-device-usage}/SKILL.md`
  and `optimize/references/evidence-checklist.md`.
- Workspace: Live `main`, base `5ebe699604445e7b30ea0813e15aa913efbd3188`,
  staged/unstaged stage changes, no new stage commit at review time.
  Production decoder SHA256 remains
  `13b403518fb8b71dcf5b502bddeb9be79bcd6acf27b1bb51ab47c4ef902d952f`.
- Changed code: `tests/run_optimized_multichip_validation.sh`,
  `tests/summarize_multichip_validation.py`,
  `tests/check_optimized_multichip_evidence.py`. Read only the relevant
  unchanged mixed runner, inherited prefill planner, mesh prefill path,
  source provenance and FUNCTIONAL byte model to verify the claims.
- Artifacts: Round-1 report; new watcher JSON/sidecars/logs/summary and
  referenced tensor availability; old/new gate summaries and final check
  log; changed README/work-log/AutoFix status text; restored FUNCTIONAL
  samples and native provenance; targeted pre-commit log.
- Commands: Read-only `git diff/status`, `rg`, bounded file reads,
  `bash -n`; small `python -B` JSON/hash/statistical analyses, isolated host
  plan evaluation, direct CPU gate methods and in-memory negative controls.
  CPU gate calls did not import TTNN. Selected decoder/helper/runner hashes
  match the new evidence; native CCL sources match current files. All three
  new sidecars carry the same loaded-native identities as restored FUNCTIONAL.

## Residual Risk

- This is inspection of completed evidence, not independent device execution.
  The new watcher subset closes P2 for the recorded selected implementation;
  subsequent implementation/build changes require affected requalification.
- New dump counts reflect the selected short fixtures, not a claim that every
  possible shape was exercised. Existing unaffected stage evidence remains
  subject to the round-1 scope and limitations.
- The stage remains incomplete because P1 is a failed mandatory gate.
  A failed bounded AutoFix is an evidenced stopping outcome, not a pass,
  performance waiver, or proof that no further kernel approach can succeed.
