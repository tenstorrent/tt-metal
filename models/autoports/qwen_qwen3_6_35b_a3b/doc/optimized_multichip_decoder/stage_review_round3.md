# Stage Review

Verdict: **pass-with-carried**

Independent diff-only round 3 of 3 for stage 5, `optimized-multichip-decoder`,
`Qwen/Qwen3.6-35B-A3B`, dated 2026-11-14. The Required Work and Other Concerns
in [round 2](stage_review_round2.md) were read verbatim. This review checks
their closure under the subsequent explicit owner decision and inspects the
changed acceptance code and records. It does not re-review the entire stage.
No hardware, TTNN import, reset, server, tuning, long experiment or additional
reviewer was used. Only this report was written.

## Required Work

- **None. P1 is CLOSED by explicit owner acceptance of the recorded performance
  shortfall, not by a performance repair.** The source owner message exactly
  matches [owner_decision.md](owner_decision.md) and
  [limitations.json](../eval_readiness/limitations.json): accept the recorded
  linear speed for stage 5 and perform no further tuning for this floor.
  This supersedes only the performance acceptance requirement for this row.
  It grants no numerical, trace, context, other-batch, other-layer or
  future-stage exemption and makes no claim that HF reproduces the shortfall.
  `original_model: "not_applicable"` correctly records that distinction.

  Independent recomputation of all 384 samples gives linear
  **0.719504473863708 ms/token**, useful-byte roofline
  **0.0518283125 ms/token**, and **7.203334292%**. The original target remains
  10%, corresponding to **0.518283125 ms/token**. Both raw pass flags remain
  false. The restored full-attention control independently gives
  **0.3408368071783722 ms/token**, **10.105496025%**, and a measured pass.

  The changed checker restricts the exception to
  `restored_functional_l0.json`, layer 0, B1, true 1x4 mesh, the 128-in/128-out
  sequence and the exact accepted measurements/source/build identities. The
  inherited paired checks still enforce whole-output PCC, state coverage,
  fallback status, replicas, deterministic execution and trace behavior;
  `functional()` additionally validates the changing-input sequence, timing
  samples, byte model and original target. The stage-4 gate and full-attention
  row retain strict raw-pass requirements.

## Other Concerns

- **P2 remains CLOSED.** The prior targeted watcher remediation is unaffected.
  The `watcher_non_aligned_paths` row and all 31 other non-FUNCTIONAL gate rows
  are identical to the preserved round-2 summary. No decoder, device runner,
  watcher fixture or raw measurement changed in this follow-up. The round-2
  watcher finding therefore needs no new device run.
- The prior validation-code concern is closed: the base `Gate` change only
  extracts its existing strict result check into a method; the override is
  confined to the named stage-5 artifact. Six CPU regression tests pass.
  Additional independent in-memory controls reject changed owner text, a
  lowered target, wrong sequence positions, baseline trace failure, and
  native identities altered consistently in both the limitation record and
  provenance. The latter fail against the actual current files.
- Existing handoffs remain in the work log's **Carried to next stage**:
  the accepted linear performance as a planning limitation with no further
  stage-5 floor tuning; the replicated BF16 hidden-2048 residual and serial
  shared-workspace lifetime contract; deferred production formatting; and
  optional untried optimization work. The earlier B32 prefill tradeoff and
  unselected native precision/CCL candidates remain historical findings,
  without new measurement or promotion. Future stages retain their own gates.
- The round-2 README reproduction-list note is closed: `watcher_paths` appears
  in the current list. Status text distinguishes the owner acceptance from
  the original failed floor and failed AutoFix; historical failure reports
  remain available. No new correctness or accuracy defect was found in the
  changed files.
- A passing local stage checkpoint must follow this review. At inspection
  time the live branch is `main`, HEAD
  `bdba2e3af7ae463d4666429a56b4042c3c2ce4ca`; the previously recorded incomplete
  checkpoint is `8ef853637be7da3da63aa1aa00d945ded86437cb`. The run plan permits
  local stage-owned commits and prohibits autonomous pushes. No commit or
  push was performed by this reviewer; the stage owner records the accepted
  checkpoint and final status after review.

## Hard-Check Gaps

- The round-2 stale FUNCTIONAL selection concern is closed. Both kinds now
  use `restored_functional_l{0,3}.json`. The actual two recorded native source
  files and all three loaded libraries independently match their recorded
  identities for both rows. The production decoder SHA256 remains
  `13b403518fb8b71dcf5b502bddeb9be79bcd6acf27b1bb51ab47c4ef902d952f`.
- The production decoder, both restored result JSONs and both provenance
  sidecars are byte-identical to HEAD. The saved
  `gate_summary_before_owner_acceptance.json` is byte-identical to the HEAD
  33/34 summary. Only the two FUNCTIONAL rows changed in the new summary.
  Current **34/34 acceptance** means **33 measured/check passes plus one
  explicit owner-accepted performance shortfall**. The linear detail retains
  `raw_floor_passed=false`, and the top-level accepted-limitations list names
  exactly `stage5-linear-decode-performance`.
- Completed full-gate and pre-commit logs were inspected; the full gate was
  not rerun because its entrypoint writes evidence files. The focused CPU
  tests and independent artifact analyses were rerun read-only. No additional
  required evidence gap was identified in this diff-only scope.

## Anomaly Ledger

- Observed anomaly: Linear FUNCTIONAL remains below the original 10% floor.
  Evidence: `restored_functional_l0.json`, `owner_decision.md`,
  `../eval_readiness/limitations.json`, `gate_summary.json` and the preserved
  failed summary/AutoFix record.
  Affected path: Selected B1 linear decoder on the true 1x4 mesh.
  Control or comparison: Restored full-attention measured pass; earlier
  failed bounded repair controls, unchanged in this follow-up.
  Likely subsystem: The previously investigated ordered linear computation
  and data movement; no new root cause is asserted.
  Investigation performed: Recomputed samples and useful-byte bound; verified
  unchanged raw evidence and actual source/build identities; inspected and
  negatively tested the scoped acceptance gate.
  Resolution: **controlled** by explicit owner acceptance of this performance
  result only. The shortfall persists; no repair, speedup or HF waiver is claimed.
- Observed anomaly: Changed non-aligned/resumed prefill paths previously lacked
  watcher coverage.
  Evidence: Round-2 report and unchanged required `watcher_non_aligned_paths`
  gate/evidence.
  Affected path: Both layer kinds' non-aligned/resumed and singleton prefill.
  Control or comparison: The previously reviewed targeted watcher controls.
  Likely subsystem: Validation coverage.
  Investigation performed: Confirmed this follow-up does not change the
  implementation, fixtures or accepted watcher gate.
  Resolution: **fixed** in round 2; closure retained.

## Scope Inspected

- Goal and owner message:
  `/home/ttuser/qwen36-35b-a3b/multigoal_logs/05-05-optimized-multichip-decoder.prompt.txt`
  and `05-05-optimized-multichip-decoder.resume-2.message.txt`.
- Skills: Installed `tt-model-bringup/0.1.24` stage-review and model-bringup
  instructions; relevant optimize, multichip, tt-device-usage and
  `optimize/references/evidence-checklist.md` requirements. Review mode only.
- Changed code: `tests/check_optimized_multichip_evidence.py`,
  `tests/summarize_multichip_validation.py`,
  `tests/test_owner_performance_acceptance.py`. Read the unchanged runner's
  raw-pass calculation and FUNCTIONAL byte model to verify the exception.
- Records: Changed README, work log, AutoFix status, limitation and owner
  decision records; previous review; old/new summaries; restored FUNCTIONAL
  JSONs/sidecars; actual native source/library identities; completed
  `logs/owner_acceptance_{unit,precommit,gate_check}.log`.
- Commands: Read-only Git status/diff/show, bounded file reads and `rg`;
  `python -B -m unittest models.autoports.qwen_qwen3_6_35b_a3b.tests.test_owner_performance_acceptance -v`
  (**6/6 pass**); small `python -B` statistical, byte-comparison, identity and
  in-memory negative-control analyses. Confirmed no TTNN module was imported.

## Residual Risk

- This is independent inspection of completed device evidence, not a new
  device execution. Changed implementation or native builds require affected
  requalification; the owner decision does not extend to new measurements.
- The standalone device FUNCTIONAL runner intentionally still fails the raw
  linear performance assertion. The CPU stage-acceptance checker applies the
  explicit exception and preserves its disposition. Consumers must retain
  that distinction when reporting the 34/34 acceptance count.
- Existing full-model handoffs remain applicable. This review does not
  establish full-model generation or serving readiness.
