# Stage Review

Verdict: more-work-needed

Interim independent review of stage 12 on live `v5-extended-run`, HEAD `8a68d14dbf6788609fe0356d2a8c67e2b37b08d1`, 2026-11-15. This is not a rejection of the active P19 panel. No new blocking defect was established in the completed short-accuracy round or inspected implementation changes. The verdict records existing unresolved stage work; it does not authorize work beyond the deadline or terminal `blocked` status.

Paths below are relative to `doc/agentic_evals/` unless specified.

## Required Work

- P1: Finish or explicitly time-bound the unresolved SWE anomaly investigation before stage closure (existing work, not a new finding).
  Evidence: `current/swe-extended85-head-bfp8/collected-final/final-score-with-adjudication.json` retains 23 successes, 26 wrong, 22 fetched-solution failures, 11 primary anomalies and three unknowns. Even its upper score bound, 26/85, is below the reference band. `setup/swe85-conditional-hf/full-other-diagnostic-20/` and the coverage inventory retain additional full-TT anomalous observations without completed controls. Completed HF9 outputs do not resolve all of these distinct observations. P19's short accuracy pass is not a successful scored-list precision fix.
  Why this matters: shortened controls establish specific motifs under specific conditioning, not a blanket model-behavior waiver or an extended-mode pass.
  Required next step: continue the already admitted, bounded investigation; preserve observation-level dispositions and remaining uncovered cases. If a candidate is selected, complete its applicable acceptance/memory checks, measure only the stage-11 rows the rerun uses, and rerun the whole fixed list. Otherwise retain the result as **not passed yet** at the phase/stage deadline, with unknown grades null. Do not extend work past `1794791725` (2026-11-16 01:15:25 UTC) to satisfy this review.

## Other Concerns

- Current machine-readable summaries still contain superseded live states. In `results.json`, `precision_accuracy_followups.p18_active` says no completion/terminal review beside a completed rejected P18 record; the pipeline narrative retains P16/P17/P18/P20 “active” sentences. P19's description says “from P18” while restoring P18's gate/up group; the actual policy and REPORT correctly describe a down-only reduction from P17. Before final handoff, consolidate current status and mark historical claims explicitly. These are bookkeeping concerns, not grounds for rerunning completed measurements.
- REPORT links the historical `setup/swe85-other-hf-coverage-c.json`, which still says HF9 has zero completed outputs and group 1 is running. Preserve it as the historical audit, but link a current disposition index alongside it. Likewise the short-branch review says later-loop capture unresolved; the report describes subsequent completed evidence. Avoid treating old `next` fields as current status.
- Keep unrelated root `AUTODEBUG.md`, `AUTOFIX.md`, `mock-result.json`, and the already staged benchmark REPORT/stage_usage changes out of the stage-owned commit. Final independent review and local commit remain explicitly pending; no push is authorized.

## Hard-Check Gaps

- P19 fixed108 is active and unselected. Its lack of terminal artifacts is not a finding. Its completed outputs, cap dispositions, matching full-policy controls and eventual selection decision require the planned terminal review.
- The changed pool-sizing and batch-specific rate helpers are preparation, not live acceptance. Their first candidate use must validate observed precision, actual per-device pre-cache memory, pool geometry, smallest suitable validated batch and request accounting. No additional benchmark grid or profiler is requested.
- Recomputed all 14 compact accuracy rows against saved `readiness_accuracy.json`: prefill/decode/natural rows agree; all have 297 AIME and 511 natural model replays and zero new captures. Saved runtime policies equal the table policies and enumerate 60 layers. Full BF16, P17 and P19 pass the stated 90/98/100 per-reference bars; the other rows fail natural top-1. This supports **short-reference diagnostic** ranking, not long-context/free-running/served acceptance.
- Telemetry is scoped correctly: exact eight measured 4096-input/128-output B1/C1 requests in `current/headline-head-bfp8-b1/perf-b1.json`, TTFT 755.6318 ms, mean TPOT 28.7599 ms (34.7706 tokens/s). Packet values match; device time and device roofline fields remain null. Diagnostic teacher-forcing rates are not substituted into this headline.

## Anomaly Ledger

- Observed anomaly: P6 produced malformed or extraneous text despite its focused panel success.
  Evidence: raw fixed108 `anomaly.sympy__sympy-18763.call1.onset.seed42.cap256.json` includes a fenced command followed by `Neptune` and an asserted script result; the completed panel/root records identify regressions 0/32/85/96.
  Affected path: reduced attention/down precision, free-running diagnostic continuation.
  Control or comparison: recorded normal same-case full-BF16 controls; P2 is correctly identified as head-only, not full precision.
  Likely subsystem: precision-sensitive generation; no specific kernel cause established here.
  Investigation performed: broader panel, attribution correction, subsequent precision candidates.
  Resolution: P6 rejected; replacement acceptance remains pending.

- Observed anomaly: caret collapse and localized contraction duplication.
  Evidence: directly read all nine completed HF9 `results/**/*.private.json` outputs. Four are caret-only through 512 tokens. Matplotlib call91 begins `’ve` after an inherited contraction; Sphinx call70 and Matplotlib call86 terminate with tool handoffs. Pytest7432 exits finite traceback repetition and hands off; its unusual quoting has a saved stock-parser acceptance audit and is not independently called a parser failure here.
  Affected path: reconstructed shortened sampled continuations.
  Control or comparison: matched TT records and pinned HF outputs, with documented differing RNG/seed scope.
  Likely subsystem: reference-compatible motifs for the reproduced cases; precision-sensitive branching remains possible elsewhere.
  Investigation performed: shortened controls, cap extensions and forced-history comparisons.
  Resolution: controlled only at demonstrated scope; outstanding cases remain more-work-needed. A tool-yield EOS is not a fabricated tool result.

- Observed anomaly: BF16 activation prefill exhausted L1.
  Evidence: P10 failure and `setup/all-bfp8-lofi-activation-bf16-accuracy-p10-retry/run/readiness_accuracy.json`; scoped `_linear` MLP blocking change.
  Affected path: BF16 prefill MLP activation with reduced weights.
  Control or comparison: same-policy retry completed all accuracy phases; selected BFP8-activation branch is unchanged.
  Likely subsystem: live L1 allocation/block sizing.
  Investigation performed: AutoDebug/AutoFix, branch tests, full-model retry.
  Resolution: allocation defect fixed; policy still rejected on natural accuracy (422/512).

## Scope Inspected

- Goal/skills: AGENTS.md, run_contract.json, inputs.json; stage-review, agentic-serving, datatype-sweep, tt-device-usage and qualitative-check instructions. Operator local-only and deadline decisions take precedence over older contract fields.
- Artifacts: REPORT, results, learned, work log, timing, run ledger; accuracy table and referenced raw/runtime records; HF9 raw outputs and parser correction; P6 raw outputs; SWE collected score; headline raw/summary/logging; both mandatory telemetry packet files.
- Code: diffs for precision_repro_hf, run_datatype_candidate, optimized_decoder, benchmark_work, kv_pool_budget, generator_vllm, collect_candidate_sizing, remeasure_agentic_rates and associated host-test coverage. Inspected new validation/plumbing statically; no newly established implementation defect.
- Commands: read-only `cat`, `sed`, `rg`, `git status/diff` and small standard-library JSON/artifact analyses. No TT commands, model imports, HF loads, servers, profiling, resets, tests or external services. Only this report was written.

## Residual Risk

This inspection sampled completed output evidence and checked every compact accuracy row; it did not independently adjudicate every scored trajectory or all 108 trials. P19 remains unselected. Passing the new 512-position reference cannot establish long-context or served correctness. Unknown verifier outcomes and uncovered anomalies remain explicit; deadline closure must preserve them rather than claim extended-mode readiness. A fresh final review is still required.
