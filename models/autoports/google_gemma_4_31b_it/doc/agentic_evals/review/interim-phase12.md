# Stage Review

Verdict: more-work-needed

Independent **interim** review of stage 12, completed phase 1/phase 2 evidence and supporting stage-owned code. Snapshot: branch `v5-extended-run`, HEAD `67f50e0d140100d3c478829fd056fba83c0cdfb2`, live worktree, 2026-11-14. This is not final stage closure or authorization to admit extended lists. Paths below are relative to `models/autoports/google_gemma_4_31b_it` unless otherwise stated.

## Required Work

- **P1: Finish the Astropy reference comparison and resolve the reproduced degeneration.**
  Evidence: `doc/agentic_evals/anomalies/swe-astropy-precision/tt-controls-summary.json:1`, actual `served/astropy.onset.json`, `served/astropy.second.onset.json`, and corresponding `full/` outputs. Direct reading confirms repeated intentions and phrase cycles in all twelve continuations, including detector-negative samples. `hf-longprefix-timeout.json:1` records exit 124 with 64 teacher-forced positions but no completed free-running sample; `short-hf-batch/launch.json:1` records the replacement bounded run as running. Partial short-HF evidence is not treated as a completed verdict.
  Why this matters: datatype-sweep's Anomalies step 3 requires reference comparison when full TT precision still fails. The numerical SWE score/timing gates alone cannot classify this as reference-model behavior. Both original capped replies remain part of the first attempt and timing.
  Required next step: complete and inspect the already-running matched shortened-prefix HF controls, compare actual semantic behavior with their reproduced served/full TT counterparts, and record the conclusion and limitations for both onset cases. If HF is coherent while TT is degenerate, investigate the stack by numerical comparisons and repair it; if HF reproduces the same failure, document the controlled model behavior. Apply any resulting precision fix and required remeasurement/list rerun under the skill. Do not call an unflagged sample coherent without reading it. This is open required work, not an operator-authorization blocker.

No additional critical blocker was established in the completed-phase scope inspected.

## Other Concerns

- `doc/agentic_evals/REPORT.md:47` accurately discloses that SWE had no saved full-list prelaunch forecast. Post-run replay does not recreate a prediction. Preserve that historical omission; save the full-suite forecast and actual candidate-list admission projection before any extended launch. The completed phase timing gate itself is independently supported.
- `doc/agentic_evals/setup/readiness_network_deviation.json:1` records network-capable installer downloads during initial image preparation despite the no-GitHub instruction; the actual release destination was not logged. This prevents an unqualified claim of strict no-GitHub compliance. Subsequent offline preparation and failed metadata fetches are distinguished from acquired task solutions. Preserve the disclosure and the enforced offline route; no retrospective rerun can undo the preparation action.
- `doc/agentic_evals/REPORT.md:49` describes the earlier three-seed HF plan, while the replacement `short-hf-batch/launch.json` uses seed 42 and 512 output tokens. Refresh this narrative when controls settle. This is a live-worktree reporting note, not a second independent closure blocker.
- GPQA's saved replay uses `current/head-bfp8-rates/measurement/serving_rates-gpqa.json`. The current combined `doc/benchmark/serving_rates.json` produces 0.1760410801 hours for the same calls, whereas the phase-specific rate artifact reproduces the claimed 0.1926410801 hours. Preserve the phase-specific source when recording replay provenance. Both estimates satisfy the one-sided timing gate.

## Hard-Check Gaps

- No new hard-check evidence format is demanded. The existing logs, exact task identities, native response bodies, hash-checked grading recovery, and accounting artifacts substantiate the completed-phase metrics.
- Mechanical degeneration flags miss semantic phrase loops. Supplemental semantic triage addresses this for the Astropy controls; a machine-only pass cannot supersede it.
- Extended results, final telemetry reconciliation, final evidence commits, and final clean review are deliberately outside this interim scope. Their absence is not presented as an additional finding here.

## Evidence Re-derived

- Official GPQA sample rows give original BFP4 **6/10**, and selected BFP8-head **8/10**, with selected failures 93 and 167. Read the actual generated answer tails for all ten selected tasks. The original score remains preserved; the selected run is a complete fixed-list precision-fix rerun rather than a replacement single-task score.
- Independently recomputed Wilson 95% intervals: 8/10 = **[0.4901624715, 0.9433178485]**, containing 0.843; 4/10 = **[0.1681803297, 0.6873262303]**, containing 0.648.
- GPQA phase-specific rate replay, using installed `explain_hours.py` plus the recorded single global overhead, reproduces **0.1926410801 h**. Measured **0.1293296303 h** passes. Stage timing places the selected rerun and phase-2 start before the 12-hour handoff boundary.
- SWE independently recounts **434 calls** across ten tasks, with exactly two `length` finishes, both Astropy. Each task's lifecycle minus native call durations matches its recorded outside time. Installed rate replay reproduces **1.5230584178 h** against measured **1.4036113561 h**. Long-context extrapolation is correctly labelled approximate.
- Re-executed the host-only grading overlay in memory: all nine recovery records satisfy identity, raw-result, original-log, task-config, parser, report, and provenance checks. Four are resolved: Django, Matplotlib, Sphinx, Pylint. The overlay correctly leaves SymPy unresolved because its native reward is absent. Its separate zero-score adjudication is supported by the agent overwriting the initializer needed by setup.py, the retained verifier failure, and fresh task triage; it does not manufacture a native reward or remove the task from the denominator.
- Failed SWE tasks have fresh triage tied to concrete patches/verifier failures. Their completed submissions are distinct from the two within-response Astropy anomalies. No progressing limit cutoff requiring a doubled limit was identified in the complete native-call accounting.
- The GPQA precision reproduction contains visible apostrophe repetition in BFP4/LoFi and BFP4/HiFi2, one of three samples each. BFP8-head and full BF16 controls show no equivalent collapse in the three-sample panel. Trace counters record 1023 replay steps with zero new captures in 1024-token controls. The selected config changes only the head precision; full-control SDPA and BF16 block-size plumbing are explicit. Resumed-prefill and matched greedy qualitative acceptance artifacts are retained. This supports the tested intervention, not a claim of zero future anomaly probability.
- Fresh candidate memory evidence accounts for the 168 MiB/device head increase and validates the fixed pool against launch allocator free space. The shared-mesh accounting basis and analytical transient reserve are disclosed. Serving context remains 262144. Ten agents fit the recorded conservative working-set capacity; B32 is the smallest measured 1/8/32 batch that holds ten.
- The saved short-request logging preflight records body, token counts, and finish reason before selected GPQA traffic. SWE collection retains discarded agent responses and excludes the identified logging probe. Original bodies and capped replies remain available.

## Anomaly Ledger

- Observed anomaly: GPQA task195 token collapse under original precision.
  Evidence: original official samples/digest and `anomalies/gpqa-phase1-precision/timed-onset-*/task195.onset.json`.
  Affected path: sampled autoregressive output with BFP4 head.
  Control or comparison: same reconstructed onset under BFP4/LoFi, BFP4/HiFi2, BFP8/LoFi head, full BF16/HiFi4; selected full-list rerun.
  Likely subsystem: precision sensitivity at the output head; short panel does not establish exclusive causality.
  Investigation performed: traced runner reproductions, policy comparison, selected-policy acceptance, serving-rate remeasurement and full fixed10 rerun.
  Resolution: controlled/fixed for the tested panel and phase-1 acceptance; residual stochastic risk remains.

- Observed anomaly: two Astropy capped replies and semantic loops in served/full TT diagnostic continuations.
  Evidence: native response IDs `chatcmpl-9d9f50568802ad41`, `chatcmpl-bdff2676d029ceb6`; actual served/full onset outputs.
  Affected path: SWE reasoning before tool execution.
  Control or comparison: both onsets at served and full TT precision, three seeds each; shortened reproduced panels; pending pinned HF control.
  Likely subsystem: model behavior versus stack behavior remains unresolved; higher precision alone does not remove it.
  Investigation performed: complete native reply mapping, fresh anomaly triage, full TT controls, semantic review beyond detector flags, bounded long-HF attempt and shortened replacement.
  Resolution: more-work-needed, required item above.

- Observed anomaly: verifier wrapper yielded zero/missing rewards despite completed test logs.
  Evidence: `current/swe-phase2-ready/verifier-recovery-final/`, raw trial results, separate SymPy adjudication.
  Affected path: offline grading, and separately a model-damaged SymPy package setup.
  Control or comparison: pinned official grading over original test output with identity/hash checks; generic fix branch `fix/swebench-offline-log-grading`, commit `e98973d132ec705baa4f59ca62680587ef5281a9` recorded in recovery provenance.
  Likely subsystem: unused network-dependent metadata construction in grading; SymPy failure is model-created workspace corruption.
  Investigation performed: nine recovered reports, preservation of raw rewards, concrete six-failure task triage, in-memory overlay validation by reviewer.
  Resolution: controlled for scoring; no model attempts rerun or dropped.

- Observed anomaly: initial setup used network-capable installer provisioning.
  Evidence: `setup/readiness_network_deviation.json`, preparation logs and subsequent offline validation records.
  Affected path: task-image preparation, not fetched task solutions.
  Control or comparison: offline local provisioning/validation and failed blocked metadata requests.
  Likely subsystem: mistaken build-cache assumption.
  Investigation performed: exact download records and offline remediation retained.
  Resolution: operationally contained, historical instruction deviation remains disclosed.

## Scope Inspected

- Goal/skill paths: checkout `AGENTS.md`, `bringup/run_contract.json`, `bringup/agentic_evals/inputs.json`; installed model-bringup 0.1.27 stage-review, agentic-serving, datatype-sweep, tt-device-usage; tt-review-core contract.
- Artifact paths: `doc/agentic_evals/{results.json,runs.json,learned.json,REPORT.md}`; selected and original GPQA official samples, selected calls/replay/logging check; SWE `collected-final` calls, grading overlay and all failure triages; `verifier-recovery-final` reports/provenance; solution-fetch audit and preparation deviation; precision controls and selected acceptance; candidate memory audit and phase-specific serving rates.
- Code paths: stage diff from `45d940bf` through HEAD, particularly `tt/generator_vllm.py`, `tt/serving_memory.py`, BF16/SDPA guards in `tt/multichip_decoder.py`, selected precision config, and host-only `tools/{agentic_collect_swe.py,agentic_overlay_swe_rewards.py,precision_repro_hf.py}`; supporting collector/replay code used in the read-only arithmetic checks. This was a focused stage review, not an exhaustive re-review of stages 1–11 or every helper line.
- Commands run: read-only `cat`, `sed`, `rg`, `git status/log/diff`; small Python JSON/output inspection, Wilson arithmetic, call recount/lifecycle checks, installed timing replay, and in-memory recovery-overlay validation. No hardware commands, servers, resets, Docker executions, long tests, network requests, source changes, or commits. Only this authorized report was written.
- Push state: operator run plan prohibits pushes and requires local commits only. No push was performed or inferred from reflog.

## Residual Risk

Ten tasks give wide score intervals. Reconstructed diagnostic prefixes are not exact native sampled token histories; count discrepancies and HF/TT RNG differences constrain causal conclusions. Short passing controls do not prove long-trajectory correctness. The worktree is live, so later HF outputs or code changes require follow-up review. Extended admission and final stage closure still need their own evidence after the open anomaly is resolved.
