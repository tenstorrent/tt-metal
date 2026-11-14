# Stage 12 GPQA bracket correction — independent scoped review

## Verdict

**clean-pass**, limited to the generic GPQA bracket-preservation correction and corrected score/cost accounting. No required work in this scope. This is **not** a full Stage 12 pass: nine semantic-anomaly investigations and the extended SWE85 phase are excluded and still require their own evidence and final stage review.

Reviewed the live worktree at tt-metal HEAD `e4f11621bec6c1aa8b070797b88c2401fdf31eeb`, target `google/gemma-4-31B-it`, pinned HF revision `842da3794eaa0b77d5f08bae87a17459d91ff475`. Generic fork: `bringup/artifacts/lm-eval-gpqa-preserve-brackets`, branch `fix/gpqa-preserve-answer-brackets`, commit `6290a8f7e24a7fa8b5e2cfe50e16e2b9b1f49335`, base `af401ef3dd3d2d90e55680990ed1f0c9f7892448`. Review used host-only inspection and short arithmetic/tests; no model requests, TT imports, devices, servers, benchmarks, network fetches, or pushes.

## Evidence and independent checks

- The entire generic-fork diff comprises removal of the square-bracket deletion regex and its unused import, plus 39 lines of synthetic regression tests. Whitespace/title normalization, shuffle, gold-index selection, task YAML, and answer extraction/scoring are unchanged. The fork worktree is clean. Four tests pass on the fixed implementation; executing the same tests against the original source in memory produces six assertion failures and no errors. Cases include mathematical/nested brackets, chemical notation, existing whitespace/title behavior, and correct indices under eight shuffle seeds.
- Re-derived the audit from all **198 original raw sample documents**, rather than trusting `audit.json`. Comparing original and corrected preprocessing over all four source answer fields yields exactly **12** affected IDs: **8, 24, 27, 32, 64, 76, 81, 97, 118, 171, 177, 185**. The other **186** rows are unaffected. Checked every original choice multiset and gold index against the original preprocessing; checked every corrected choice multiset and gold index against bracket-preserving preprocessing. Corrected rows retain the original question, record ID, and source answer fields. Checked actual serialized sample prompts contain their processed choices. No gated question/answer text is reproduced here.
- Raw sample files contain exactly 198 original and 12 corrected unique document IDs, with one recorded sample attempt per task. Re-executed the pinned `process_results_gpqa` over all **210** stored filtered responses; every exact-match score agrees with the raw sample and collected records. This also checks the changed prompts against their own gold indices rather than transferring old answer letters.
- `score-accounting-decision.json` records unconditional replacement of all 12 malformed prompts before the correction run begins. Recomputed every corrected-aggregate task assignment. Original result is **159/198**; the affected original subset contributes **5/12**, corrected subset **3/12**, leaving **154/186** unchanged and **157/198** counted. IDs **24 and 64 regress from correct to wrong and remain counted as wrong**. IDs 97, 171 and 177 remain correct; the other seven affected IDs remain wrong. There is no best-of selection or dropped denominator.
- Independent Wilson calculation with z=1.959963984540054 gives **[0.7311759450484459, 0.8435325448611946]** for **157/198 = 0.7929292929292929**. The 0.843 reference lies narrowly inside the interval. `corrected_aggregate.json`, `results.json` and the current REPORT headline agree; the correction subset is explicitly not a separate representative phase.
- The two raw harness result configurations agree on model/tokenizer revision, route, 12 concurrent clients, thinking override, zero few-shot examples, sampling (temperature 1, top_p .95, top_k 20, seed 42), output cap 126976, max_length 131072 and request timeout 14400 seconds. The saved commands retain the same main wrapper and 12-hour outer timeout. Differences select the isolated generic fork, offline cached inputs, affected IDs and output destination. The source diff does not modify scoring or model-specific behavior.
- Re-ran the host-only installed `explain_hours.explain` arithmetic against saved per-call accounting and `current/head-bfp8-rates/measurement/serving_rates-gpqa.json`. Original replay reproduces **3.5510317337830863 h**, including **24.594241619110107 s** global overhead once; correction replay reproduces **0.4786798774374856 h**, including **11.80755877494812 s** once. Both are nonapproximate in the saved coverage model.
- Run timestamps are original **11:24:19–12:53:07 UTC**, correction **12:54:23–13:05:27 UTC** on 2026-11-14. The **76 s** gap is included once. Recomputed end-to-end wall is **1.6855555555555555 h** and explained cost including the gap is **4.050822722331683 h**. Both waves, including all malformed original executions, are charged. The one-sided wall <= 1.25 × explained gate passes; faster measured time reflects pessimistic replay rates and is not a correction speed claim.
- All 39 original wrong metrics have coordinator rows (7 affected infrastructure, 23 wrong, 9 anomaly); the five originally correct but malformed prompts are also classified infrastructure. All nine corrected wrong metrics have individual coordinator/triage rows. Correction outcomes remain ordinary wrong answers, not a new infrastructure exception. The nine original anomalies remain explicitly pending in results and REPORT; the score gate is not presented as completing their investigation.
- `learned.json` records the generic defect, base/fixed commits, affected IDs, tests and original-result preservation. Original and corrected collected files remain separate. Local-only/no-push is the active run contract.

## Required Work

None for this scoped correction/accounting review.

## Other Concerns

`learned.json.current_v5_phase_status.extended` still contains an earlier running/score-pending snapshot, while the authoritative corrected aggregate and REPORT are current. Refresh this status during the next normal consolidation; it does not contradict the retained raw evidence or invalidate the correction.

`audit.json.native_pending_ids` retains ID 185 as pending at initial audit time. The completed corrected and original raw samples now supply its source/document/prompt evidence; this review checked it together with the other 11. This historical audit note is not missing current evidence.

## Hard-Check Gaps

No additional required check identified for this scope. This pass independently rechecks prompt preservation, scoring, replacement policy and replay arithmetic; it does not remeasure the underlying serving rates or certify model-quality interpretations in the outstanding anomaly controls.

## Anomaly Ledger

The harness defect deletes substantive bracketed answer content, affecting the listed 12 records; ID 171 collapses the four original choices. Generic source inspection, synthetic baseline-negative/fixed-positive tests, and the full sample audit establish a harness subsystem defect. Resolution: generic local patch plus unconditional corrected replacements, with wasted execution cost retained.

Original semantic-anomaly IDs **30, 48, 110, 127, 129, 130, 139, 145, 147** remain first-attempt counted outcomes. Their running TT/HF controls are outside this review and are not waived by this clean-pass.

## Scope Inspected

- Contracts: `bringup/run_contract.json`, `bringup/agentic_evals/inputs.json`, provided AGENTS.md/operator decisions; installed `stage-review/SKILL.md` and `agentic-serving/SKILL.md` (tt-model-bringup 0.1.27).
- Artifacts under this autoport's `doc/agentic_evals/`: `setup/gpqa-bracket-fix/`; `current/gpqa-extended198-head-bfp8/` raw samples, collected, corrected aggregate, coordinator and explained files; `current/gpqa-bracketfix-affected12/` raw samples/results, collected, coordinator and explained files; `results.json`, `runs.json`, `learned.json`, `REPORT.md`; saved GPQA serving rates.
- Code: isolated generic fork preprocessing, tests and unchanged r1 GPQA scoring; installed host replay utility.
- Commands: git status/diff inspection; Python JSON audits and Wilson/replacement/timestamp calculations; four synthetic unit tests; in-memory baseline-negative tests; pinned scoring replay over 210 responses; host replay over two saved accounting files.

## Residual Risk

The statistical pass is narrow and is not proof of equal accuracy. The corrected subset has 3/12 successes, but is selected by an objective prompt defect rather than a representative accuracy sample. The remaining anomaly work and extended SWE phase determine final Stage 12 readiness. This review writes only this report and makes no implementation/result changes.
