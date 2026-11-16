# Stage Review

Verdict: more-work-needed

Independent interim review of stage 12, google/gemma-4-31B-it, live branch `v5-extended-run` at supplied HEAD `8a68d14dbf6788609fe0356d2a8c67e2b37b08d1`, including subsequent live evidence. This is not a closure review. No new blocking correctness or evidence defect was identified in the completed evidence sampled here. The verdict remains more-work-needed because the explicitly disclosed work below is unfinished.

## Required Work

No newly discovered P1/P2 requirement in the completed scope.

Previously disclosed pending inventory (not new findings):

- Complete the admitted 165 matched diagnostic jobs and adjudicate the below-band SWE result, including conditional HF controls or precision remediation when the results require it. The inspected `setup/swe85-remainder-execution85/progress-summary165.json` snapshot has 87 served and one full result, with one verified comparison. The four reviewed supplemental jobs are included in the 165 denominator. Neither a passing policy comparison nor completion of this investigation is claimed.
- Resolve or explicitly retain the three unknown verifier outcomes through the pending operator decision in `setup/requests-verifier-operator-question.json`. Do not assign inferred zero or success, remove these tasks, or claim a point score meanwhile.
- Update final result/report/learned records from completed work, obtain the final whole-stage independent review, and make isolated local stage-owned commits. No push is authorized. This interim review does not authorize stage completion or terminal blocked status.

## Other Concerns

- A small stale sentence remains in `current/swe-extended85-head-bfp8/live-tasks/sympy__sympy-14711-first/coordinator.json`: its analysis says “No caps, discarded responses or observed anomalies” immediately before documenting a localized protocol anomaly. The structured `anomaly: true`, anomaly call, accepted task grade, and current REPORT all preserve the correct distinction. Remove the superseded no-anomalies clause during report cleanup; this is not missing investigation or grounds for another hardware run.
- Phase 2's missing prelaunch full-list forecast is already explicitly disclosed in REPORT; post-hoc explained hours are not presented as the missing forecast. This is an existing planning omission, not a newly discovered execution defect.
- Historical uniform prompt reconstruction gap of minus one is explicitly recorded, without token insertion or claims of native-ID equivalence. Keep that qualification in final diagnostic conclusions.

## Hard-Check Gaps

- Mechanical detector flags miss semantic repetition and localized protocol artifacts. Directly inspected generated continuations support the distinction: Requests5414 caret collapse reproduces in the sampled served trials; SymPy13551 includes both ordinary serialized tool continuations and a stray prose preamble; Requests1921 has normal tool serialization in one sample and stray/incomplete material in others. A raw tool marker alone is not proof of visible-assistant corruption.
- The loaded-policy checker validates the actual runner-reported `generator.model.precision_config`, not the output directory name alone. Its guard covers weight groups, fidelity fields, activation/residual/CCL/KV/head-related types and context; full mode additionally requires HiFi4 SDPA. Source inspection connects these fields to model construction and the decoder policy mapping. This supports current admitted configurations, not a blanket proof for future policies.
- This review sampled raw outputs and checked aggregates; it did not independently repeat all 85 trajectory audits, every semantic fixture validation, or every earlier stage test. Prior scoped reviews remain separately identified evidence, not newly rerun tests.

## Anomaly Ledger

- Observed anomaly: original phase-1 task195 collapse and the selected precision intervention.
  Evidence: REPORT, results run records, original-policy serving-rate snapshot, and prior scoped `review/gpqa-final-controls-headline-review.md`.
  Affected path: original BFP4 head versus selected BFP8/LoFi head.
  Control or comparison: retained original 6/10 result, counted selected-policy 8/10 fixed-list rerun, separate selected-policy rates and original-rate replay.
  Likely subsystem: head precision sensitivity established by the earlier scoped reproducer review.
  Investigation performed: checked result linkage, policy separation, score/timing records and source precision propagation; did not repeat the original device trial panel.
  Resolution: controlled/fixed within the existing three-seed reproducer evidence; no claim of zero population anomaly probability.

- Observed anomaly: phase-2 Astropy repetition and extended GPQA semantic repetition despite mechanical flags often being false.
  Evidence: `anomalies/swe-astropy-precision/short-hf-batch/results/*.hf.private.json`, `anomalies/gpqa-extended-precision/conditional-hf-prep/extension1024/results-seed42/task129.onset.suffix512.hf.private.json`, final control summaries, and prior scoped review.
  Affected path: reconstructed shortened diagnostic inputs on served TT, full TT and pinned HF.
  Control or comparison: directly read both Astropy HF outputs and GPQA129 extended HF output. Astropy repeats unchanged edit planning or the same attribute-access phrase; GPQA129 cycles unchanged wavelength calculations. These agree with the semantic classifications rather than the mechanical detector alone.
  Likely subsystem: reference continuation behavior on these diagnostic inputs.
  Investigation performed: checked raw generated text, summary qualifications and original inconclusive/timeout preservation.
  Resolution: controlled on the shortened inputs. Native long-prompt equivalence, equal incidence and exclusion of earlier precision drift are not established or claimed.

- Observed anomaly: extended SWE native protocol/caret anomalies and below-band score.
  Evidence: final-score-with-adjudication, sampled raw served controls, served-semantic-coordinator-all36, progress-summary165 and pending verifier record.
  Affected path: selected serving policy and ongoing matched diagnostic controls.
  Control or comparison: 36 served onset cases have 108 semantic classifications; the inspected report states 42 normal, 25 native-motif reproductions, 16 other anomalies and 25 inconclusive. Full comparisons remain unfinished.
  Likely subsystem: unresolved; current served observations alone cannot distinguish reference behavior from precision or stack effects.
  Investigation performed: independently recounted retained task outcomes, checked all coordinator links exist, sampled graded/anomalous/unknown coordinator records and raw diagnostic text, and inspected loaded-policy and fixture-alignment checks.
  Resolution: more-work-needed, already tracked in the active investigation. No grade or policy change follows from this review.

## Scope Inspected

- Goal/skill paths: AGENTS.md; bringup/run_contract.json; bringup/agentic_evals/inputs.json; installed 0.1.27 stage-review, model-bringup startup, agentic-serving phase/size/settings/failure-analysis contract, datatype-sweep Anomalies, tt-device-usage serialization and bounded-job rules.
- Artifact paths: `doc/agentic_evals/{results.json,REPORT.md,learned.json,work_log.md}`; corrected GPQA aggregate; final SWE85 adjudication; fresh SWE85 admission and logging probe; operator question; policy-validation and live diagnostic summaries; sampled coordinator and private generated-output artifacts; prior GPQA/headline, collection and supplemental-fixture reviews; `doc/benchmark/{serving_rates.json,serving_rates-stage11-head_bfp4.json}`.
- Aggregate checks: corrected GPQA has exactly 198 unique listed rows, 157 solved, with 186 original and 12 unconditionally corrected-wave results. SWE85 has 85 rows: 23 counted successes, 59 zero outcomes and three null outcomes; all coordinator paths exist. Causes reconcile to 26 wrong, 22 fetched solutions, 11 primary anomalies and three infrastructure outcomes, with 23 passes. The unknown-grade interval envelope remains explicitly separate from an observed point-score interval.
- Admission checks: saved forecast 9.590058h times 1.25 equals 11.987573h, below 12h. Fresh launch evidence retains full context262144, measured free27.448525824GB/device, fixed pool25.691422720GB/device and 12-agent capacity at the131072-token planning working set. Response-body/token/finish check passed before launch. No smaller context or changed scheduler is inferred from measured traffic lengths.
- Code paths: `tools/precision_repro_batch.py`, `tools/precision_repro_probe.py`, `tt/model.py`, policy mappings/validation in `tt/precision.py` and `tt/multichip_decoder.py`, and `doc/agentic_evals/setup/summarize_swe85_tt_controls.py`.
- Commands run: local read-only cat/sed/rg/ls, short Python JSON/text/arithmetic inspections, and git status. No network, TTNN imports, device commands, servers, benchmarks, resets, containers or implementation mutation. Only this report was written.

## Residual Risk

The live worktree can advance beyond the inspected 87/165 served snapshot. Final full-TT comparisons, any conditional HF investigation, unresolved verifier outcomes and whole-stage closure are outside the completed scope. Extended SWE remains below the reference band even if all three unknowns eventually pass. Finishing diagnostics must not be described as making that score pass. Current reporting correctly distinguishes diagnostic resolution from scored performance, shortened controls from native replay, and host-wall diagnostic cost from serving/device performance.
