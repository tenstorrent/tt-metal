# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Resolve the unwaived IFEval accuracy failure.
  Evidence: the unrestricted stock lm-eval task completed all 541 IFEval prompts through `local-completions` at `/v1/completions`. The log records prompt-strict 18.669%, instruction-strict 30.935%, prompt-loose 19.963%, and instruction-loose 32.494%; the configured/report score is prompt-strict 18.67 against publisher 34.3. The configured `gpu_reference_score` in the invalid report merely repeats the publisher number, not a same-command GPU/HF control. No issue URL proves a correct canonical implementation fails identically.
  Why this matters: IFEval is an explicit mandatory text-LLM release gate. Methodology ambiguity may explain the comparison, but disclosure is not a waiver and does not establish TT correctness.
  Required next step: run the identical harness/task revision, raw-completion prompt format, tokenizer/model revision, generation settings, and metric keys against a trusted HF/GPU implementation. If the control passes materially better, diagnose and fix the autoport/serving path and rerun; if it reproduces the TT result, establish the correct publisher methodology or a current row-specific issue with canonical evidence before waiving anything.

- P1: Run and pass the GPQA gate.
  Evidence: `release_run_invalid.log` shows `gpqa_diamond_generative_n_shot` exited before any model request because `Idavidrein/gpqa` is gated. The report has no score and marks the row `FAIL`; there is no linked waiver showing a canonical path fails for a non-autoport reason.
  Why this matters: GPQA/Falcon-equivalent quality is a mandatory gate under the stage contract. Missing credentials are an execution prerequisite, not release-readiness evidence.
  Required next step: obtain authorized dataset access without copying credentials or samples, rerun GPQA through the same external autoport server path, and include the scored row in the regenerated report.

## Other Concerns

- No additional concern remains from the remediation-only rereview. The two accuracy gates above still prevent stage closure.

## Hard-Check Gaps

- None remaining from the remediation-only rereview. `api_smoke_summary.json` now records HTTP 200, the raw completions endpoint, response structure, and 5+8 token usage without copying generated text. `RUN_NOTES.md` now records the focused test command and 175-pass result, exact server launch, reset/list/mesh sequence and result, and copied-artifact inventory.

## Anomaly Ledger

- Observed anomaly: release acceptance says PASS despite two failed accuracy rows.
  Evidence: `release_report_invalid.md` and `release_report_data_invalid.json` show `0/2 passed, 2 waived`, with maturity-only waiver text.
  Affected path: TTI report acceptance, not the generated autoport selection path.
  Control or comparison: local TTI commit `ca152fe2` changes eval enforcement to unconditional and includes all-status failure tests plus explicit-known-issue tests. Offline rendering of the preserved schema through that commit produced `release_report_corrected_fail.md` and `release_report_data_corrected_fail.json`, which report FAIL, two blockers, and zero waivers.
  Likely subsystem: `report_module/acceptance_criteria.py`.
  Investigation performed: inspected the report JSON, invalid markdown, committed diff, test cases, and AutoFix report.
  Resolution: fixed as a reporting anomaly. The corrected artifacts are explicitly an offline rendering of the preserved run, not a fresh or passing model run. The underlying accuracy failures remain more-work-needed.

- Observed anomaly: IFEval is substantially below the publisher score, while alternate aggregates are closer.
  Evidence: prompt strict 18.67, instruction strict 30.94, prompt loose 19.96, instruction loose 32.49, versus publisher 34.3.
  Affected path: accuracy methodology and potentially model/serving correctness.
  Control or comparison: no identical HF/GPU control exists; the publisher card does not establish the same harness recipe.
  Likely subsystem: unresolved between reference methodology and TT generation/serving correctness.
  Investigation performed: inspected the exact lm-eval command, task settings, aggregate output, Base-model prompt-format metadata, report configuration, and TTI AutoDebug analysis.
  Resolution: more-work-needed.

- Observed anomaly: GPQA is reported as an eval failure without a score.
  Evidence: gated-dataset `DatasetNotFoundError` occurs before requests; report records `no eval results parsed (rc=1)`.
  Affected path: release accuracy coverage.
  Control or comparison: none; no canonical scored row or issue waiver.
  Likely subsystem: external Hugging Face dataset authorization.
  Investigation performed: inspected the exact command, traceback, report row, and notes.
  Resolution: more-work-needed.

## Scope Inspected

- Goal/skill paths: `.agents/skills/stage-review/SKILL.md`, `.agents/skills/tti-release/SKILL.md`, `.agents/skills/tt-device-usage/SKILL.md`, and the supplied Stage 11 contract.
- Artifact paths: all files under `models/autoports/tiiuae_falcon3_7b_base/doc/tti_release/`, including the corrected-fail report/data and API smoke summary; `doc/context_contract.json`; and relevant `doc/optimized_vllm` context claims.
- Code paths: TTI commit `ca152fe2` over base `c8509ac2`, especially runtime-spec parsing, remote eval/benchmark callers, eval config, acceptance criteria, `workflow_module/workflows.py`, and their tests.
- Commands run: read-only `sed`, `find`, `jq`, `rg`, `git status/log/show`, process listing, and `tmux list-sessions`. The remediation rereview used only `sed`, `find`, `jq`, `rg`, and `git status/show`. No server, device, container, eval, benchmark, or test was started.

## Residual Risk

- Autoport provenance is well supported: both runtime specs name `models/autoports/tiiuae_falcon3_7b_base`; no stock implementation path appears in the selected implementation fields.
- Context preservation is well supported: the runtime spec carries `max_context=32768`, `max_model_len=32768`, and `max_num_batched_tokens=32768`, matching `doc/context_contract.json`.
- External/no-Docker and trace-disabled smoke are well supported by runtime controls and the successful 1/1 benchmark JSON. The 129-token benchmark request also supports nonaligned prompt handling.
- Direct API smoke is now supported by a privacy-safe summary: health HTTP 200 and one `/v1/completions` response with 5 prompt tokens and 8 completion tokens; generated text was not copied.
- The apparent missing API child is explained by source: `ReleaseWorkflow.llm_children` is exactly `("evals", "benchmarks", "spec_tests")`; parameter conformance is routed through `spec_tests`, whose suite discovery transparently returned no match for this custom model. This remains NA rather than a hidden failure or waiver.
- The invalid original PASS is retained only as diagnostic history. The corrected offline rendering truthfully reports `acceptance_criteria=false`, both eval blockers, and no waivers. A fresh run is still required after the accuracy prerequisites are resolved.
- Cleanup is consistent with current read-only process/tmux inspection: no live vLLM/EngineCore/TTI process or tmux session was found. Defunct historical tmux server processes are not live sessions.
- Release readiness remains unestablished until a corrected report contains passing or validly waived mandatory accuracy rows.
