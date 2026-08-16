# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Establish a valid comparison or row-specific issue waiver for both mandatory accuracy rows.
  Evidence: `TTI_RELEASE_REPORT.md` marks `meta_ifeval` and `meta_gpqa_cot` as `NA`, with both published and GPU-reference ratios `N/A`. `tti_release_report_data.json` likewise has null published/reference scores. `RELEASE_WAIVERS.md` waives the missing baselines locally, but cites no current issue, release note, or canonical implementation result. The report nevertheless says all acceptance criteria passed, and `RUN_NOTES.md` calls the stage passed.
  Why this matters: The release and stage-review contracts explicitly classify missing/incomparable accuracy rows as required work unless a row-specific current issue or release note demonstrates that the canonical implementation has the same non-autoport limitation. Running the tasks and obtaining nonzero scores proves execution, but does not prove that the mandatory quality gates passed. A local disclosure is not a valid waiver.
  Required next step: Add evidence-backed reference thresholds/results for `meta_ifeval` and `meta_gpqa_cot` and regenerate the report, or cite a current row-specific issue/release note with canonical control evidence that validly waives each missing comparison. Until then classify the result as `release-workflow-pass/readiness-fail`, not a Stage 11 readiness pass.

- P2: Record the required quantitative unrestricted-runtime projection for the CI-nightly exception.
  Evidence: `RUN_NOTES.md` records the 5% limit and says only that an unrestricted sweep would scale the workload "materially." It provides no projected unrestricted runtime and no reservation-window comparison. The copied GPQA evidence records 10 effective samples out of 198 and a measured total evaluation time of about 12,773 seconds; IFEval records 28 of 541 and about 318 seconds, so the artifacts contain enough data to make an explicit estimate.
  Why this matters: The TTI release contract requires an estimate before using `ci-nightly`, plus why unrestricted execution was prohibitive. This distinguishes a justified nightly subset from an arbitrary reduction in release coverage.
  Required next step: Add a numeric projection based on measured task runtimes/sample counts (with assumptions, especially GPQA concurrency and long-tail behavior), compare it with the available reservation window, and retain the `release-readiness-ci-subset-*` classification.

- P2: Finish and accurately record local checkpoint/cleanup state.
  Evidence: The tt-metal handoff is committed locally as `a08f7ac8f33` on `mvasiljevic/fmf/google-gemma-4-26b-a4b-it` and is one commit ahead of its remote, while the five requested TTI commits end at `daa1fe6f` on `codex/gemma4-stage11`. However, `RUN_NOTES.md` still says `Final tt-metal artifact commit: pending` and `Stage review: pending`. The TTI checkout also has two untracked stage-related files, `AUTODEBUG_GEMMA4_EVAL_PARSER.md` and `AUTOFIX_GEMMA4_STAGE11.md`.
  Why this matters: The stage contract requires local commits with no push, compact cleanup, and an exact handoff record. Stale "pending" entries and unexplained stage-owned untracked files make completion and ownership ambiguous.
  Required next step: Update `RUN_NOTES.md` with both repo paths, branches, and final local commit SHAs; record this review path; then either commit the two TTI documents if they are stage evidence or remove/archive them through the stage owner’s normal cleanup flow. Keep commits local and do not push.

## Other Concerns

- The copied `logs/autoport_server.log` is zero bytes. This is not independently blocking because the runtime spec, OpenAI response, benchmark/eval JSON, and TTI logs tie successful requests to the expected model and endpoint, but the empty file should not be described as server-log evidence.
- `logs/tti_release_ci_nightly_final.log` is an earlier failed run (`rc=1`); the definitive successful log is `logs/tti_release_ci_nightly_gate.log` (`rc=0`). The names are easy to misread. `RUN_NOTES.md` should explicitly identify the gate log as definitive or remove stale copied attempts if they are no longer useful.

## Hard-Check Gaps

- I did not start a server, run vLLM, open devices, or re-run hardware tests, per reviewer scope. Cleanup and post-run device-health statements are therefore supported only by `RUN_NOTES.md`, not independently re-observed.
- Raw eval sample dumps were intentionally removed. Aggregate result JSON proves task identity, chat-template use, sample fraction, scores, context, and absence of reported inference failures, but direct inspection of individual generated GPQA/IFEval outputs was not possible.
- The TTI commits were inspected as local commit diffs; their unit tests were not rerun during this read-only review.

## Anomaly Ledger

- Observed anomaly: Mandatory eval rows have real scores but are reported as `NA` while overall acceptance is `PASS`.
  Evidence: `TTI_RELEASE_REPORT.md`, `tti_release_report_data.json`, and `RELEASE_WAIVERS.md`.
  Affected path: Stage 11 release-readiness classification.
  Control or comparison: No published score, GPU reference, canonical-control result, or current linked issue is supplied for either row.
  Likely subsystem: Release-spec reference metadata and acceptance/report policy.
  Investigation performed: Compared final markdown and structured report data with the release/stage-review waiver rules and inspected the TTI result-alias changes in `e3ea566d`.
  Resolution: more-work-needed.

- Observed anomaly: Earlier copied release logs fail GPQA timeout and conformance, while the definitive run passes.
  Evidence: `tti_release_ci_nightly_final.log` ends `rc=1`; `tti_release_ci_nightly_gate.log` ends `rc=0`, with both eval rows, 8/8 benchmark requests, and 22/22 conformance cases represented in copied final artifacts.
  Affected path: Eval client timeout and vLLM parameter-conformance harness.
  Control or comparison: Commits `82e52455`, `61473555`, and `daa1fe6f`; final eval/conformance JSON and gate log.
  Likely subsystem: TTI client timeouts and generic conformance assertions, not model context reduction.
  Investigation performed: Inspected the relevant commit diffs, final commands (`max_length=262144`, GPQA `max_gen_toks=32768`, timeout 14400), and definitive gate result.
  Resolution: controlled; the repaired definitive run passed. Stale-log naming remains an Other Concern.

- Observed anomaly: Runtime captured spec contains a non-null packaged Docker image even though the run is declared no-Docker.
  Evidence: `tti_runtime_model_spec.json` contains an inferred image string, but both embedded `cli_args` and `runtime_config` set `docker_server=false`, `local_server=false`, service port 8000, and the autoport implementation path.
  Affected path: Server topology evidence.
  Control or comparison: Definitive TTI command omits Docker; logs print both server booleans false; report metadata says API mode.
  Likely subsystem: TTI runtime-spec enrichment of version metadata.
  Investigation performed: Compared source spec, captured runtime spec, final command, and logs.
  Resolution: controlled; the image is metadata and does not contradict the executed external-server topology.

## Scope Inspected

- Goal/skill paths: `.agents/skills/stage-review/SKILL.md`; `.agents/skills/tti-release/SKILL.md`; supplied Stage 11 contract for `google/gemma-4-26B-A4B-it`.
- Artifact paths: `models/autoports/google_gemma_4_26b_a4b_it/doc/tti_release/`; `doc/context_contract.json`; `doc/optimized_vllm/README.md`, `work_log.md`, and selected before/after JSON evidence.
- Code paths: TTI checkout `/home/mvasiljevic/tti-gemma4-stage11`, commits `02e81d32`, `e3ea566d`, `82e52455`, `61473555`, and `daa1fe6f`.
- Commands run: read-only `sed`, `find`, `cat`, `rg`, JSON inspection, `git status`, `git log`, `git show`, `git branch -vv`, and `git ls-files`/history checks.

## Residual Risk

- Even after the documentation/checkpoint findings are closed, the mandatory eval comparison finding must be resolved with external reference or issue evidence; raw nonzero CI-subset scores alone do not establish release quality.
- The CI-nightly scores have small effective sample counts (IFEval 28, GPQA 10) and should never be presented as unrestricted full-set accuracy.
- The conformance repair accepts any deterministic fixed-seed content change as evidence of penalty effect. The final seed reproducibility/non-uniform tests reduce the stochastic-confound risk, but this remains weaker than token/logit-level parameter validation and should be tracked as residual test-strength risk rather than a release blocker given the passing suite.
