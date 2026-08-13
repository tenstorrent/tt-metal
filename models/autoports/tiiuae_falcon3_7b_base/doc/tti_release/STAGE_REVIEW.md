# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Make the final TTI eval client preserve the advertised 32,768-token
  context contract, then regenerate the nightly release evidence. The final
  run log records `local-completions` as `Using max length 2048 - 1` for both
  IFEval and GPQA, while `doc/context_contract.json`, the release spec, runtime
  spec, and server all advertise 32,768. The sampled prompts happened to fit,
  but that does not satisfy the explicit requirement that release evals
  preserve the context contract. Configure the local-completions adapter's
  effective maximum from the model spec (32,768 here), verify the final command
  or log reports that value, and rerun the no-Docker nightly release. Do not
  shorten or align the samples.

- P1: Update the handoff's current status and commit ledger. `RUN_NOTES.md`
  still begins with `Stage 11 is not ready to pass` and says no replacement
  passing report exists, even though a later appended section documents the
  passing CI-nightly run. It also does not record the final tt-metal release
  artifact commit `53521e54ffb`. Replace the stale top-level status with the
  current nightly-equivalent PASS claim (while retaining the unrestricted
  failure as historical evidence) and record final stage-owned tt-metal and TTI
  SHAs. This is a release handoff, so a reader must not have to reconcile two
  contradictory readiness states.

## Other Concerns

- Spec tests remain NA because no custom-model suite matches Falcon3 Base.
  This does not block the explicitly required Stage 11 nightly-equivalent gate:
  the earlier health/completions smoke and successful OpenAI-compatible eval
  and benchmark traffic supply the available API-path evidence. It must remain
  disclosed as NA, not represented as spec-test coverage.

## Hard-Check Gaps

- The copied paired-reference JSON contains aggregate/config evidence rather
  than per-sample outputs, as required by the privacy and artifact-size
  constraints. Exact subset identity is supported by deterministic first-5%
  selection, task versions, seed, snapshot, sample counts, and reference labels;
  raw prompt/completion dumps were appropriately not copied.

## Anomaly Ledger

- Observed anomaly: the final TT subset scores differ by one item or improve
  over their paired BF16 CPU controls (IFEval 6/28 versus 5/28; GPQA 6/10 versus
  5/10).
  Affected path: accuracy acceptance.
  Control: exact-snapshot, same task/version, raw-prompt, zero-shot, seed-42,
  first-5% HF controls in `hf_paired_ci_references.json`.
  Investigation: inspected the mode-specific TTI references, generic
  sample-count-aware acceptance implementation/tests, final report data, and
  effective sample counts in the run log.
  Resolution: controlled. Both TT results are at least as good as the paired
  references and pass without a waiver; unrestricted publisher references were
  not replaced.

- Observed anomaly: the historical unrestricted report claimed PASS while both
  evals failed or were missing.
  Affected path: TTI acceptance reporting.
  Control: corrected FAIL render and the later fresh CI-nightly report.
  Investigation: inspected the retained invalid/corrected artifacts and TTI
  commits `ca152fe2` and `bd15f1cd`.
  Resolution: the acceptance bug is fixed; the invalid report is clearly kept
  as diagnostic history. The final nightly report has zero blockers and zero
  waivers.

- Observed anomaly: the final run says the API eval backend uses a 2,047-token
  maximum while the served model contract is 32,768.
  Affected path: TTI local-completions evaluation adapter.
  Investigation: compared the exact eval command/log with the custom and loaded
  runtime specs and `context_contract.json`.
  Resolution: unresolved; required work.

## Scope Inspected

- Original Stage 11 prompt and the full stage-review skill contract.
- All committed files under
  `models/autoports/tiiuae_falcon3_7b_base/doc/tti_release/`, the optimized-vLLM
  checkpoint, and `doc/context_contract.json`.
- Final release markdown/data/runtime spec/log/benchmark, smoke summary and
  report, provenance excerpt, paired-reference aggregate, git history/status,
  and TTI commit `bd15f1cdcf1bbb12187bd68b120e814b7e8a1e83` over the preceding
  AutoFix commit.
- Read-only cleanup checks: no live vLLM, EngineCore, TTI runner, or lm-eval
  process; no tmux session; no listener on TCP port 8000. Docker is not
  installed in the reservation container, consistent with the no-Docker path.
  No server or TT device was opened by this review.

## Verified Evidence

- Optimized-vLLM artifacts predate Stage 11 and their independent review is
  `clean-pass`.
- Custom and loaded specs use external serving with `docker_server=false`,
  `local_server=false`, `service_port=8000`, embedded `workflow=release`,
  `limit_samples_mode=ci-nightly`, and trace capture disabled.
- Runtime provenance selects
  `models/autoports/tiiuae_falcon3_7b_base`, names
  `tt/generator_vllm.py`, and the server provenance records the import
  `models.autoports.tiiuae_falcon3_7b_base.tt.generator_vllm`; no selected stock
  `models/tt_transformers` or `models/demos` path appears.
- The required pre-release smoke passed health, one raw OpenAI completion with
  a non-aligned 5-token prompt, and a 1/1 no-Docker benchmark.
- The final nightly report is a fresh no-Docker release: acceptance PASS,
  IFEval PASS, GPQA PASS, 1/13 graded benchmark rows PASS with 12 transparent NA
  sweep rows, zero acceptance blockers, and zero waivers. All benchmark requests
  succeeded, including ISL 16,384.
- Server/spec context remains 32,768 with block size 32 and no benchmark
  alignment workaround. The remaining context issue is specifically the eval
  client's reported maximum.
- TTI changes and the release artifacts are locally committed and were not
  pushed; the unrelated untracked `third_party/tt-metal/` tree remains
  untouched.

## Residual Risk

- CI-nightly is a small deterministic subset, especially GPQA at ten samples;
  the contract explicitly permits nightly-equivalent readiness instead of an
  approximately 45-hour CPU full-set control. The handoff must label this as
  subset readiness, not unrestricted readiness.
- Final closure depends on eliminating the eval-client context contradiction
  and rereviewing the regenerated report and corrected notes.
