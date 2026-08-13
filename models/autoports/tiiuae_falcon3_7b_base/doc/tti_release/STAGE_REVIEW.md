# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- Stage 11 establishes CI-nightly subset readiness, not unrestricted full-set
  readiness. This is explicitly allowed by the goal and is now stated at the
  top of `RUN_NOTES.md`. The historical unrestricted failure artifacts remain
  clearly labeled diagnostic evidence.
- Spec tests are NA because TTI has no matching Falcon3 Base custom-model
  suite. This is transparently reported, not waived. The required direct health
  and completions smoke plus successful OpenAI-compatible eval and benchmark
  traffic supply the available API-path evidence.

## Hard-Check Gaps

- The paired HF controls are retained as aggregate/config evidence rather than
  raw samples, consistent with the requirement not to copy large raw eval
  dumps or generated text. Exact snapshot, task versions, seed, raw-prompt
  mode, zero-shot configuration, deterministic first-5% document selection,
  and effective sample counts are recorded.

## Anomaly Ledger

- Observed anomaly: the first passing nightly run left lm-eval's API backend at
  its 2,048-token default despite a 32,768-token model contract.
  Affected path: TTI `local-completions` command construction.
  Control: custom and loaded runtime specs both carry `max_context=32768` and
  server `max_model_len=32768`.
  Investigation/fix: TTI commit `e26e723bf0266cde85f674e381fbee10068ae0ec`
  propagates `DeviceModelSpec.max_context` to text API evals and adds focused
  tests. The regenerated log shows `max_length=32768` in both IFEval and GPQA
  commands and `Using max length 32768 - 1` from each backend.
  Resolution: fixed and verified by a fresh full release workflow.

- Observed anomaly: TT scores differ by one item or improve over their paired
  BF16 CPU controls (IFEval 6/28 versus 5/28; GPQA 6/10 versus 5/10).
  Affected path: subset accuracy acceptance.
  Control: same-snapshot, same-task/version, raw-prompt, zero-shot, seed-42 HF
  controls in `hf_paired_ci_references.json`.
  Investigation: inspected mode-specific references, sample-count-aware
  acceptance implementation/tests, final effective counts, and report rows.
  Resolution: controlled. Both mandatory TT rows pass without waivers and the
  unrestricted publisher/reference configuration was not weakened or re-keyed.

- Observed anomaly: an earlier report incorrectly waived failed evals based on
  experimental model maturity.
  Affected path: TTI acceptance reporting.
  Control: retained invalid report, corrected FAIL rendering, fixed acceptance
  tests, and the fresh context-preserving nightly report.
  Resolution: fixed. The final report has two of two evals passing, zero
  blockers, and zero waivers.

## Scope Inspected

- Original Stage 11 goal and full stage-review contract.
- All current files under
  `models/autoports/tiiuae_falcon3_7b_base/doc/tti_release/`, optimized-vLLM
  checkpoint evidence, and `doc/context_contract.json`.
- Regenerated release markdown, report data, runtime spec, complete run log,
  benchmark JSON, smoke evidence, paired references, provenance excerpt, and
  current `RUN_NOTES.md` modification.
- TTI commits `ca152fe2`, `bd15f1cd`, and final
  `e26e723bf0266cde85f674e381fbee10068ae0ec`, including the context propagation
  implementation and focused tests; tt-metal artifact commits
  `53521e54ffb856b331a3015ffd6320ed9a1a8412` and
  `2e76369011bbd804609bbadf9c3ab2539de3ae60`.
- Read-only cleanup checks. No server, container, test, eval, benchmark, or TT
  device was started by this review.

## Verified Gates

- Optimized-vLLM artifacts exist and predate release; that stage's independent
  review is `clean-pass`.
- The selected implementation is
  `models/autoports/tiiuae_falcon3_7b_base`. Runtime and server provenance name
  `models.autoports.tiiuae_falcon3_7b_base.tt.generator_vllm`; no selected stock
  `models/tt_transformers`, `models/demos`, or other packaged path appears.
- The custom spec embeds `workflow=release`,
  `limit_samples_mode=ci-nightly`, `docker_server=false`,
  `local_server=false`, `service_port=8000`, and trace capture disabled. The
  loaded runtime spec preserves those controls.
- Context is consistently 32,768 in `context_contract.json`, custom and loaded
  specs, server arguments, and now both eval adapter commands. No prompt
  alignment, truncation workaround, or invalid-request waiver was introduced.
  The required smoke also proves a valid non-aligned five-token prompt works.
- The required pre-release external/no-Docker smoke passed health, one raw
  OpenAI completion, and a 1/1 small benchmark with trace capture disabled.
- The regenerated release workflow completed with rc=0 and acceptance PASS:
  IFEval PASS, GPQA PASS, zero failed mandatory rows, zero blockers, and zero
  waivers. The graded 128/128/concurrency-1 benchmark passes all target tiers;
  all 13 sweep points completed successfully through ISL 16,384, with 12
  transparently ungraded NA rows rather than missing metrics.
- Hardware recovery, reservation-container serving, exact commands, key
  non-secret environment, report paths, server mode, host context, TTI SHA,
  no-Docker decision, pass/fail summary, autoport proof, and artifact inventory
  are recorded in `RUN_NOTES.md`. Its current status and SHA ledger supersede
  the historical diagnostic sections unambiguously.
- TTI and context-preserving tt-metal changes are locally committed and were
  not pushed. The remaining `RUN_NOTES.md` and this review are final
  review-record changes for the stage owner to commit. The unrelated untracked
  `third_party/tt-metal/` tree is untouched.
- No live vLLM, EngineCore, TTI runner, or lm-eval process exists; no tmux
  session or TCP port-8000 listener remains. Docker is absent in the reservation
  container and was not used.

## Residual Risk

- GPQA's ten-sample and IFEval's 28-sample nightly subsets have coarse
  quantization. The paired controls and sample-count-aware policy make the
  result valid for nightly readiness, but it must not be advertised as a
  full-set accuracy result.
- Parameter-level spec-test coverage remains unavailable for this custom model;
  the handoff correctly reports that limitation as NA rather than silently
  claiming coverage.

## Runner-side verification remediation — 2026-08-13

- Reproduced `.agents/prompts/model_bringup_multigoal/11-tti-release.check.sh`
  with `MODEL_DIR=models/autoports/tiiuae_falcon3_7b_base` and
  `HF_MODEL=tiiuae/Falcon3-7B-Base`; it exited 2 because the valid copied final
  report was named `release_report_ci_nightly_pass.md`, while the handoff gate
  requires the native TTI `report_*.md` naming contract.
- Renamed only that final report to
  `report_tiiuae__Falcon3-7B-Base_2026-08-13T105032+0000.md`, matching its
  embedded `report_id`. No model, server, spec, context, raw result, acceptance
  decision, or report content changed, so rerunning hardware workflows would
  not add relevant evidence.
- The first remediation rerun then exposed a second latent handoff-format
  defect: `RUN_NOTES.md` lacked the gate's literal “Autoport implementation
  check” label even though its provenance section contained the underlying
  evidence. A second rerun showed that the gate requires the label and target
  path on the same physical line; the PASS line was formatted accordingly.
- The next rerun reached artifact provenance and showed that the gate accepts
  `run_specs/*.json`, report data containing `code_path`, or a copied
  `*model*spec*.json`; the canonical final runtime spec was present but named
  `release_runtime_spec_ci_nightly.json`. Renamed it to
  `runtime_model_spec_ci_nightly.json` so the unchanged embedded
  `runtime_model_spec.impl.code_path` is discoverable by the independent gate.

Fresh independent rereview verdict: `clean-pass` with no required work. The
reviewer confirmed both renamed artifacts are byte-identical to their prior Git
blobs, the native report filename matches its embedded report ID, the runtime
spec preserves the autoport/no-Docker/32,768-token contract, the runner exits
0, and no serving process or copied `.env` remains.
