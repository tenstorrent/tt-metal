# Prepared nightly candidates

Three independent source-only patches are ready against TTI bb321bf0f066dd472f51bd0b3f8875d47a0b3cfb:

- r1_gpqa_diamond.patch: frozen nightly 10 documents, planned_clients 2.
- swe_bench_verified.patch: frozen nightly 5 tasks, planned_clients 2.
- terminal_bench_2_1.patch: frozen nightly 10 tasks, planned_clients 2.

Each enables only its named suite in CI_NIGHTLY (peer suite limits zero), retains existing release limits/precision/sampling/context and task-defined resources, and preserves nested smoke membership. No model compute, shared harness, sandbox, selector or network-policy code changed. Patches include model-specific test expectations for the new nightly cohorts. They are alternatives, not a patch stack. Select only after that suite's smoke is valid and its measured projection permits the next rung. The coordinator must record schedule/admission and select the batch; these patches retain two clients rather than assuming capacity from unmeasured smoke.

Stock common eval smoke-test overrides the map with global SMOKE_TEST_EVAL_LIMIT=3. Expanding GPQA doc_ids to nightly10 therefore makes smoke-test select its first3. This is tested as stock behavior, not repaired. The original immutable smoke publication remains the exact smoke2 artifact. Use CI_NIGHTLY for these nightly candidates. No smoke or nightly was run by this preparation.

Host validation: each candidate passed exact stock common/Harbor route selection and peer-disabling checks in verify.py. Each also passed 23 existing model-specific tests and Ruff checks. No device usage. Active publication checkout HEAD and working tree were untouched. Detached preparation worktree was restored clean after all candidates.

## Apply and test after smoke validation

Use a fresh worktree based on bb321bf0f066dd472f51bd0b3f8875d47a0b3cfb. Example for GPQA (replace suite name for the other two):

```bash
git apply --check /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/nightly-preparation/r1_gpqa_diamond.patch
git apply /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/nightly-preparation/r1_gpqa_diamond.patch
/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/.pre-commit/bin/python /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/models/autoports/google_gemma_4_31b_it/doc/agentic_evals/runs/nightly-preparation/verify.py r1_gpqa_diamond
/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/.pre-commit/bin/pytest -q tests/workflows/test_gemma4_autoport_smoke_suites.py tests/workflows/test_eval_config_mode_reference.py
```

Run full publication hooks in the coordinator's clean environment before committing. If its base changed, rebase and reconfirm; no patch here authorizes dispatch before valid smoke. Reproduction scripts prepare.py then prepare_tests.py rebuild these artifacts in the dedicated detached worktree; they are local host preparation tools and must not enter publication commits.
