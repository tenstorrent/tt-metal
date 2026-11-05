# PC-02: validate allowed tokens independently of visible special-token text

Status: **authorized and applied; affected live checks passed; full profile passed; stage review clean-pass; local commit `90f33d2e392eb8bfda6948df9dabe712a724cb1a` complete**. The operator authorized PC-02 and PC-03 in checkout `AGENTS.md`, decision 3 dated 2026-11-05 22:30 UTC. Only plugin tests changed. Proposed patch: `PC-02-allowed-token-ids.patch`, against `tenstorrent/vllm-tt-plugin` commit `ca7a18246e4111629ed6cb4a7d1192002749fccf`.

<!-- current-plugin-validation -->
Current validation for the complete PC-02–PC-05 test tree: pinned hooks passed; full host-stub suite **2303 passed, 16 skipped, 5 warnings in 109.37 seconds** with pytest 8.4.2 (`readiness_vllm/plugin_pc02_pc05_host.log`). Affected live checks and all three calibrated top-k cases passed. Each top-k case predeclared two cohorts of 80 draws before sampling (`readiness_vllm/final_topk_calibrated.log`).

Final canonical full sampling passed: **76 passed, 75 skipped, 2 warnings in 512.16 seconds** (`readiness_vllm/final_full/sampling_tests.log`). Passed-test stdout retains all three predeclared cohort plans. Independent stage review returned clean-pass; committed locally as `90f33d2e392eb8bfda6948df9dabe712a724cb1a`. No push.

Runtime `src/`, interfaces and serving defaults are unchanged.
<!-- /current-plugin-validation -->

## Requirement and failure

The full sampling profile must verify host-compatible `allowed_token_ids` enforcement. The original `tests/tt/test_host_only_params.py::TestHostOnlyParameters::test_allowed_token_ids` requests allowed IDs `[1,2,3]`, then requires nonempty decoded text. With pinned Gemma tokenizer revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, these IDs are `<eos>`, `<bos>`, and `<unk>`; all are special and decode to empty text with `skip_special_tokens=True`. vLLM's default is True (`vllm/sampling_params.py`, `SamplingParams.skip_special_tokens`), and `tests/tt/utils.py::send_request` does not override it. The observed full-profile assertion is therefore incompatible with valid API behavior. The live response/token-ID reproducer passed: [1,2,3] emitted EOS ID1 and empty text; all five allowlists satisfied token membership and usage counts. Explicit skip_special_tokens=false exposed ten EOS tokens; an ordinary-token allowlist produced visible A characters. Evidence: readiness_vllm/allowed_token_ids_live.json.

Minimal live reproducer, using the running model: POST `/v1/completions` with prompt `Allowed: `, `max_tokens=10`, `allowed_token_ids=[1,2,3]`, and `return_token_ids=true`. Preserve returned token IDs, usage, text and finish reason. Nonempty generated IDs in the allowlist and empty visible text are consistent; visible text is not evidence of successful allowlist enforcement.

## Ownership and model-side alternative

This is a shared plugin **test defect**, not a model capability gap. The model could produce visible text only by emitting a disallowed token or changing special-token detokenization semantics. Neither is a valid model-side repair. The proposed test checks actual IDs and usage, preserving all five original low-ID allowlists and adding an ordinary-text allowlist derived by the serving tokenizer.

## Exact proposed changes

- `tests/tt/utils.py`: optional test request field `return_token_ids`, forwarded through the existing upstream completion parameter.
- `tests/tt/test_host_only_params.py`: retain all original allowlists; request full responses; require nonempty generated IDs, membership in the request allowlist, and matching usage. Add a visible ordinary-token case derived through the existing `/tokenize` API with `add_special_tokens=false`. No model name/revision branches or local tokenizer dependency.
- `tests/test_allowed_token_contract.py`: host regression proving a valid special-only response can have empty text, while disallowed IDs, missing IDs and inconsistent usage fail.

No `src/` files, serving defaults, model capabilities, plugin options, environment variables, registration, sampling implementations, or model-facing interfaces change. The added test helper is not a runtime plugin interface. Existing `return_tokens_as_token_ids` controls logprob spelling and is a different parameter; it cannot replace raw completion `return_token_ids`.

## Existing mechanisms and other models

No capability, additional-config option or registration mechanism can make special-only visible text mandatory without violating API semantics. An optional host-sampling route already exists and is the appropriate route for these requests. Changing that route would not repair the assertion.

All tested models retain the original allowlist requests. Every model now must demonstrate the stronger token-level property and usage consistency. Ordinary-token visible text is still required using the served tokenizer, so the correction neither skips a model nor merely permits arbitrary empty responses. HTTP tokenization uses the already-installed `httpx` dependency and existing server API; no new CLI flags or packages are introduced.

## Authorization

Authorized by the operator in checkout `AGENTS.md`, decision 3 (2026-11-05 22:30 UTC): apply PC-02 and PC-03 test-only changes, validate, and create a local plugin commit without pushing. The stock-plugin requirement covers runtime code and interfaces; both remain unchanged.

## Validation and repository rules

Read plugin `AGENTS.md` and `CONTRIBUTING.md`. Proposed changes respect the model-independent/test-only boundary, SPDX headers, and assertions in tests. Before any implementation commit: run the plugin's pinned pre-commit hooks, its host suite with `PYTHONPATH=tests/host-stubs` and pytest8, then the affected live sampling check and full profile. Record the local plugin commit SHA and never push.

Completed locally without API or device calls:

- `git -C <pinned-plugin> apply --check <model>/doc/vllm_integration/PC-02-allowed-token-ids.patch`: passed; did not apply.
- `python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.check_allowed_token_contract_proposal`: passed. This executes the assertion helper extracted from the unapplied patch, verifies the actual pinned tokenizer suppression, and rejects disallowed visible output, missing tokens, and wrong usage. Compact artifact: `readiness_vllm/pc02_host_regression.json`.

Live original-allowlist diagnostic passed before implementation. The patch is now applied; pinned hooks passed after routine Ruff formatting. Hook artifact: `readiness_vllm/plugin_pc02_pc03_hooks.log`. Host-stub suite passed with isolated pytest 8.4.2: 2288 passed, 16 skipped, 5 warnings in 135.86 seconds (`readiness_vllm/plugin_pc02_pc03_host.log`); corrected affected live pytest passed; full sampling passed; Independent stage review returned clean-pass; committed locally as `90f33d2e392eb8bfda6948df9dabe712a724cb1a`. No push. Independent stage review returned clean-pass.

Applied validation command (from plugin checkout): `PYTHONPATH=tests/host-stubs:/tmp/gemma-stage09-pytest8 /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/python_env/bin/python -m pytest tests/ --ignore=tests/tt`. Pytest8 was installed into an isolated target directory; serving dependencies were not modified. Pinned `pre_commit run --files` covered all six modified/new files and passed. `git diff --check` passed. Runtime files remain unchanged; affected live validation passed, with independent stage review clean-pass and local commit `90f33d2e392eb8bfda6948df9dabe712a724cb1a` complete.

Earlier affected live snapshot: 7 passed in 29.78 seconds (`readiness_vllm/final_affected_sampling.log`), covering allowed-token IDs, temperature batch sizes 7/10/19/32/37, and mixed frequency penalties.

Earlier PC-02/03/04 host snapshot: 2299 passed, 16 skipped in 96.79 seconds (`readiness_vllm/plugin_pc02_pc03_pc04_host.log`); superseded by the complete-tree validation above.

Final checkpoint: independent stage review clean-pass; local test-only plugin commit `90f33d2e392eb8bfda6948df9dabe712a724cb1a` on `mvasiljevic/gemma4-stage09-sampling-tests`. Runtime `src/` diff against pinned base is empty. No push.
