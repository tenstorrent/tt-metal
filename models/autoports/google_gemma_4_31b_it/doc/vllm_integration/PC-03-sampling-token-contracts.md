# PC-03: token-level temperature variety and frequency-penalty controls

Status: **authorized and applied; affected live checks passed; full profile passed; stage review clean-pass; local commit `90f33d2e392eb8bfda6948df9dabe712a724cb1a` complete**. Patch: `PC-03-sampling-token-contracts.patch`. Pinned plugin base: `ca7a18246e4111629ed6cb4a7d1192002749fccf`. Runtime dependency: PC-02's test-only `RequestConfig.return_token_ids` forwarding. Both patches were applied under operator decision 3 in checkout `AGENTS.md` (2026-11-05 22:30 UTC).

<!-- current-plugin-validation -->
Current validation for the complete PC-02–PC-05 test tree: pinned hooks passed; full host-stub suite **2303 passed, 16 skipped, 5 warnings in 109.37 seconds** with pytest 8.4.2 (`readiness_vllm/plugin_pc02_pc05_host.log`). Affected live checks and all three calibrated top-k cases passed. Each top-k case predeclared two cohorts of 80 draws before sampling (`readiness_vllm/final_topk_calibrated.log`).

Final canonical full sampling passed: **76 passed, 75 skipped, 2 warnings in 512.16 seconds** (`readiness_vllm/final_full/sampling_tests.log`). Passed-test stdout retains all three predeclared cohort plans. Independent stage review returned clean-pass; committed locally as `90f33d2e392eb8bfda6948df9dabe712a724cb1a`. No push.

Runtime `src/`, interfaces and serving defaults are unchanged.
<!-- /current-plugin-validation -->

## 1. Requirement and failure

The full profile must test actual first-token diversity and per-request frequency-penalty behavior. Two failures in `readiness_vllm/sampling_tests.log` do not inspect those properties correctly:

- `test_temperature_varied_in_batch[7]` calls string slicing `x[:1]` a first-token check. It failed because seven outputs began with a space, after its complete-output variety assertion had passed. A leading character can be shared by distinct generated tokens.
- `test_frequency_penalty_mixed_batch` requires the first unpenalized output to contain strictly more character `a` occurrences than the first penalized output. It failed with counts1/1 before reaching its determinism checks. Frequency penalty modifies generated-token counts, not character counts, and need not reduce an arbitrary character.

Minimal reproducers retain original requests: seven concurrent raw `Random letter: ` requests, temperature2, cap10; and32 alternating raw `a a a a a a a a a` requests, greedy, frequency penalty0/2, cap15. Capture raw output IDs and text. The completed `sampling_failure_probe.json` records three temperature waves with4/4/5 distinct first IDs and all32 mixed penalty rows exactly matching corresponding homogeneous, solo and native controls. That does not retroactively recover the original unrecorded first IDs.

`raw_prompt_hf_controls.json` supplies an independent full60 pinned-HF counterexample to the frequency assertion. Correct output-count penalty2 changes the generated sequence, but both penalized and unpenalized outputs have exactly one character `a`. The first unchanged token contributes it. The CPU reference explicitly subtracts penalty times generated-token counts; prompt tokens are excluded. The raw instruct-model outputs are themselves repetitive; that separate formatting observation does not exempt model quality gates.

## 2. Owner

These are shared test defects. `tests/tt/utils.py::send_request` returns decoded text; first-token identity therefore cannot be inferred from `x[:1]`. The plugin's host path (`src/vllm_tt_plugin/model_runner.py`, host SamplingMetadata assembly) supplies generated output history to the upstream frequency sampler. The intended properties remain mandatory and are tested more directly in the proposal.

## 3. Model-side alternative

Forcing diverse leading characters or forcing fewer occurrences of one letter would alter valid model/sampling behavior to satisfy a tokenizer-dependent assertion. Changing model weights, temperature interpretation, penalty arithmetic or detokenization is not a valid repair. No adapter gap was found in the targeted controls: all32 rows preserve their respective policy exactly. That evidence does not justify skipping either test.

## 4. Existing mechanisms and exact change

Existing raw completion `return_token_ids` identifies actual tokens; PC-02 exposes it through the test request helper. `logprobs=0` is an existing diagnostic request option; on this TP4 setup it holds frequency controls on the host route used by the original mixed penalty batch. No new capability, additional-config key, environment variable, registration or model hook applies or is introduced.

Exact proposed files:

- `tests/tt/test_seeding_and_variety.py`: preserve temperatures, prompts, batch sizes, token caps and both diversity thresholds; compare actual first generated IDs instead of first characters.
- `tests/tt/test_tt_penalties.py`: preserve original alternating requests; require every mixed row's IDs to equal same-policy homogeneous and solo controls, and require a real token-sequence change between penalty0 and2. Replace only the invalid strict character-count condition. The checks are stricter about all-row isolation and actual penalty effect.
- `tests/test_sampling_token_contract.py`: host counterexamples for distinct tokens sharing a character, constant tokens still failing the same diversity threshold, changed token sequences with unchanged character counts, contaminated slots, and missing penalty effect.

## 5. Effect on other models

All models retain the original prompts, sampling parameters, caps and diversity threshold2. No model-specific branch, skip, xfail or tolerance is added. The frequency test gains two homogeneous batches and two solo controls. Exact all-row greedy token equality is required by this proposed test; any cross-route discrepancy discovered on other supported models must be investigated rather than hidden. Full-profile runtime increases modestly from those additional requests.

## 6. Authorization

The operator explicitly authorized PC-02 and PC-03 in checkout `AGENTS.md`, decision 3 (2026-11-05 22:30 UTC), including test-only application, validation and a local commit without pushing. Runtime and shared interfaces remain unchanged.

## 7. Validation

Read plugin `AGENTS.md` and `CONTRIBUTING.md`; production source remains untouched. Completed without new API/device calls:

- `git -C <pinned-plugin> apply --check <model>/doc/vllm_integration/PC-03-sampling-token-contracts.patch`: passed, unapplied.
- `python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.check_sampling_token_contract_proposal`: passed. Executes the proposed helpers extracted from the patch against retained serving/HF evidence and injected incorrect controls. Artifact: `readiness_vllm/pc03_host_regression.json`.

PC-02 then PC-03 have been applied. Pinned pre-commit hooks passed after routine Ruff formatting (`readiness_vllm/plugin_pc02_pc03_hooks.log`); the isolated pytest 8.4.2 host-stub suite passed: 2288 passed, 16 skipped, 5 warnings in 135.86 seconds (`readiness_vllm/plugin_pc02_pc03_host.log`). Affected live tests passed across all original parameterizations. The final full sampling profile passed. Token IDs and frequency controls are retained in the diagnostic artifacts. The diversity threshold is unchanged; PC-05 calibrates the cohort size. Independent stage review returned clean-pass; committed locally as `90f33d2e392eb8bfda6948df9dabe712a724cb1a`. No push.

## 8. Status

Applied to the local plugin tests only; affected live checks and the full profile passed. Independent stage review returned clean-pass; committed locally as `90f33d2e392eb8bfda6948df9dabe712a724cb1a`. No push. The independent qualitative investigation is closed in the stage quality record; these test corrections do not alter model output quality.

Applied validation command (from plugin checkout): `PYTHONPATH=tests/host-stubs:/tmp/gemma-stage09-pytest8 /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/python_env/bin/python -m pytest tests/ --ignore=tests/tt`. Pytest8 was installed into an isolated target directory; serving dependencies were not modified. Pinned `pre_commit run --files` covered all six modified/new files and passed. `git diff --check` passed. Runtime files remain unchanged; affected live validation passed, with independent stage review clean-pass and local commit `90f33d2e392eb8bfda6948df9dabe712a724cb1a` complete.

Earlier affected live snapshot: 7 passed in 29.78 seconds (`readiness_vllm/final_affected_sampling.log`), covering allowed-token IDs, temperature batch sizes 7/10/19/32/37, and mixed frequency penalties.


## Extension: all first-token assertions

The subsequent full profile (`readiness_vllm/final_full/sampling_tests_pc02_pc03.log`) finished with 2 failed, 73 passed and 75 skipped in 439.42 seconds. `test_topk[19]` passed full-output variety but failed its supposed first-token check because all ten first **characters** were `R`. The same defect existed in seeded variety, temperature between batches, and top-k within/between batches. These now all use `_first_generated_token_ids` and full-response token IDs. All original prompts, generation caps, seeds, batch sizes, full-output checks and token-variety thresholds are unchanged. No skip, model branch or relaxation is introduced. The independent mixed-request routing failure is PC-04.

The generic extension is authorized by operator decision 3. The refreshed PC-03 patch includes every occurrence in this file. Added pairwise host regression proves distinct first token IDs can share `R`, while insufficient actual token variation still fails the unchanged threshold. Pinned hooks passed (`readiness_vllm/plugin_pc03_extension_hooks.log`); focused host tests passed 8 tests in 2.84 seconds (`readiness_vllm/plugin_pc03_extension_host.log`). The earlier seven live passes cover the earlier PC-03 scope only; the extended live scope subsequently passed except the independently diagnosed PC-05 defect; see current validation above.

Earlier PC-02/03/04 host snapshot: 2299 passed, 16 skipped in 96.79 seconds (`readiness_vllm/plugin_pc02_pc03_pc04_host.log`); superseded by the complete-tree validation above.

Final checkpoint: independent stage review clean-pass; local test-only plugin commit `90f33d2e392eb8bfda6948df9dabe712a724cb1a` on `mvasiljevic/gemma4-stage09-sampling-tests`. Runtime `src/` diff against pinned base is empty. No push.
