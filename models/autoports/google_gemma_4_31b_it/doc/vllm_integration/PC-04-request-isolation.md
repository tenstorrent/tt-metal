# PC-04: isolate request parameters within stable sampling routes

Status: applied; host validation and affected live checks passed; final full profile passed; stage review clean-pass; local commit `90f33d2e392eb8bfda6948df9dabe712a724cb1a` complete. Runtime stays at pinned plugin commit `ca7a18246e4111629ed6cb4a7d1192002749fccf`. Patch: `PC-04-request-isolation.patch`.

<!-- current-plugin-validation -->
Current validation for the complete PC-02–PC-05 test tree: pinned hooks passed; full host-stub suite **2303 passed, 16 skipped, 5 warnings in 109.37 seconds** with pytest 8.4.2 (`readiness_vllm/plugin_pc02_pc05_host.log`). Affected live checks and all three calibrated top-k cases passed. Each top-k case predeclared two cohorts of 80 draws before sampling (`readiness_vllm/final_topk_calibrated.log`).

Final canonical full sampling passed: **76 passed, 75 skipped, 2 warnings in 512.16 seconds** (`readiness_vllm/final_full/sampling_tests.log`). Passed-test stdout retains all three predeclared cohort plans. Independent stage review returned clean-pass; committed locally as `90f33d2e392eb8bfda6948df9dabe712a724cb1a`. No push.

Runtime `src/`, interfaces and serving defaults are unchanged.
<!-- /current-plugin-validation -->

## 1. Required behavior and observed failure

Request parameters and seeds must remain isolated across reordered batches. The original ten-config test failed with a temperature-2/seed-42 request emitting ` l l l lL` versus ` l l l額額` (`readiness_vllm/final_full/sampling_tests_pc02_pc03.log`). Its existing docstring acknowledged distinct host/device random streams but constrained only top-k; penalty companions could still change the sampling route.

## 2. Evidence and ownership

`model_runner.py::check_perform_device_sampling` chooses one sampler for a forward and sends penalties to host when the model declares `supports_device_penalties=False`. The model correctly declares that limit. Final server log lines around 2500–2568 show an all-host first wave and a second wave admitting a singleton on device before the remaining nine requests on host. This is a shared-test route assumption, not evidence of request state contamination.

The controlled `readiness_vllm/mixed_isolation_probe.json` preserved all ten configurations: native pairs 1 and 2 differed (Random and Number requests), while all 40 within-host comparison rows and all 40 cross-host comparison rows matched exact token IDs and text. Neutral-bias control waves were entirely host sampled. This proves the route-sensitive failure mechanism without changing server runtime.

## 3. Existing mechanisms and minimal repair

A nonempty `logit_bias={0:0.0}` uses an existing host-only request parameter while adding exactly zero to logits. `SamplingInputBatch.has_active_logitsprocs` recognizes the bias dictionary; upstream `LogitBiasLogitsProcessor.apply` adds its zero tensor. This forces host on every supported mesh. `logprobs=0` alone cannot guarantee that on meshes with 8 or 32 devices, so it is only a diagnostic control.

## 4. Exact files and semantics

`tests/tt/test_request_isolation.py` factors the original ten configurations and unchanged reordered-output assertions into helpers. The original test retains all ten configurations with only neutral bias added; a second native-route test retains the original six penalty-free configurations. All original prompts, caps, temperatures, seeds, top-k/top-p and penalty values remain unchanged. Both retain exact reproducibility assertions. `tests/test_request_isolation_contract.py` checks request preservation, six-config membership, rejection of contamination, actual upstream zero-bias identity, and actual plugin host routing in prefill/decode across 1, 4, 8 and 32 devices.

## 5. Other models and defaults

No model name, skip, xfail, tolerance, capability, environment variable, default or interface is added. Every model runs both tests; native mode follows that model's existing sampler eligibility. Test request count increases from two mixed batches to four. The existing requirement that served default top-k fit the device sampler remains. Bias identity retains the original stochastic distribution but deliberately holds RNG ownership constant for the mixed-penalty assertion.

## 6. Authorization and boundaries

Operator decision 3 in checkout `AGENTS.md` (2026-11-05 22:30 UTC) authorizes further justified generic test-only corrections without a new exception. No plugin runtime or shared interface changes are authorized or included. No push. Live validation and independent clean-pass review completed; local commit is `90f33d2e392eb8bfda6948df9dabe712a724cb1a`.

## 7. Validation

Controlled live diagnosis is retained in `readiness_vllm/mixed_isolation_probe.json` and its per-wave server log byte/line boundaries. Pinned hooks passed (`readiness_vllm/plugin_pc04_hooks.log`). Combined focused pytest 8.4.2 host-stub regressions passed: 18 passed, 2 warnings in 3.62 seconds (`readiness_vllm/plugin_pc04_host.log`). The command from the plugin checkout was `PYTHONPATH=tests/host-stubs:/tmp/gemma-stage09-pytest8 /home/mvasiljevic/tt-metal-gemma4-31b-pipeline/python_env/bin/python -m pytest tests/test_allowed_token_contract.py tests/test_sampling_token_contract.py tests/test_request_isolation_contract.py`. Earlier full host suite passed 2288 tests; runtime remains unchanged. Corrected affected live checks passed; the final full canonical profile also passed.

## 8. Status and artifacts

The source-only change is applied locally. Exact patch and this record are model-owned artifacts; runtime `src/` is unchanged. Independent stage review returned clean-pass; committed locally as `90f33d2e392eb8bfda6948df9dabe712a724cb1a`. No push. Qualitative and performance evidence is recorded separately in the stage report.

Earlier PC-02/03/04 host snapshot: 2299 passed, 16 skipped in 96.79 seconds (`readiness_vllm/plugin_pc02_pc03_pc04_host.log`); superseded by the complete-tree validation above.

Final checkpoint: independent stage review clean-pass; local test-only plugin commit `90f33d2e392eb8bfda6948df9dabe712a724cb1a` on `mvasiljevic/gemma4-stage09-sampling-tests`. Runtime `src/` diff against pinned base is empty. No push.
