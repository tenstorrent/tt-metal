# PC-05: calibrate first-token top-k coverage before sampling

Status: applied; hooks and focused host regressions passed; affected live checks passed; full profile passed; stage review clean-pass; local commit `90f33d2e392eb8bfda6948df9dabe712a724cb1a` complete. Test-only patch `PC-05-topk-cohorts.patch` depends on the expanded PC-03 patch. Runtime remains pinned stock `ca7a18246e4111629ed6cb4a7d1192002749fccf`.

<!-- current-plugin-validation -->
Current validation for the complete PC-02–PC-05 test tree: pinned hooks passed; full host-stub suite **2303 passed, 16 skipped, 5 warnings in 109.37 seconds** with pytest 8.4.2 (`readiness_vllm/plugin_pc02_pc05_host.log`). Affected live checks and all three calibrated top-k cases passed. Each top-k case predeclared two cohorts of 80 draws before sampling (`readiness_vllm/final_topk_calibrated.log`).

Final canonical full sampling passed: **76 passed, 75 skipped, 2 warnings in 512.16 seconds** (`readiness_vllm/final_full/sampling_tests.log`). Passed-test stdout retains all three predeclared cohort plans. Independent stage review returned clean-pass; committed locally as `90f33d2e392eb8bfda6948df9dabe712a724cb1a`. No push.

Runtime `src/`, interfaces and serving defaults are unchanged.
<!-- /current-plugin-validation -->

## 1. Requirement and failure

Top-k sampling must restrict candidates, remain stochastic and isolate greedy rows. After the valid character-to-token correction, `test_topk[19]` still observed ten identical first IDs 19095 (`readiness_vllm/final_affected_extended_sampling.log`). Top-k limits support, not entropy; distinct outputs in a small finite cohort are not guaranteed.

## 2. Measured diagnosis and ownership

`readiness_vllm/topk_distribution_probe.json` measured the selected TT model's unfiltered first-token log probabilities: after temperature 2 and top10 normalization, pmax=0.7962614604553842 and probability of ten identical draws=0.1024606156680557. Native draws selected ID19095 22/30 times; neutral-host controls selected it 25/30 times, with valid alternatives on both. This supports a distribution-sensitive shared-test defect, not a sampler failure. No result is waived or obtained by retrying until a pass. Reproducible diagnostic source: `tests/probe_topk_distribution.py`, retained from the executed `/tmp/probe_gemma_topk.py`; it was formatted but not rerun.

## 3. Existing API and numerical basis

One explicit reference request uses existing `logprobs=20`, `return_tokens_as_token_ids=True`, temperature 1, top_k=-1 and top_p=1, with mathematically neutral zero bias to force the host route. Unique token-ID keys avoid collisions between tokens decoding to the same string. Installed upstream `vllm/v1/sample/sampler.py:80–88` computes raw log probabilities before temperature/penalties; its default mode is raw_logprobs. The unfiltered temperature 1 reference also avoids a transformed-distribution ambiguity. Softmax of the top10 returned scores divided by 2 recovers the target probabilities because a common log normalization constant cancels. No runtime change is required.

## 4. Exact files and behavior

`tests/tt/test_seeding_and_variety.py` obtains the reference before native draws and fixes two cohort sizes. Every wave retains original batches 15/19/32, original prompt, greedy split, stochastic temperature 2/top_k10/top_p1, and max_tokens10. The top10 support includes exact cutoff ties; if the twentieth candidate still ties the cutoff, the test fails explicitly because support is unresolved. Cohorts require membership, at least two unique first IDs each and at least two paired differences. The original first wave from each cohort retains every full-text greedy-equivalence, diversity and pairwise check. No threshold is lowered.

`tests/test_topk_cohort_contract.py` checks skewed-distribution cohort growth, cutoff ties, bounded calibration failure, rejection of outside candidates, constant output and insufficient pair differences. The pure planning function accepts reference probabilities and row count only; observed draws cannot alter the count.

## 5. Statistical contract, limits and other models

For independent first draws with maximum probability m, a constant cohort of N has probability at most m^(N-1). Pair-match probability is at most m, so fewer than two pair differences is bounded by m^N + N(1-m)m^(N-1). The union bound for both constant cohorts plus the pair assertion is fixed below 1e-6/3 per batch-size case, giving 1e-6 across the three first-token cases. This bound does not claim to calibrate the separately retained full-text checks. The plan prints m, cohort size and bound before sampling; no observation-driven retries occur. At most 1024 sampled rows per cohort are permitted; uncalibratable distributions fail explicitly, never skip. Probability estimates assume stable model first-token logits and independent request RNG; native support/frequency controls remain separately inspectable. Test cost increases by the predetermined cohort count, not by whether prior draws pass.

All models use the same logic and APIs; no model branch, capability, environment variable, serving default or runtime interface is added. Tied candidates are admitted only at the measured cutoff, not through a numerical tolerance. A missing top20 reference is a test failure, not an exemption.

## 6. Authorization

Checkout `AGENTS.md` operator decision3 (2026-11-05 22:30 UTC) explicitly permits further justified generic test-only corrections under plugin-change rules. The change affects tests only. Live/full checks and independent clean-pass review completed; local commit is `90f33d2e392eb8bfda6948df9dabe712a724cb1a`. No push.

## 7. Validation

Pinned plugin hooks passed (`readiness_vllm/plugin_pc05_hooks.log`). Focused pytest8 host-stub tests passed 22 tests with 2 warnings in 3.31 seconds (`readiness_vllm/plugin_pc05_host.log`), covering PC02–PC05 helpers. Model hooks passed for the retained diagnostic (`readiness_vllm/topk_distribution_probe_hooks.log`). No new TT/API calls were made by the implementation agent. All three affected top-k live parameterizations passed; the complete sampling profile also passed.

## 8. Status and exact artifacts

Applied and committed locally as `90f33d2e392eb8bfda6948df9dabe712a724cb1a`; runtime unchanged. Diagnosis: `readiness_vllm/topk_distribution_probe.json`, `.log`, and `final_affected_extended_sampling.log`. Implementation: the two test files named above and this patch. The affected live result is recorded below; independent stage review returned clean-pass and the local plugin commit is complete.

Final full host-stub validation including PC-05 passed: **2303 passed, 16 skipped, 5 warnings in 109.37 seconds**, pytest 8.4.2. Artifact: `readiness_vllm/plugin_pc02_pc05_host.log`. PC02→PC03→PC04→PC05 patches also apply cleanly in sequence to copied pinned original files. No runtime changes, device/API calls, or commit occurred during this validation.


Affected live validation passed: **3 passed in 99.96 seconds** (`readiness_vllm/final_topk_calibrated.log`). Each batch-size case predeclared two cohorts of 80 stochastic first draws before sampling: 10 waves/cohort at B15, 8 at B19, 5 at B32. Each measured pmax was 0.7962614604553842 with conservative per-case first-token failure bound 2.912912866206252e-7. Candidate membership, both diversity thresholds and original full-text checks passed. Final canonical sampling passed with `PYTEST_ADDOPTS=-rP` retaining successful cohort plans; no test-selection change was made. Independent stage review returned clean-pass; committed locally as `90f33d2e392eb8bfda6948df9dabe712a724cb1a`. No push.

Final checkpoint: independent stage review clean-pass; local test-only plugin commit `90f33d2e392eb8bfda6948df9dabe712a724cb1a` on `mvasiljevic/gemma4-stage09-sampling-tests`. Runtime `src/` diff against pinned base is empty. No push.
