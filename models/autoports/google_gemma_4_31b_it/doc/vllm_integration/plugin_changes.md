# Plugin change records

## PC-01: device sampling for mixed intermediate/final prefill rows

Status: **rejected by operator on 2026-11-05; not implemented**.

Operator accepts the stock mixed-chunk host-sampling rule as an explicit compatibility limit. PC-01 is resolved without a plugin change. The proposal below is historical.
Pinned base: `tenstorrent/vllm-tt-plugin@ca7a18246e4111629ed6cb4a7d1192002749fccf`.

1. **Requirement and failure.** Stage09 requires chunked mixed-shape serving with concurrent requests and the canonical device token-out path, without full-logits readback or a Python token-feedback loop. `model_runner.py:2665` unconditionally sets `perform_device_sampling=False` when any prefill row is intermediate. `submit_prefill` then omits sampling parameters, and mixed intermediate/final output extraction requires host logits and calls the host sampler. A model cannot opt out. `readiness_vllm/chunk_sampling_probe.log` records the real pinned input builder and submit call: an 8-token request scheduled for2 tokens together with a final2-token request produces mask `[true,false]`, disables an otherwise eligible device route, and supplies neither sampling parameters nor the mask to the adapter. This is host control-flow evidence, not a served/device result.

   Reproduce from the tt-metal root:
   ```bash
   VLLM_TT_PLUGIN_ROOT=/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09 \
   PYTHONPATH=/home/mvasiljevic/gemma4-31b-pipeline-v3/vllm-tt-plugin-stage09/tests/host-stubs \
   python_env/bin/python models/autoports/google_gemma_4_31b_it/doc/vllm_integration/probe_chunk_sampling.py
   ```

2. **Owner.** Case2, missing shared behavior. `docs/MODEL_EXECUTION.md` explicitly documents host sampling whenever a submitted prefill contains intermediate chunks. This is not a claim that the plugin violates that documentation. Its interface cannot express device sampling only for emitting rows, while suppressing RNG advancement for intermediate rows. The existing generator also needs model-owned continuation, sampling-state and async work; this proposal does not waive those gaps.

3. **Model-side alternatives.** Returning sampled IDs when the plugin expects logits violates its output contract. Fabricating one-hot logits still routes through host sampling, discards requested stochastic semantics/logprobs, and violates the user's path contract. Skipping intermediate sampling alone cannot help final rows, because their sampling parameters were withheld. Disabling chunked prefill, reducing context, or arranging the scheduler never to mix rows would evade the required workload. A compatibility host-logits mode can serve shared tests but cannot replace the required mixed-traffic device path. The generator must implement masked sampler state correctly after the shared interface exists.

4. **Existing mechanisms checked.** `supports_sample_on_device`, `max_device_top_k`, `supports_chunked_prefill`, `supports_prefix_caching`, `supports_device_penalties`, `decode_input_update_contract`, `supports_async_decode`, selective host-readback capabilities, `sample_on_device_mode=all`, and runtime class registration do not override this unconditional prefill rule. The decode-only grammar/deferred-sampling interface does not cover prefill. `empty_slots` supplies destination slots, not an emission mask. `prompt_lens` is a scheduled end, not the complete prompt length, so the adapter cannot reconstruct which rows emit from it.

5. **Proposed shared interface and other models.** Add documented Boolean capability `supports_masked_prefill_sampling` (default false), and keyword `prefill_sample_mask` on `prefill_forward` for opted-in models. The mask is in submitted row order: true means final/emitting, false means intermediate. Supply ordinary sampling parameters on device-eligible opted-in prefill; intermediate rows must neither emit nor advance request RNG/penalty/token state. Keep placeholder suppression in the plugin. Existing models with the default false retain the current route. No model-name branches, environment variables, or additional-config keys. The mask is a shared scheduler fact, not a workaround for missing model behavior.

6. **Authorization at proposal time (superseded by rejection below).** Requested human authorization for exactly the capability `supports_masked_prefill_sampling` and `prefill_forward(..., prefill_sample_mask=...)` semantics in item5. No permission inferred from general bringup authorization. No plugin source changes made.

7. **Planned validation.** Extend the real input-builder reproducer with opted-in and legacy models: all-intermediate, all-final, and mixed batches in both row orders; assert legacy behavior unchanged, opted-in routing/device parameters/mask correct, and intermediate outputs suppressed. Add seeded-stream tests proving intermediate rows do not advance and final rows do not switch sampler ownership. Run plugin host suite and hooks. Then run reduced real-layer serving with concurrent decode/chunk starts/partial tails, sampler-state checks, tracked trace reuse and page growth; finally all-layer full sampling, prefix/qualitative, mixed traffic and both benchmarks. No validation of the proposed implementation is claimed.

8. **Local commit.** PC-01 was rejected and has no implementation commit. Later authorized test-only PC-02–PC-05 changes are recorded below; runtime remains at the pinned base.

### Operator response (2026-11-05)

Not authorized. Keep the pinned stock plugin and accept its documented host sampling for
prefill batches that contain intermediate chunks as a shared-plugin limit; decode and
all-final prefill stay on the device sampler. Serving with `top_k=32` via
`--override-generation-config` is authorized. Details: `AGENTS.md`, "Operator decisions
(stage 9, 2026-11-05)", and `operator_decisions` in `bringup/run_contract.json`.

## PC-02 and PC-03 — authorized test-only corrections

Both are applied to tests under the explicit operator decision below. The runtime plugin and upstream vLLM are unchanged. Full-profile failure evidence and exact proposed patches are in [PC-02-allowed-token-ids.md](PC-02-allowed-token-ids.md) and [PC-03-sampling-token-contracts.md](PC-03-sampling-token-contracts.md). Pinned hooks pass and the corrected host suite reports 2288 passed, 16 skipped; corrected live validation passed and the local commit is complete after independent clean-pass review. Original-request controls and the pinned-HF counterexample are recorded there. This does not reopen or authorize PC-01.

### Operator response to PC-02 and PC-03 (2026-11-05, 22:30 UTC)

Authorized as proposed: apply both test-only patches to the local plugin checkout, validate,
and commit locally; never push. "Keep the pinned stock plugin" covers the runtime only.
Details: `AGENTS.md`, decision 3, and `operator_decisions` in `bringup/run_contract.json`.


## PC-03 extension and PC-04 — token identity and stable sampling routes

The final full profile exposed additional first-character-as-token assertions and a mixed-request test that switched seeded requests between host and device RNG streams. PC-03 now corrects every first-token assertion in its test file without changing thresholds. [PC-04-request-isolation.md](PC-04-request-isolation.md) records controlled live evidence, all eight change-rule items, and the generic test-only repair: retain all ten original configurations under a mathematically neutral host control, and separately exercise the six original penalty-free configurations on the native route. Operator decision 3 authorizes these further justified test-only corrections. No runtime/interface change is included.

Pinned hooks and 18 focused host regressions passed. The final full host-stub suite passed 2299 tests, with 16 skipped and 5 warnings in 96.79 seconds (`readiness_vllm/plugin_pc02_pc03_pc04_host.log`); expanded affected live checks and the full profile passed; independent stage review returned clean-pass and local plugin commit is complete. No push is authorized for this plugin checkout.


## PC-05 — predeclared statistical top-k cohorts

[PC-05-topk-cohorts.md](PC-05-topk-cohorts.md) records a measured 10.25% probability of ten identical valid first draws, the native/host control, and the generic test-only repair. Both first-token uniqueness and paired-difference thresholds remain 2; cohort sizes are fixed from reference logits before sampling, with support checks and explicit bounded failure. The original prompts, recipe, wave shapes and full-text assertions remain. Pinned hooks and 22 focused host tests passed; affected live/full validation passed, stage review returned clean-pass and local commit is complete.

Final full host-stub validation including PC-05 passed: **2303 passed, 16 skipped, 5 warnings in 109.37 seconds**, pytest 8.4.2. Artifact: `readiness_vllm/plugin_pc02_pc05_host.log`. PC02→PC03→PC04→PC05 patches also apply cleanly in sequence to copied pinned original files. No runtime changes, device/API calls, or commit occurred during this validation.

PC-05 affected live validation passed all 3 top-k parameterizations in 99.96 seconds (`readiness_vllm/final_topk_calibrated.log`), with two predeclared cohorts of 80 draws per case. Final full sampling passed. Independent stage review returned clean-pass; committed locally as `90f33d2e392eb8bfda6948df9dabe712a724cb1a`. No push.

Final canonical full sampling passed: **76 passed, 75 skipped, 2 warnings in 512.16 seconds** (`readiness_vllm/final_full/sampling_tests.log`). Passed-test stdout retains all three predeclared cohort plans. Independent stage review returned clean-pass; committed locally as `90f33d2e392eb8bfda6948df9dabe712a724cb1a`. No push.

Final checkpoint: independent stage review clean-pass; local test-only plugin commit `90f33d2e392eb8bfda6948df9dabe712a724cb1a` on `mvasiljevic/gemma4-stage09-sampling-tests`. Runtime `src/` diff against pinned base is empty. No push.
