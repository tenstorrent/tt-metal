# Proposed serving plugin changes

## Entry 1: Authoritative original prompt lengths for prefill penalty history

**Status: implemented locally at `215effdee4e473f8c50af608d9812fafc56fd48b`; host and focused device regressions pass. Reduced serving checks and integrated64 full-model tracked correctness/output-quality acceptance pass. Strict forced-replay token equality remains failed; final performance/full P2 and independent stage review remain pending.**

Sections describing proposed semantics and planned regressions retain the historical approval design; current implementation and validation results appear below.

Pinned base: `tenstorrent/vllm-tt-plugin` commit `042a94863ad27d88207bcf71abdf67122c6f4c27` at `/home/ttuser/ornith-9b/dependencies/vllm-tt-plugin`. Investigation date: 2026-11-13.

### 1. Requirement and failure

A resumed request can replay its prompt plus previously generated output through prefill. Its next sampled token must apply presence/frequency penalties to **generated output only**, while repetition penalties consider both prompt and output. This must work on the device and on the host without changing the client request.

The pinned plugin supplies replay tokens and scheduled chunk ends, but not original prompt lengths or separate histories, to `prefill_forward`. Ornith therefore classifies the replay as prompt and clears output history before sampling. The pinned host-prefill path independently constructs empty output history. Both can omit required presence/frequency penalties.

Minimal distinguishing example: prefill `[7,8]`, sample `9`, release slot 0, then prefill `[7,8,9]` from zero into the reused slot. A fresh request after finish and a resumed request after preemption produce identical model-visible calls. The correct output histories are respectively `[]` and `[9]`. With logits `{9:2.0,10:1.5}`, presence penalty `0.75`, frequency penalty `0.5`, repetition penalty `1.0`, and temperature zero, the correct sampled tokens differ: fresh `9`, resumed `10`.

Reproduce without importing TTNN or launching a server:

```bash
python_env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/tests/vllm_resume_contract_probe.py \
  --plugin /home/ttuser/ornith-9b/dependencies/vllm-tt-plugin \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/resume_contract_probe.json
```

The probe extracts the actual pinned prefill submission/release methods and sampling dataclass with AST. Its recorder traces establish information loss; illustrative logits establish why the missing distinction matters. It does not execute the scheduler or actual model. The full information-flow proof and source references are in [resume_contract_investigation.md](resume_contract_investigation.md); executable evidence is in [resume_contract_probe.json](resume_contract_probe.json).

### 2. Owner and proposed semantics

**Plugin defect / shared information delivery:** the plugin owns the authoritative request prompt/output boundary. `docs/MODEL_EXECUTION.md:54-55` explicitly permits generated-token replay during resume. `model_runner.py:2394-2480` knows original prompt counts and resumed status; `:2679-2694` builds separate histories only for decode; `:4094-4148` omits the boundary at prefill submission. The existing standard penalty semantics distinguish prompt and generated output. A model cannot recover a distinction that the entire callback transcript omits.

Proposed exact opt-in interface:

- Add boolean `model_capabilities['supports_original_prompt_lens']`, default **False** when absent. Reject a present non-boolean value consistently with existing capability validation. Declare it on both Ornith's class and instance as applicable.
- For opted-in models, every real `prefill_forward` call receives `original_prompt_lens`, a CPU list of Python integers, exactly one per submitted prefill row, in the same order as `tokens`, `prompt_lens`, and `empty_slots`.
- Each value is the request's **original prompt token count**, taken from authoritative input-batch/request metadata using the exact row selection used to build that prefill. Capture it in the per-submission payload rather than rereading mutable batch layout at submission time. In the front-packed builder the source is `input_batch.num_prompt_tokens[req_indices]`; any lane-specific construction must preserve that lane's selected row order.
- The value stays constant across original chunks, prefix-cache hits, and resumed chunks for that request. It is not the chunk length, computed-token count, scheduled end, or current total history length. On intermediate prompt chunks it may exceed `prompt_lens[i]`.
- For the available prefix ending at `end=prompt_lens[i]`, split prompt history as `tokens[i,:min(original_prompt_lens[i],end)]` and output history as `tokens[i,original_prompt_lens[i]:end]` when the original length is below the end. Do not include row padding or unscheduled future tokens. Final sampled replay has the complete committed history through `end`.
- Intermediate chunks continue using existing output suppression and must not advance the device sampler. The new argument carries metadata only; no sampler ownership change is proposed.
- Omit the new keyword entirely for non-opted-in models. Do not infer opt-in from model names, method signatures, extra attributes, or environment variables. No new config/environment option, request ID, resume flag, or lifecycle hook is proposed.
- Warmup callbacks retain their signatures; models may supply their own original lengths for any internal synthetic prefill calls. Decode history arguments and reload commands retain their semantics.

**Model implementation (approved and implemented):** Ornith accepts the argument and uses its existing separate-history reset mechanism before final device-prefill sampling. Direct canonical callers lacking this serving argument retain their established standalone contract; serving opt-in calls must not silently replace missing authoritative metadata with guesses. The adapter must validate row count, integer type, and sane nonnegative original lengths without incorrectly requiring original length to be at most an intermediate chunk end.

**Related plugin-owned host correction:** independently of model opt-in, construct host-prefill penalty `SamplingMetadata` using the authoritative prompt/output split for that submission, replacing the current empty output history at `model_runner.py:4371-4396,4418-4437`. Preserve row selection, `-1` history padding/vocabulary sentinel conversion, intermediate-chunk suppression, and inactive-row handling. This corrects resumed-prefill semantics for existing host callers; it is explicitly included in the requested authorization because it changes their default erroneous behavior. It must not rely on `supports_original_prompt_lens`, since host sampling already belongs to the plugin.

This is a concrete proposed design, not a claim that a capability is the only possible implementation. A different plugin orchestration might reuse existing calls, but it would still need compatibility and semantic justification. A new capability is proposed here to preserve the call signatures of all existing adapters while delivering the smallest sufficient boundary metadata.

### 3. Model-side alternative and why it is incorrect

Ornith already accepts separate histories in `SamplingSlotState.reset_history`. Extending its prefill to use that existing API is necessary but not sufficient: without authoritative boundary metadata the histories cannot be reconstructed.

Retaining slot state, storing old prompt lengths, matching replay-token prefixes, or caching all past output sequences fails the two-world example. Finish and preemption both call `release_request(slot)`, and slots/pages can be reused. The complete model-visible transcript can be identical while the correct output history differs. A subsequent decode history update arrives after the wrongly sampled prefill token. Stack inspection or a model-owned runner monkeypatch is not implementation within the documented interface.

Routing all penalties or resumed work to host would bypass device sampling rather than implement it, and the pinned host-prefill path currently has the same semantic omission. Declaring penalties unsupported would hide an already implemented and required model feature.

### 4. Existing mechanisms checked

- `supports_sample_on_device`, `supports_device_penalties`, `max_device_top_k`: sampling eligibility/routing; no history boundary.
- `supports_prefix_caching`, `supports_chunked_prefill`: cache/chunk support; disabling either does not implement preemption replay.
- `supports_async_decode` and `decode_input_update_contract=1`: scheduling and decode update ownership; no prefill metadata delivery.
- `empty_slots`, `slot_remap`, `note_state_slots_moved`, `release_request`: physical ownership only; no persistent request ID or finish/preemption distinction.
- Decode `prompt_tokens`/`output_tokens`: correct existing vocabulary but delivered too late for the prefill sample.
- `supports_stage_inputs`: optional request-owned vLLM-Omni payload, not automatic scheduler metadata; ordinary text requests have no payload. Requiring clients to add one changes the request contract.
- M-RoPE and multimodal metadata: unrelated meanings; ordinary text deltas/payloads cannot encode this distinction.
- `supports_stage_outputs` and block-output admission: genuine different output contracts whose preemption restrictions cannot legitimately be claimed merely to avoid the bug.
- Existing `tt.sample_on_device_mode`, `tt.seeded_sampling_policy`, `tt.always_compat_sampling`: choose sampling paths; do not provide missing model metadata and do not repair current host history construction.
- Trace/warmup/fabric/layout configuration and memory/concurrency limits: resource/execution choices; no general resume correctness.
- `TT_MODEL_CLASS_OVERRIDES`, `EXTRA_MODELS_DIR`, built-in registration: select adapters, not request history. Existing runtime registration remains sufficient for Ornith; no built-in autoport registration is proposed.
- No documented environment variable or `--additional-config` key supplies original prompt lengths to prefill. No new one is proposed.

### 5. Effect on other models and callers

For models without the new opt-in, device-prefill kwargs and behavior remain unchanged. Existing `tt_transformers` and Llama 3 70B Galaxy adapters are examples of callers whose old signatures must remain usable. They will not be silently opted in or claimed fixed for device replay by this proposal. Ornith is the only planned new opt-in in this bringup; any later opt-in must implement and validate the semantics.

The host-history correction applies to ordinary host-prefill sampling for all models, including requests routed to host because of logprobs or unsupported device controls. Fresh requests still have empty output history and retain the existing result. Resumed requests receive their committed generated-token history, which can intentionally change their sampled token to satisfy penalties. Intermediate outputs remain discarded. Lane-DP and ordinary front-packed row selection must both be tested before a shared change is accepted; no architecture-specific branches are allowed.

Update plugin `docs/MODEL_CAPABILITIES.md` and `docs/MODEL_EXECUTION.md` alongside implementation. Keep the plugin change and model opt-in paired and documented; do not push either repository.

### 6. Authorization

**Approved by the owner through the multigoal runner in this thread:** “Entry 1 in doc/vllm_integration/plugin_changes.md is approved as proposed (supports_original_prompt_lens and the shared host-prefill history correction). Implement it as a local commit in the vllm-tt-plugin checkout only; do not push. Then continue the stage.” Authorization covers the exact two items below. Implementation is locally committed; completed validation and remaining limits are recorded below.

Exact requested items:

1. Add `model_capabilities['supports_original_prompt_lens']` (boolean, absent default False), forwarding the per-row `original_prompt_lens` keyword to prefill only for opted-in models, with the semantics above; opt Ornith into it.
2. Correct shared host-prefill penalty histories from authoritative prompt/output metadata, including resumed requests on models that do not opt into the new model-facing keyword.

The applicable policy is [plugin-changes.md](/home/ttuser/ornith-9b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/vllm-integration/references/plugin-changes.md), which requires: “Make such a change only after a human authorizes that exact item.” Its list explicitly includes new model capabilities and changes of default behavior for models that do not opt in. Plugin `AGENTS.md` separately requires a specific human request for a new capability. A general bringup/fix authorization does not satisfy either requirement. The owner approval quoted above resolves that authorization requirement.

### 7. Validation

**Existing proposal evidence:** the host probe ran successfully on the exact pinned SHA and recorded four checks in [resume_contract_probe.json](resume_contract_probe.json): identical complete callback transcript, omitted internal histories, differing required penalty outputs, and empty normalization of absent history. Status `contract_gap_reproduced` intentionally means a bug is reproduced, not that serving passes.

**Original regression plan (retained as the approval record; results below):**

- Opted-in fresh versus resumed prefill: identical tokens but original lengths `3` and `2` arrive in the correct rows; verify output split `[]` versus `[9]` and corresponding presence/frequency behavior.
- Absent/False opt-in receives no new kwarg; legacy strict signatures remain callable. Present non-boolean values fail clearly.
- Capture original lengths with submission rows and prove a later input-batch reorder cannot change their association. Exercise nontrivial `req_indices`, `empty_slots`, multirow unequal prompt lengths, and lane-selected rows.
- Original chunk, continuation with original length beyond the current end, replay chunk before/crossing/after the original boundary, and final replay. Include duplicated token histories with distinct original boundaries and reused physical slots/pages.
- Host sampler metadata and token results: presence-only, frequency-only, their combination, repetition-only, and neutral penalties; compare with authoritative vLLM prompt/output history. Verify fresh behavior is unchanged and resumed behavior is repaired.
- Mixed intermediate/final prefill retains discarded outputs and no unintended device sampling/RNG advance; padded histories do not count sentinels or future tokens.

Run the plugin host suite using its `AGENTS.md` recipe and supported environment, plus required pre-commit checks before a local plugin commit. Record failures honestly. Run focused model host tests for boundary translation/reset, then smallest authorized hardware tests exercising actual forced preemption/resume, fresh history collision, slot reuse/remap, chunked replay, and both host/device penalties. Re-run the affected full-model Stage 9 serving checks after the focused checks pass. Implementation, host, focused device, reduced serving and integrated64 full-model tracked acceptance results are now recorded below. These supersede the pre-implementation validation status; strict forced-replay equality and full-stage completion are not claimed.

### 8. Status and revision record

**implemented, authorized** — local plugin commit `215effdee4e473f8c50af608d9812fafc56fd48b` atop allocator repair `a09b0ef5f600305c738da0482bb76fee0e2e776a`; no push. Base SHA: `042a94863ad27d88207bcf71abdf67122c6f4c27`.

The record below includes the local plugin commit, failing-before/passing-after regressions, plugin host results, paired model changes and affected reduced/full-model serving results. Implementation does not establish stage completion or exact numerical replay equivalence.

## Entry 2: Preserve continuing prefill state-slot ownership

1. **Requirement and failure.** A continuing chunk must address its existing recurrent state. Diagnostic6 failed on B's continuation at896 after `_alloc_prefill_state_slots` assigned its old slot1 to new C and moved B to2 without a gather. Minimal host case: capacity3, live ownership `{A:0,B:1}`, incoming `[C,B]`: current result `[1,2]`, required `[2,1]`. See async_continuation_investigation.md and diagnostic6_async/server.log.
2. **Owner.** Plugin defect against docs/MODEL_LIFECYCLE.md request-state ownership and existing test_state_slots.py assertion that re-prefilled live requests keep their slots. Prefill has no gather argument, so the allocator must preserve existing owners.
3. **Model alternative.** Guessing which other slot holds the prefix is ambiguous for identical histories and bypasses the plugin's authoritative empty_slots mapping. Relaxing the exact-prior-end gate would corrupt state. The generator already honors whole-pool decode remaps.
4. **Existing mechanisms.** Use the current `_req_state_slot`, empty_slots and release lifecycle. No new capability, environment/config option, registration or callback is needed. Decode slot_remap is unrelated to assigning slots to prefill before any gather.
5. **Other models.** All front-packed models retain their owned state during continuation; fresh allocation still prefers a free decode row. Existing release/preemption and lane-DP contracts are unchanged. Stateless models see equivalent tokens with correct slot bookkeeping.
6. **Authorization.** Adds no shared interface or new default policy; repairs existing documented ownership. Covered by the user's instruction to fix failures under plugin-change rules. Entry1 was separately authorized by the owner and implemented at `215effd`.
7. **Validation plan.** Add host regressions with a simulated state array for newcomer-before-continuation, reversed continuing rows, and release/re-admission. Show failures before allocator repair and passes after it; run plugin host suite and pre-commit, then repeat the reduced async mixed workload after bounded device health verification. Record results here before claiming implemented completion.
8. **Status.** Implemented: local commit `a09b0ef5f600305c738da0482bb76fee0e2e776a`, base `042a94863ad27d88207bcf71abdf67122c6f4c27`. No push.

**Validation results.** New state-array regressions failed before the repair: 3 failures, 16 passes (`slot_regression_before.log`). After repair the plugin host suite passed: 2322 passed, 1 skipped, 4 warnings (`slot_plugin_host_tests.log`), using host-stub TTNN and pytest8.4.2 per plugin instructions. Staged pre-commit passed (`slot_plugin_precommit.log`). The post-crash TP4 trivial-op health check passed (`post_crash_mesh_health.log`). Reduced async serving in `diagnostic7_slots/` completed device mixed requests, explicit host-logprobs requests, then reordered device mixed requests on the same server. All 12 device responses matched the synchronous diagnostic4 control token-for-token. Actual continuation offsets include768,1024,1792,2048 with concurrent decode counters increasing between chunks. Six original trace captures persisted; shutdown counters report captures6, releases0 before teardown. See `events_summary.json` and `server.log`. The server was stopped only when running/waiting metrics were zero; the runner exited0 and device/process audits were empty. This is reduced-layer regression evidence, not final full-model readiness. Entry1 now has separate explicit owner authorization.

### Entry 1 implementation evidence

Owner-approved local commit `215effdee4e473f8c50af608d9812fafc56fd48b` implements the exact capability and shared host-history correction. The new suite reproduces 18 callback/metadata failures before the fix, and actual host sampling reproduces 3 incorrect resumed penalty results. After: 26 focused tests pass; full plugin stub suite 2348 passed, 1 skipped, 4 warnings; pre-commit passed. Logs: `entry1_before.log`, `entry1_before_penalties.log`, `entry1_targeted.log`, `entry1_host_suite.log`, `entry1_precommit.log`. Canonical generator and adapter consume original boundaries through existing history/seed reset APIs. Focused device validation passed14 boundary/history checks (`original_boundary_device.json`); the combined model host suite passed37 (`entry1_all_model_host.log`), and the adversarial decode battery passed (`adapter_decode_after_history.json`). Reduced mixed serving passed all nine checks for15 requests in diagnostic8 and diagnostic9. Forced-preemption requests completed, but all three strict trajectory comparisons remain failed. The canonical served-context control reproduces the first seeded difference with identical sampler state and different raw logits; the later `replay_greedy_follow50.json` also reproduces both served greedy branches through50 outputs and their first divergence at index49 with identical preceding tokens and sampler state. The aligned151 snapshot now passes structural checks and exact host/device replay reconstruction; KV differences start at the decode boundary129. See `replay_model_state151_interpretation.md`; strict parity is still failed and no rounding tolerance is inferred. See `diagnostic9_pages/preemption_analysis.md`.

**Final64 full-model tracked follow-up.** `readiness_vllm/sampling64_tracked/acceptance_index.json` records completed correctness/output-quality acceptance on the frozen full32-layer TP4/B32/context262144 source and unchanged plugin HEAD `2a1504554aa7babb1d426dfbca8c7db8c0a795ef`. Full sampling passed67/skipped75 on first EngineCore1265789; `completed_sampling_reuse.json` preserves that origin and the qualitative outputs separately from retry EngineCore1293243. The first mixed attempt failed concurrent-decode coverage only; its failure is retained. Retry mixed and lifecycle each pass10 checks, chunked-prefill passes3 tests, tracked reasoning P2 records644 device/0 host calls, P3 critical checks pass (render_c07 advisory retained), and P5 passes9/9 with all exact answers read. Qualitative verdict passes after all12 served texts and12 fresh TT controls were read, with HF metadata reuse verified. The matched standalone logit verdict passes27 positions across9 groups, plus12 HTTP comparison groups; tracked benchmark/prefill routing records261 device/0 host and238 steady calls. See the named reports in `sampling64_tracked/`, including `p5_text_review.json`, `qualitative_observations.json`, `logit_verdict.json`, and `runtime_audit.json`. Retry flow exits0 with normal shutdown. These are full-model integration regressions, not a new full-model forced-preemption parity test: all three reduced strict trajectory failures remain failed. Final tracking-off performance/full P2 and independent stage review remain separate pending gates.

## Entry 3: Replace persistent block-table rows on immediate preemption resume

1. **Requirement and failure.** A resumed request must use the replacement physical KV blocks allocated by the scheduler. Upstream development API `pause(mode=keep,clear_cache=false)`, `reset_prefix_cache(reset_running_requests=true)`, `resume` can preempt/re-admit a request while it still occupies the persistent runner row. `diagnostic8_history/preemption.json` records successful resets and complete responses, but token divergence; source inspection finds cached block IDs replaced while the persistent row appends replacements to freed old blocks. Minimal host case: persistent row R has `[1,2]`, resumed metadata replaces it with `[7,8]`; required row `[7,8]`, current row `[1,2,7,8]`. A later ordinary growth `[9]` must produce `[7,8,9]`.
2. **Owner.** Plugin defect in `model_runner.TTModelRunner._update_states` existing-row branch. `input_batch.apply_cached_req_state_update` explicitly documents/replaces blocks on resume; upstream `scheduler.reset_prefix_cache(reset_running_requests=True)` clears previous scheduled IDs so immediate resumes arrive as resumed cached requests. Lane-DP `input_batch` already handles resumed requests separately. Model execution/page-table contracts require authoritative physical pages; the front-packed persistent table disagrees with the scheduler and its own cached request state.
3. **Model-side alternative.** The generator cannot recover the replacement table from an input containing the old prefix plus appended pages: ordinary page growth has the same representation. Guessing/using trailing pages would corrupt normal requests and violate vLLM ownership. The adapter already uses the supplied table and validates its geometry. No correct model-only repair exists.
4. **Existing mechanisms.** Use existing `resumed_req_ids`, `CachedRequestState.block_ids`, persistent row replacement/removal-and-readdition, and decode-layout invalidation. `supports_original_prompt_lens` supplies history boundaries only; prefix/chunk/async capability flags and sampling routing do not repair scheduler page ownership. Existing registration and TT config likewise do not identify a replacement row. No new hook/config/flag is needed.
5. **Effect on other models.** Applies to every front-packed model resumed while still present in its input batch. Ordinary continuation appends only incremental blocks as before; removed/re-added resumes already use fresh rows. Lane-DP's separate existing behavior remains. Peer request rows must remain intact. No model-name branch.
6. **Authorization.** No new shared interface, opt-in, default policy, or feature is added. This repairs existing documented page ownership under the task's authorization to fix serving failures. Entry1's separate explicit interface approval is not being extended to a new interface.
7. **Validation plan.** Reproduce using actual `_update_states` and real host InputBatch/block table: immediate resume, repeated resume, later incremental growth, and unaffected peer. Record failing before, passing after, full stub suite and pre-commit. Repeat same reduced real upstream forced-preemption workload and mixed-shape sequence, inspect actual replacement pages and preserved sampler state. Any numerical replay difference remains separately evaluated against raw logits, not silently waived.
8. **Status: implemented locally.** Commit `2a1504554aa7babb1d426dfbca8c7db8c0a795ef` atop Entry1 `215effdee4e473f8c50af608d9812fafc56fd48b`; original pinned base `042a94863ad27d88207bcf71abdf67122c6f4c27`. No push. The eight-part proposal above was recorded before the runtime edit. Existing `add_row` replaces a retained resumed row, and the layout-change flag requests decode input reload; ordinary growth remains append-only.

**Validation results.** Actual `_update_states` with real host InputBatch and block tables reproduced2 failures/2 passes before (`entry3_before.log`). Focused tests passed96 (`entry3_after.log`); full host-stub suite passed2354, skipped1, warnings4 (`entry3_host_suite.log`); pre-commit passed (`entry3_precommit.log`). Coverage includes retained and removed rows, repeated resume, later block growth, one/two cache groups, unaffected peers, pending output application and forced stale-suffix dropping.

Reduced diagnostic9 repeats the real upstream forced-preemption workload. Model-facing rows now visibly replace, with replay start0/end151/original129. Baseline outputs are unchanged; all requests complete100 tokens and all resets succeed. Strict baseline/resumed equality still fails: greedy device/host first differ at index49 (53 versus498); seeded device at index22 (264 versus51). Old/new forced host scores first change at index32/absolute decode position160, consistent with the repaired first-growth alias. The original canonical control reproduces the seeded first divergence. The later `replay_greedy_follow50.json` reproduces both served greedy branches through50 outputs, including index49 (53 versus498), with identical preceding49 tokens and sampler state. The aligned151 comparison now confirms matching prefix/positions/pages, exact host/device replay states and preserved sampled inactive regions; original-prompt KV rows0..128 are exact and differences start at decode row129. This identifies a canonical computation-route difference without proving harmless rounding or exact trajectory parity; see `replay_model_state151_interpretation.md`. Mixed serving passes all nine checks for15 requests. See `diagnostic9_pages/preemption_analysis.json` and `.md` for exact evidence and limitations. This is a page-ownership repair, not a full-model or exact-replay acceptance claim.
