# Proposed shared prefill sampling mask

Status: **authorized; implementation committed; host and affected hardware serving validation passed; final stage acceptance remains in progress**. Owner approval received2026-11-16 via multigoal runner; no upstream vLLM change is proposed. Base plugin: `tenstorrent/vllm-tt-plugin@042a94863ad27d88207bcf71abdf67122c6f4c27`; upstream vLLM `v0.31.0`.

## Requirement, observed failure and reproducer

The goal requires canonical on-device sampling without full-logits readback or host token feedback in the optimized serving path, including a mixed-shape workload with chunk continuations while other requests decode. The adapter follows the plugin contract, but the plugin forces intermediate prefill chunks onto its host path.

Reproduce with the unchanged plugin:

```bash
bash models/autoports/qwen_qwen3_8_flash_next/doc/vllm_integration/run_reduced_server.sh \
  models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/reduced_control_v3
# In the configured serving environment, against that live server:
python models/autoports/qwen_qwen3_8_flash_next/tests/check_vllm_mixed.py \
  --server-log models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/reduced_control_v3/server.log \
  --output models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/reduced_control_v3/mixed.json --reduced
```

The saved run completed all ten requests with input lengths129/130/131/257/513/515,96 output tokens and concurrency2. Actual scheduler starts were0/256/512, with live concurrent decode. Each capture signature appeared once and no trace retired. Seven intermediate prefill forwards had `sampling_mode=host`; all decode steps and final device-eligible prefill steps remained device sampled. The result correctly reports `device_sampling_only=false` and overall `passed=false`. These seven forwards are forced intermediate fallback, **not** optional host-feature requests.

Artifacts: `readiness_vllm/reduced_control_v3/mixed.json` and its `server.log`; `tests/check_vllm_mixed.py`. Source tracing shows host logits are read by the generator. These seven calls are all-intermediate batches: the plugin bypasses its host sampler, creates placeholders and suppresses them. Mixed final/intermediate batches do invoke host sampling. The mandatory full-logits transfer remains a violation of the requested optimized path. No device-level profiler was used.

## Owner

This is **missing shared behavior**, not a violation of the pinned plugin contract. `docs/MODEL_EXECUTION.md:57–61` explicitly requires host logits or sampled token IDs, states there is no prefill readback hook, and selects host sampling when intermediate chunks are present. `docs/SCHEDULING.md:216–221` documents the same behavior. `src/vllm_tt_plugin/model_runner.py:2665–2669` applies the forced host route; output suppression occurs at3851–3868; the all-intermediate host-sampler bypass is at4311–4323. Cloned host generators preserve request RNG for discarded intermediate draws.

The plugin possesses per-row finality (`intermediate_prefill_mask`) but does not pass it to the adapter. This affects any stateful model that must execute intermediate cache updates without sampling or host-logit transfer.

## Model-side alternatives

Finality cannot be recovered from existing inputs. At `model_runner.py:2444–2474`, `prompt_lens` is the scheduled chunk end and `tokens` is truncated to the maximum scheduled end. A final64-token request and the first64-token chunk of a96-token request can have identical adapter inputs. Mixed rows also pad to another row's maximum end. Tokens, alignment, nonzero start position and page allocations do not disambiguate them.

`sampling_params=None` is equally ambiguous: it means either intermediate prefill or a legitimate final request for host-only features such as logprobs/top-k40. Suppressing all such reads would break valid final requests. Returning fabricated logits is incorrect. Returning a device tensor through the existing host-only prefill output contract is also incorrect and would not prevent the plugin host sampler. Increasing the scheduler budget or disabling chunking removes required coverage rather than fixing the interface.

The generator's existing `return_device=True` and active-slot sampler can implement the model side after receiving an explicit mask. No duplicate sampling algorithm is needed.

## Existing mechanisms checked

- `supports_chunked_prefill`: allows chunked inputs; conveys no sampling/finality mask and documents the current host route.
- `supports_sample_on_device`, `max_device_top_k`, `supports_device_penalties`, `seeded_sampling_policy`, `sample_on_device_mode=all`: govern request feature support, not intermediate-row finality. Changing their meaning would itself require authorization.
- Decode `read_from_device`/async and selective host readback: decode-only contracts; no prefill hook.
- Device grammar and stage-input hooks: unrelated payloads; neither is a valid way to smuggle scheduler finality into a model.
- Runtime `TT_MODEL_CLASS_OVERRIDES`: already selects this adapter correctly; registration cannot fix missing execution metadata.
- No documented environment or `--additional-config` option expresses this behavior.

## Exact interface proposed for authorization

1. New model capability **`supports_prefill_sample_mask: True`**, default false.
2. For opted-in models, new `prefill_forward` keyword **`prefill_sample_mask`**, a CPU Boolean sequence of lengthB. True means the row completes the scheduled request history and needs an output sample; false means cache updates only.
3. False rows neither sample nor return host logits. An all-false call may return **`None`** (inside the existing MRoPE tuple when applicable); the plugin produces empty intermediate outputs without invoking either sampler.
4. Mixed calls retain submitted row order. On-device final rows use the existing token output `[B,1]` with ignored sentinel entries for false rows; only true rows advance sampling state. Explicit host-feature final rows retain the existing host compatibility path, restricted to true rows.
5. Remove the blanket intermediate-to-host override **only for opted-in models**. Keep all existing behavior for models that do not opt in. No model-name branch, new environment variable, TT config switch, or upstream-vLLM change.

## Effect on other models

Default-false models retain their existing arguments, routing, RNG behavior and outputs. Any model can adopt the documented mask after proving that false rows do not sample or read logits and that mixed rows preserve state/order. This autoport would be the first opted-in adapter. Host-feature finals remain supported; the proposal does not conceal missing device sampling features.

## Authorization

**Approved by the owner on2026-11-16**, delivered in this thread by the multigoal runner: “the shared plugin change proposed in doc/vllm_integration/plugin_changes.md is approved exactly as proposed: the capability supports_prefill_sample_mask, the prefill_sample_mask argument, and the all-intermediate no-output semantics described there.” The owner instructed implementation with planned validation and continuation of stage9. Original goal prompt copy: `/home/ttuser/qwen38-flash/multigoal_logs/09-09-vllm.prompt.txt`. This authorizes the five exact interface items above; no additional shared interface or served-default change is implied.

## Planned validation

Add plugin regression tests for indistinguishable64-final/64-intermediate inputs; all-intermediate batches; mixed final/intermediate rows in both orders; nonzero continuation starts; and unchanged final tokens/RNG versus an unchunked control. Verify no model/readback/sampler output handling is attempted for all-false outputs, false rows do not advance RNG, explicit host-feature finals still work, and non-opted-in adapters retain exact old behavior. Run the plugin host suite (`pytest tests/ --ignore=tests/tt`) per its contribution instructions.

On hardware, rerun reduced stale-input/page-growth/async checks and the same-server mixed-shape workload with allocation tracking, requiring zero forced host forwards/readbacks, stable traces, and greedy agreement with the non-overlapped control. Then run final full48 serving, sampling-full, P2/P3/P5 accuracy, qualitative controls, context acceptance and benchmarks. Commit any authorized plugin change locally in its own repository and record the SHA; never push.

## Implemented shared change and host validation

Plugin local commit `edb95ca695d719a5340d919ae3d6fc2f6dd219a8` implements exactly the approved opt-in interface in front-packed and lane paths, with final-only eligibility/sampling, all-intermediate `None`/MRoPE handling, final-row grammar selection and host metadata preservation. Upstream vLLM is unchanged. The plugin working tree was clean after commit; no push. Host-stub validation: targeted151 passed, full `pytest tests/ --ignore=tests/tt -q`2343 passed/16 skipped (two Numba warnings), explicit `pre-commit run` passed. Eight initial mask regression failures are preserved in `plugin_mask_before.log`. Exact commands and outputs: `plugin_mask_targeted.log`, `plugin_mask_host_suite.log`, `plugin_mask_precommit.log`, `plugin_mask_commit.log`. These are CPU contract tests; the subsequent affected hardware validation is recorded below.

Hardware follow-up (2026-11-16): the full48 `readiness_vllm/logit_baseline.json` passes canonical mask token/RNG equality and zero all-intermediate output/readback/sampling. Reduced synchronous and async serving both pass the affected mask/routing checks (`masked_control_v1/mixed.json`, `masked_async_v1/mixed.json`): device/none only, actual starts0/256/512, concurrent decode, no OOV, stable trace signatures. The async mixed comparator remains failed for6/8 greedy outputs, while its equal-width three-wave synchronous comparison passes exactly; the later matched-scheduler replay resolved that comparison coverage without deleting these failures. Full-layer affected-path validation subsequently passed as recorded below.

## Shared tile-test bootstrap defect (2026-11-16)

1. Requirement/failure: the required full `run_vllm_server --stages sampling --sampling-profile full` must collect the pinned plugin's complete `tests/tt` suite. Collection fails in `tests/tt/test_tile_size.py:5`: direct platform import enters vLLM's initialization before `current_platform` is available. Reproducer is the unchanged test imports in a fresh process; exact live-suite traceback is `readiness_vllm/sampling_tests.collection_failure.log`.
2. Owner: plugin test defect in its documented pytest entrypoint, not an adapter capability or model gap. `tests/tt/conftest.py` already documents the analogous incomplete platform-bootstrap hazard in its host-fixture override. The existing tile test is required to compare the plugin admission constant with TTNN's real tile size.
3. Model-side alternative: model changes cannot repair Python import ordering before test collection reaches any serving request. Preloading modules through an ad hoc test wrapper could hide the defect; preserving the standard documented runner and fixing the test's dependency order is the minimal correct change.
4. Existing mechanisms: normal runtime plugin registration already resolves correctly; no model capability, TT config option, model override, or sampling setting controls test-module import order. No test selector/skip or served-default change is appropriate.
5. Effect on others: initialize vLLM before direct plugin platform import in this one test. The same tile constant and real TTNN equality assertion remain unchanged; production code, other models and runtime defaults are untouched.
6. Authorization: adds no shared interface, hook, environment variable, capability or default behavior. Under the plugin-defect rules this test-only repair does not require additional owner authorization.
7. Validation: fresh processes with a guard requiring the host TTNN stub reproduced the exact original import error and confirmed that `import vllm` before the platform import resolves tile32. Planned retained regression record, plugin host suite, explicit pre-commit and full served sampling rerun; do not claim completion until those results are recorded.
8. Status: justified before edit. Proposed minimal change: add `import vllm  # noqa: F401` with bootstrap-order comment in `tests/tt/test_tile_size.py`; no production change. This was the pre-edit proposal; the following paragraph records its completed local commit and validation.

Validation for shared tile-test bootstrap fix: guarded stub regression reproduces old circular-import failure and passes after the one-line import; `logs/tile_bootstrap_guarded_before_after.log`. Full host suite2343 passed/16 skipped (`logs/tile_bootstrap_host_suite.log`), explicit pre-commit passed (`logs/tile_bootstrap_precommit.log`). Local plugin commit `ef1ec79b9679344d7a6820f8aae661bbb94d5909`, parent `edb95ca695d719a5340d919ae3d6fc2f6dd219a8`; only `tests/tt/test_tile_size.py` changed, runtime unchanged, clean worktree, no push. The unchanged real tile assertion passed in the final full sampling profile:67 passed/75 skipped, runner exit0; `../../readiness_vllm/sampling_tests.full_budget1528_v1.log`.

### Shared unseeded-variety workload defect (2026-11-16)

1. **Failing contract/evidence:** full sampling reports `test_uniform_noseed_varied` failure at configuredB2. Source `tests/tt/test_seeding_and_variety.py:238–248` creates exactly `max_batch_size` requests then requires at least5 unique responses; `utils.assert_varied` applies that absolute threshold. At B1/B2, this is mathematically impossible even with perfect diversity. Full failed-run assertion will be retained in sampling log.
2. **Owner/boundary:** this is a shared-test workload defect. The documented `--tt-max-num-seqs` permits low-capacity servers; client request count can exceed resident capacity and queue. Model and adapter cannot produce more distinct responses than submitted requests.
3. **Why adapter/generator fix is wrong:** inventing extra responses, lying about server capacity, or lowering the diversity threshold would corrupt the test. Keep actualC262144/B2 and the5-unique assertion.
4. **Minimal proposed fix:** generate `max(32, max_batch_size)` requests for this test, retaining its default32-request statistical workload even when resident capacity is smaller. Larger configured batches retain their existing count. No sampling values, output lengths, seed semantics or diversity assertions change; the low-capacity workload becomes larger.
5. **General applicability:** every supported adapter with resident capacity1–4 currently cannot pass. The fix separates sample count from scheduler resident capacity using the existing default32 workload; it adds no runtime capability, environment variable or operator setting.
6. **Alternatives rejected:** shrinking the5-unique oracle weakens coverage; skipping small servers removes it; overriding pytest capacity to32 misstates the runner/server configuration.
7. **Validation planned:** CPU mock regression proves32 calls at capacities1/2/4/32 and existing37 at37, preserves threshold5 and rejects outputs with only4 unique values; targeted live item then final full sampling; full plugin host suite and explicit precommit before local commit.
8. **Risk/authorization:** test-only additional queued requests on small servers, bounded by existing HTTP timeout. No runtime/interface/default behavior changes, no upstream-vLLM edit and no new owner authorization required under the skill's shared-defect rule. Justification precedes edit. No push.

Unseeded-variety fix locally committed `8fa7eb2da998547efc06bcde9fe5eef921e0019a`, parent `ef1ec79b9679344d7a6820f8aae661bbb94d5909`. CPU regression fail-before/pass-after verifies1/2/4/32 resident slots still submit32requests,37 submits37, and4-unique outputs still fail. Host2343pass/16skip and explicitprecommitpass. Evidence `logs/unseeded_variety_{workload_regression,host_suite,precommit,commit}.log` and `logs/unseeded_variety_check_workload.py`. Cleanpluginworktree, no runtime change/push. Live targeted unseeded-variety validation passed in `logs/sampling_workload_repairs_targeted_v1.log`; the final full profile passed67/75skip with runner exit0 (`../../readiness_vllm/sampling_tests.full_budget1528_v1.log`).

### Shared penalty-sweep workload defect (2026-11-16)

1. **Failing contract/evidence:** `readiness_vllm/sampling_tests.log` records failures in `test_different_repetition_penalties` (lines 424–475) and `test_different_frequency_penalties` (477–527). With truthful resident capacity B2, `tests/tt/test_tt_penalties.py` slices the original seven repetition values to `[0.1, 0.2]` and eight frequency values to `[-1.5, -1.0]`. Both submitted settings reward repetition; the saved responses repeat `a` or `5` identically. This evidence shows the workload omits neutral and repetition-suppressing settings; it does not alone prove whether every penalty is implemented correctly.
2. **Owner/boundary:** the defect is in shared test parameter coverage, not a model-specific runtime contract. The test promises varied penalty settings but conflates the number of parameter cases with simultaneous resident capacity. Existing `run_concurrent_batch` already permits queued requests above resident capacity.
3. **Model-side alternatives:** changing model output to force diversity, misreporting B2 as a larger capacity, or altering penalty math would invalidate the experiment. Generator changes cannot restore parameter cases the client never submits. Retain the actual serving capacity and all existing request values.
4. **Minimal change:** remove only `[:max_batch_size]` from these two fixed penalty lists, retaining all seven/eight original cases on every supported server. Keep prompts, generation lengths, greedy sampling, penalty values and the two-unique-response assertions unchanged. Add a short comment explaining that the parameter sweep queues independently of resident capacity.
5. **Other-model/default impact:** the documented default B32 already submits these same seven/eight requests, so its workload is unchanged. Smaller servers submit the complete existing sweep rather than a truncated prefix. Larger servers also retain the original fixed sweep. No model names, runtime branches, capabilities, environment variables or configuration defaults are introduced.
6. **Rejected alternatives and authorization:** lowering/skipping the variety oracle would weaken the test; selecting a different model-specific prompt or parameter range would change its experiment. Restoring the original complete generic range is a test-only shared-defect fix under the integration rules and needs no new interface authorization. Upstream vLLM and plugin runtime code remain untouched; never push.
7. **Validation planned:** preserve CPU AST/mock fail-before/pass-after coverage for capacities 1, 2, 4, 7, 8, 32 and 37. Verify complete ordered lists, original request fields, unchanged assertion AST/threshold, and rejection of identical mocked responses. CPU mocks establish workload and oracle behavior, not hardware penalty correctness. Run the complete guarded host suite, explicit pre-commit, and commit locally. Root owns later live targeted/full sampling validation and preserves original failure logs.
8. **Status/risk before edit:** this justification was appended before changing either test. The only operational effect is additional queued cases at low capacity (bounded by the existing seven/eight-request sweep); HTTP timeout and scheduler admission behavior remain unchanged. Local SHA and validation evidence will be reported separately; live pass remains unclaimed until rerun.

Penalty-sweep validation complete: CPU fail-before/pass-after coverage capacities1/2/4/7/8/32/37, unchanged assertion AST, and identical-output rejection all pass. Full host suite2343passed/16skipped and precommit pass. Evidence `logs/penalty_sweep_{workload_regression,host_suite,precommit,commit}.log`; local commit629f07cb385494b14d0ee3d138de039d71d09546; clean tree, no push. Both live penalty nodes passed in `logs/sampling_workload_repairs_targeted_v1.log`; the final full profile passed67/75skip with runner exit0 (`../../readiness_vllm/sampling_tests.full_budget1528_v1.log`).

## Affected hardware validation completed

The approved mask interface passed real recorded scheduler histories in both
arrival orders (`recorded_history_v1`, `recorded_history_reverse_v1`): immediate
and queued execution returned exactly equal tokens, with genuine async overlap
and zero unsafe-overlap observations. Failed earlier mixed-history comparisons
remain preserved and are not reclassified. The latest full48
`logit_baseline_table_snapshot_v2.json` verifies final-only sampling/token/RNG
and zero intermediate output/readback. Full48 `acceptance_tracking_v2/mixed.json`
passes non-aligned lengths, actual starts 0/256/512 with concurrent decode,
device/none routing, 202 stable execution signatures, zero retirements and OOV.
Its non-overlapped control is absent; the recorded-history controls are separate
artifacts with their own narrower diagnostic scope. The full48 served selected
logprob check also passes against the exact-source standalone baseline.

All four focused shared workload regressions (variety, top-k19, repetition,
frequency) passed on full48 in151.76 seconds. Three chat nodes subsequently
passed with the existing optional reasoning allowance1528 in112.10 seconds;
this used no new plugin interface or change. The native 8/9 control remains
failed, documented separately. Final full sampling, P5, maximum-context and
performance acceptance still must finish before the stage is complete.


## Stage10 proposal: remove implicit manual profiler read from mesh teardown

**Status: owner declined on 2026-11-17; NOT IMPLEMENTED. Plugin and vLLM source remain unchanged.**

The [owner decision](../optimized_vllm/owner_decision_shutdown_2026_11_17.json) supersedes the proposal and validation plan below. Retain existing default-timeout cleanup, require no owned processes/device handles, and carry slow/incomplete orderly teardown as a plugin-owned upstream limitation. The proposed patch remains unapplied. The following eight items are the historical proposal, not current instructions or approval.
Pinned plugin base: `629f07cb385494b14d0ee3d138de039d71d09546`.
Concrete unapplied patch: [proposed_plugin_no_profiler_close.patch](../optimized_vllm/proposed_plugin_no_profiler_close.patch).

1. **Requirement and reproducer.** The owner explicitly prohibits ReadDeviceProfiler in this serving stage. The selected vllm-integration skill also says “do not call `ttnn.ReadDeviceProfiler(mesh)` as part of vLLM-stage closure.” Pinned `src/vllm_tt_plugin/worker.py:1130` unconditionally calls that API from `close_mesh_device`, before closing submeshes/mesh and resetting fabric. `TTWorker.shutdown:609` reaches this helper after model capture release. [CPU reproducer](../optimized_vllm/plugin_shutdown_no_profiler_reproducer.json) executes the exact AST-extracted helper with device-reaching functions replaced by stubs: both zero/two-submesh cases attempt the forbidden call before closing anything. Removing only that call in an in-memory proposal preserves exact submesh→mesh→fabric order. This is not an installed-plugin test pass. With profiling disabled, native `ReadMeshDeviceProfilerResults` returns before queue synchronization/readback (`tt_metal/impl/profiler/tt_metal_profiler.cpp:1205`); no actual profiling collection is alleged. The conflict is the explicit no-call closure contract.
2. **Owner.** Missing shared behavior (plugin-change rules case2): ordinary mesh teardown without invoking a manual profiler API. The plugin owns mesh closure for every model; its `docs/MODEL_LIFECYCLE.md` and `TTWorker.shutdown` assign capture release to the model, followed by plugin-owned mesh closure. The required no-call behavior cannot be expressed by the current model interface.
3. **Model-side alternative.** A generator cannot stop the later plugin helper call. Closing the mesh early would violate capture/tensor ownership and leave the plugin using a closed mesh. Monkeypatching plugin functions from the adapter would conceal the shared change. Neither is correct. Skipping orderly shutdown or accepting force-kill before capture release also fails the cleanup requirement.
4. **Existing mechanisms checked.** `trace_mode`, `sample_on_device_mode`, `fabric_config`, warmup settings, model capabilities, `release_persistent_capture`, runtime adapter registration and the existing vLLM `--shutdown-timeout` do not disable this explicit helper call. Disabling `TT_METAL_DEVICE_PROFILER` stops native collection but does not prevent the Python API invocation. Starting a new environment-variable read or adding an opt-out config would add shared interface and is unnecessary. Separately, the existing `--shutdown-timeout 10` can provide teardown grace; it changes no plugin source and is planned for both final measurement arms.
5. **Effect on other models.** Remove the explicit manual profiler call for every model using this helper, preserving submesh close order, parent mesh close and fabric reset. No model-name branch, capability, environment-variable read, or JSON option is added. Profiling-enabled users no longer receive this plugin-issued final manual read/metadata submission; tt-metal's own close-time behavior remains unchanged. This shared default change is why exact authorization is requested, even though profiling-disabled native collection was already a no-op.
6. **Authorization.** Pending exact item: remove the unconditional `ttnn.ReadDeviceProfiler(mesh_device)` call and its obsolete comment from shared `worker.close_mesh_device`, for all models; add its generic regression and operator documentation. The task's general optimization request is not treated as authorization to change this shared default. Governing rule: [plugin-changes.md](/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills/vllm-integration/references/plugin-changes.md), “a change of plugin default behavior for models that do not opt in” requires authorization for that exact item.
7. **Validation plan.** The patch includes a host regression invoking the actual helper with zero/two submeshes, a failing profiler-call sentinel, and exact close/fabric order. After authorization, run that test against pinned source to retain fail-before evidence, apply the minimal runtime change, rerun it, then the full guarded host suite (`PYTHONPATH=tests/host-stubs "$VIRTUAL_ENV/bin/python" -m pytest tests/ --ignore=tests/tt`) and explicit plugin pre-commit. Add operator lifecycle text. Run final real-server teardown with the existing10s grace: every capture ID must have a shutdown-release event, no forced worker kill/error, no owned PIDs or device handles. Final matched primary/CI/burst, tracked sync/async/mixed, sampling, qualitative and P4/P5 checks use the same authorized plugin commit. Commit locally in the plugin and log its SHA; never push.
8. **Status.** Proposed only. The CPU in-memory reproduction makes the patch reviewable but does not authorize installation, establish full host-suite compatibility, or close hardware cleanup. The unchanged sweep completed: budget9 passes4K/32K worst ITL592.1/615.7ms and is selected; all owned processes/device handles are gone. The default-timeout teardown remains incomplete. Final orderly-closure runs await this authorization.
