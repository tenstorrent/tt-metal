# Stage 09 vLLM work log

## 2026-09-27 prerequisite attempt

Target: google/gemma-4-26B-A4B-it. Start: clean tt-metal worktree at
`eeb18d817be12d288c3c437dd593a468e9aa02ae`, branch `gemma-4-26b-a4b-it`.
Read the vLLM integration/device usage skills, installed model-bringup startup,
model generator, integration guide, reference adapter, datatype-sweep README,
and context contract. Enabled skill catalog and inherited runner environment
supply tt-model-bringup and tt-autodebug paths. No dependency installation.

Commands and observations:

1. `python -m readiness_check.run_vllm_server --help`: exit 1,
   missing `openai`, before argument parsing. Saved `readiness_vllm/runner_startup.log`.
2. `git ls-remote https://github.com/tenstorrent/vllm.git HEAD`: succeeded.
3. `git clone --depth 1 https://github.com/tenstorrent/vllm.git vllm`:
   succeeded, initially main. `git -C vllm fetch --depth 1 origin dev` and
   `git -C vllm checkout -b gemma4-vllm-integration FETCH_HEAD` select
   compatibility revision `5ffebf4128f81ea5cf8413175eabde52cd8c8d75`.
4. `/opt/venv/bin/python` inventory: no openai or torch. The local `vllm/`
   namespace is not an installed runtime; package metadata confirms absence.
5. `PYTHONPATH=/workspace/tt-metal/vllm:$PYTHONPATH python -m
   vllm.entrypoints.openai.api_server --help`: exit 1, missing `uvloop`.
   Saved `readiness_vllm/vllm_import.log`. Source checkout is insufficient.
6. `timeout 60 tt-smi -ls --local`: exit 0, four Blackhole p300c ASICs visible;
   saved `readiness_vllm/device_list.log`. No device opens or resets required.

`environment.json` records runtime package versions. AutoDebug independently
checks installed environments; AutoFix tests only existing-runtime recovery,
respecting the user-supplied AGENTS.md prohibition on installing dependencies.
No server, EngineCore, benchmark or model process was launched by this attempt.
The unrelated pre-existing Tracy web viewer is left untouched.

Serving command/status: no successful launch; prerequisite import fails.
TT config, max-num-seqs and sampling profile were not exercised. Planned final
context is 262144, mesh P300x2, max-num-seqs 32 and full sampling profile; no
trace-region/fabric values are claimed validated for vLLM. Primary workload
4096/128/B1/C1 and secondary CI burst 100/100/32 have no measurements.
Qualitative verdict and TTFT/TPOT/ITL P50/P99, throughput and decode t/s/u are
unavailable. No benchmark paths are fabricated.

No implementation changes or checkpoint commits; no push. A clean-pass review
and all implementation/serving gates remain outstanding. The dependency source
checkout is retained for resumption. Packet:
`bringup/artifacts/multigoal-runs/20260925T171711Z/telemetry/packets/d0f559ee-1c56-407d-8a1d-0b517673de5b.json`.

Independent AutoDebug report confirms no usable alternative interpreter: cached
serving wheels target CPython3.12, while active environments use3.10, and the
cached pytest environment points to an absent interpreter. See `AUTODEBUG.md`
for the dependency closure and plugin entry-point requirements. Source-only
import probing cannot repair these prerequisites. Final process audit found
no `EngineCore`, `vllm.entrypoints` or readiness server process. Packet JSON
parsing and `git diff --check` passed. Docs/evidence-only changes need no build.

AutoFix final result: blocked by external runtime prerequisites. Four interpreter
probes and both help checks independently reproduced the failure; see
`AUTOFIX.md` and `readiness_vllm/autofix_*`. The initial check used the packaged
runner; AutoFix also confirmed the same missing-openai failure in the repository
shared runner. No safe source-only repair resolves the missing dependency closure.

Continuation audit 2: package metadata rechecked; openai, uvloop, vLLM and
TT plugin remain absent. No new installation authorization or provisioned
environment is available. The prior turn produced source and AutoFix evidence;
this turn confirms the same external blocker. Goal remains active pending
the blocked-audit threshold; no completion is claimed.

Continuation audit 3: same four missing distributions confirmed again; no
external runtime or changed installation authorization. Previous continuation
classified as no progress toward serving. AutoFix recovery remains unsuccessful
under the no-install constraint. Three consecutive blocked goal turns reached;
goal marked blocked, not complete. All serving acceptance gates remain pending.

## 2026-09-27 operator runtime recovery

The missing-dependency conclusion applied only to the generic TT-Metal image
and `/opt/venv`. A pre-provisioned local serving image was subsequently found:
`ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.21.0-7aee19266b8c00e8d7cacbd17d1e875210db73a5-c9cfebc-103959377004`.
Its actual serving interpreter is
`/home/container_app_user/tt-metal/python_env/bin/python`, which contains vLLM,
the TT plugin, OpenAI client, uvloop, and its matching TTNN build. No dependency
installation is required or authorized.

Use `/home/mvasiljevic/gemma4-pipeline/vllm-container.sh` for Stage 9 and later.
It deliberately keeps the image's TTNN extension and `build/lib` together,
mounts the editable checkout at `/workspace/tt-metal`, overlays the current
autoport into the image's model tree, and selects the editable plugin source at
`/workspace/tt-metal/vllm/plugins/vllm-tt-plugin/src`. The corrected probe
successfully imported TTNN, that plugin source, and this checkout's Gemma
generator. See `../PIPELINE_INTERVENTIONS.md` for the full root-cause record.

## Resumed attempt 9dfe7c08-abbb-420b-a031-67f7991f6dbe

Operator provisioned an existing serving image; see ../PIPELINE_INTERVENTIONS.md.
Current package versions recorded in resume_environment.json. Shared packaged
runner --help succeeds. Initial reduced generator mesh-open failed creating
/home/container_app_user/logs/generated/watcher; setting TT_METAL_LOGS_PATH to
readiness_vllm/runtime_logs fixes writable logging. No reset or installation.
The retry of tests.check_full_trace with layers0/5 passed feedback/positions,
page-table refresh and repeated-generation checks; this validates the inherited
generator against the image runtime, not final adapter behavior or performance.
Logs: readiness_vllm/resume_generator_trace{,_retry}.log and
readiness_vllm/resume_generator_trace.json.

Adapter, per-layer page-table routing, plugin cache geometry and overlap repairs
are in progress. Dedicated source diagnosis: AUTODEBUG_contracts.md. Do not
interpret the previous dependency-only AUTOFIX.md as the current stage status.

### Adapter and runtime compatibility validation

`tests.check_vllm_adapter --output readiness_vllm/adapter_shared_pool_retry.json`
passed on the 1x4 mesh with layers 0/5, a 33-token prompt, scheduler-owned hybrid
cache aliases, stale host tokens/positions, deferred first-device token readback,
and exact standalone token agreement. Precision ID was
`head4_inner_all4_shared_down4`; observed cache dtype BFP8. This is reduced
contract evidence, not full-model serving quality/performance.

Reduced shared-runner invocation is serialized in `reduced_server_command.json`.
It preserves max_model_len=262144, max_num_seqs=32 and decode_only tracing.
Attempt3 reached healthy serving but request validation failed before execution;
AutoFix proved and repaired installed vLLM0.26 two-argument request validation
versus the legacy plugin's three arguments (5 host tests).
Attempt4 passed validation, then failed at scheduler.schedule(throttle_prefills)
versus the plugin's no-argument override. Evidence: readiness_vllm/server_attempt4.log.
No device execution for that request. Runner exited and process audit found no
EngineCore, API server or runner remnants. Scheduler AutoFix is ongoing.

Attempt5 reached prefill and `SPLIT_TRACE_READY` after scheduler repair. Its
synchronous plugin finalization calls process_decode_output_host directly on the
raw distributed device output (no read_decode_output call). Plain to_torch
required a mesh composer; selecting the first replicated token tensor fixes this
interface case without a logits readback. The reduced adapter regression now
covers both raw-device synchronous finalization and deferred host finalization.
Evidence: server_attempt5.log, request_attempt5.json, adapter_sync_async artifacts.

### Sampling compatibility source contract

Focused verification of AUTODEBUG_contracts.md finding 5 is recorded in
AUTODEBUG_sampling_contract.md. TP4 logprobs (including 0) and host-only
parameters select plugin host sampling, which omits the sampling_params kwarg;
the default adapter explicitly refuses this path. Added host-only
`tests/test_vllm_sampling_contract.py`: 26 tests passed against installed
vLLM0.26 and source TT plugin, including actual submit/refusal, real host sampler
logprob math, finalization and lane logits layouts. Black check (py310) and
`git diff --check` passed. No sampling implementation or device execution was
changed by this investigation. The report proposes an explicit optional logits
output mode in the canonical generator; it preserves default refusal and the
performance split-sampling path. Optional-mode implementation, device checks,
and the shared full sampling suite remain outstanding.

### Concurrent prefill row repair

Attempt6 completed isolated and concurrent 31/63/95-token prompts with96 output
tokens, but the latter two concurrent streams diverged immediately after
prefill. `requests_reduced_control.partial.json` and `request_batch_order.json`
preserve the failure. Server was stopped cleanly via runner SIGINT.
Direct two-row adapter probe `adapter_batch_cache.json` matched both standalone
controls, refuting a general batch-decode failure. Plugin `empty_slots` denotes
persistent state locations, while the attention page tables use compact request
rows. Passing those slots as `user_id` redirected prefill KV writes during
staggered arrivals. Adapter now uses generator's compact row default.
`adapter_batch_state_slots.json` passes with deliberately nonidentity state slots
[5,7], same disjoint scheduler-owned pools, and exact standalone token matches.
Attempt7 reruns the original concurrent/lifecycle request gate.

Attempt7 original request gate passed: requests_reduced_slot_fix.json contains
nine successful96-token completions, including31/63/95-token concurrent prompts
and33/1057/33-token lifecycle sequence. Concurrent tokens equal isolated controls,
and repeated33-token output is identical across the intervening long request.
This reduced evidence demonstrates arbitrary prompt lengths and cache/page
lifecycle, not full-model quality. Runner SIGINT closed the server cleanly;
server_attempt7.log preserved. Async reference run is next.

### Optional host sampling implementation

Implemented the explicit GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING=1 adapter mode and
canonical generator return_logits low-level option. The latter captures/replays
the existing model forward with no sampler work, seed updates, host argmax, or
low-level readback; vLLM owns host sampling after the adapter transfers logits.
Default host refusal, compact prefill table rows, selected precision, and normal
device sampling remain intact. Host/device mode switches invalidate traces and
sampling signatures; every host decode refreshes scheduler tokens/positions.
Tests/test_vllm_sampling_contract.py now passes 38 host tests (18.27 seconds),
including actual plugin submit/finalize, vLLM CPU logprob math, mode transitions,
model/sampler capture separation, and scheduler refresh. Black py310 and diff
whitespace checks pass. See AUTOFIX_sampling_contract.md and
readiness_vllm/sampling_host_contract_tests.log. No TT hardware/server execution
was performed by this repair task. The temporary async capability gate was
removed and supports_async_decode enabled after the main stage supplied its
successful nine-request synchronous/asynchronous exact-token comparison.

### Async and optional compatibility proof

Attempt8 (`reduced_async_server_command.json`) enabled async scheduling with the
probe capability flag. `requests_reduced_async.json` passed all nine96-token
requests and exactly matched the synchronous `requests_reduced_slot_fix.json`,
including allocator page growth, concurrency3, and1057-token input. Combined
with direct stale token/position and changed/unchanged page tests, this proves
`supports_async_decode=True`; the temporary gate is removed from implementation.

Attempt9 enabled explicit `GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING=1`. Logprobs3
requests returned all8 token probabilities and top alternatives;
`requests_reduced_compat.json` preserves device→host→device calls. Device tokens
before/after match exactly. The initial strict cross-mode token-equality assertion
failed: reduced logits have tied special-token maxima. A focused20-logprob
response proves device token47 and host token1 both equal -11.42144775390625 at
step0 (`reduced_tied_logprobs.json`). This is reduced tie-breaking evidence, not
full-model quality. Two host min_p=0.1/temp0.7/top_k32/seed42 calls matched all4
sampled tokens (`reduced_host_minp.json`). No sampler is executed inside the
optional logits-only trace; vLLM owns that host sampler. Normal device requests
continue the canonical split path. All reduced servers stopped via runner SIGINT.

Full30-layer shared server now launching with `full_server_command.json`:
async enabled, full sampling profile, context262144, maxseqs32, exact headline
4096/128/B1/C1 flags. Headline and CI metrics remain unmeasured.

### Full-model initial evidence and padding optimization

Full30-layer shared qualitative run completed with12 greedy/sampled outputs.
All six greedy strings preserve selected datatype-sweep control prefixes;
rendered prompts match the pinned HF controls. Read and judged every output;
see qualitative_verdict.md. Degeneracy checker exits0/no findings.

Initial exact4096/128/B1/C1 serving baseline: TTFT7473.072ms, mean TPOT58.644ms,
decode17.05197t/s/u, ITL P50/P99=58.341/69.284ms, aggregate8.5784tok/s.
Raw/normalized JSON, source hashes and server log preserved under
readiness_vllm/baseline_padded_decode/. These are actual full-model vLLM
measurements; no teacher-forcing number is substituted.

Plugin pads wire decode rows to32; the adapter initially sent all32 to the
model, which concatenated31 inactive rows at each layer for a single request.
Adapter now trims unused trailing rows at layout reset, retains interior
inactive slots and preserves stale-input device feedback. Direct padded32/two
active-row device probe matches both standalone controls
(adapter_trimmed_batch.json). Reduced async HTTP nine-request regression also
matches the synchronous reference exactly (requests_reduced_trimmed.json).
Host suite:43 cases passed, one object-identity assertion updated to check equal
zero-copy tensor views; its two parameterized cases then passed. Earlier slot
fixture was updated to use the real adapter constructor after the new sampling
mode state was introduced. Full-model same-workload remeasurement is ongoing.

### Shared allowed-token assertion repair

The full sampling suite reported test_allowed_token_ids failing. Source and a
cached CPU tokenizer probe proved Gemma allowed IDs1/2/3 are EOS/BOS/UNK and all
decode empty under vLLM's default skip_special_tokens=True. The original shared
test required nonempty text and never checked actual token membership. A CPU
control reproduced that exact false failure before changes. Shared test/helpers
now request return_token_ids and require nonempty generated IDs, all within the
per-request allowlist, at most max_tokens, and exactly matching completion usage.
Nine host tests pass, including six rejection controls and both request-helper
forwarding paths. No model/plugin runtime change or server request was made.
AUTODEBUG_allowed_token_ids.md records proof, commands, baseline/final logs and
the failed independent AutoDebug runner (nested sandbox lacked bubblewrap).
The main stage must rerun the corrected test on the full-model server; this
assertion diagnosis alone does not establish the original live token IDs.

### Shared temperature fixture correction

Installed vLLM0.26 rejects temperatures above2 before model execution. Six
shared seeding/variety request sites used5/10/50, covering13 test items. CPU
regressions using actual SamplingParams reproduced14 failures (13 cases plus
an all-files scan), with3 expected-rejection controls passing. Replacing only
those values with2.0 and updating corresponding messages gives17 passing CPU
checks. AST comparison proves all seed/top-k/prompt/repeat/variety assertion
semantics retained. No runtime changes or serverrequests; active fullsuite logs
untouched. See AUTOFIX_shared_test_temperatures.md and shared_temperature_*
artifacts. Live affected-item reruns remain the main stage's responsibility.

### Reviewer prefill penalty-history repair

Verified P1: rectangular prefill tokens included padded/stale suffix IDs in the
common repetition-penalty prompt mask. Adapter now masks sampler history after
each prompt_len to-1 while preserving the original model-input tensor. Actual
SamplingGenerator/TTPenalties host paths prove zero padding and stale19 are
excluded while valid token0 is retained. Baseline2 failed/1 passed; final full
adapter CPU suite42 passed in21.61s. Decode's existing make_prompt_token_ids_tensor
already masks each original prompt length and passes a reordered/stale-tail
regression, so decode runtime is unchanged. See AUTOFIX_prefill_prompt_mask.md
and unique prefill_prompt_mask_* logs. No server/hardware or active-log writes.
Main must restart the currently loaded server after its full suite and rerun
final serving/penalty validation to exercise this source fix.


### Full-profile first pass and verified shared-test repairs

Initial full profile:55 passed,17 failed,1 skipped in940.66s;
`sampling_tests_initial_full.log` preserves it. Thirteen failures were HTTP400
for temperatures5/10/50 rejected by installed vLLM0.26 (maximum2). Shared tests
now use2.0 without relaxing output assertions;17 CPU compatibility tests pass.
Allowed-token correction passed live (1 test,5.67s).

Two presence-penalty checks used a cyclic ABC prompt whose greedy continuation
is unchanged by presence2 in both canonical CPU and device controls. The shared
presence prompts now use a less constrained story continuation; penalty0/2
outputs differ and each exactly matches CPU sampling. Both targeted tests pass
(51.65s). Controls and repair are in AUTOFIX_presence_test.md. The remaining
mixed-parameter reproducibility failure is still under investigation, not
waived. Full live logprob repeated and AB/BA submission checks all match exactly
(`logprobs_full_server.json`).

Independent review is underway; its prompt-tail contamination finding triggered
an adapter sampler-history mask fix and actual common-penalty CPU regressions.
Full server was stopped through runner SIGINT and ps audit found no runner,
vLLM entrypoint or EngineCore afterward. Full initial server log preserved as
`server_full_initial_sampling.log`. Reduced trace-allocation functional check is
running with tracking/tracebacks enabled and program-cache checking retained;
no serving profiler was used.


### Numerical and trace-lifetime closure

`adapter_trace_allocations.json/.log` passes reduced canonical split sampling,
stale token/current-position, changed-page and deferred readback assertions with
TT_METAL_TRACE_ALLOC_TRACKING=1, TT_METAL_TRACE_ALLOC_TRACEBACKS=1 and
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0. No unsafe surviving allocations were
reported before either model or sampler replay. Reduced and full-model numeric
checks then ran with the same flags using check_vllm_logit_determinism.py
(--full-model for all30). Both pass22 exact full-vocabulary comparisons across
standalone/direct-adapter repeats, swapped batch positions, last prefill and3
fixed teacher-input decode steps. Artifacts logit_determinism_{reduced,full}.json
and .log; complete local vectors are inventoried by logit_tensor_manifest.json.
These functional runs are not serving performance or profiler evidence.

Live first-token20-logprob dictionaries from logprobs_full_server.json exactly
match torch.log_softmax(standalone_reference[request][0].float(),dim=-1) for
both31/63-token prompts (logprobs_standalone_comparison.json). Scope is explicitly
last-prefill for HTTP, three decode steps for direct adapter. Final combined
adapter host suite:47 passed in26.63s (final_adapter_host_tests.log).


### Layout-change seed state investigation

After prompt masking, all14 targeted prior sampling failures pass in273.09s
(`sampling_targeted_final.log`). A separate state audit found seed restart on
parameter/layout changes and missing penalty-history reset when a slot remap
preserves identical parameter values. CPU controls fail both cases. The live
`check_vllm_seed_lifecycle --tokens24 --peer-tokens3 7` probe then proves allfour
concurrent target/short-peer cases diverge from the bracketed isolated seed6
controls. First mismatches are index3 for3-token peers and index9 for7-token
peers, in either submission order (`seed_lifecycle_before.json/.log`). This is
required repair work, not a waived reproducibility limitation.

Server stopped via SIGINT49948, full log retained as
`server_prompt_mask_fixed.log`; process audit shows no runner/EngineCore/vLLM
entrypoints. AutoFix is implementing trusted output-count transport and
request-history reset only at state-change boundaries, preserving unchanged
on-device token feedback.

## 2026-09-27: sampling lifecycle reset repair

Verified two independent sampler state defects with failing host controls: surviving seeded requests restarted their increment sequence after another request finished; identical-parameter row remaps retained stale penalty histories. Parent HTTP control corroborated all four co-request divergences while isolated controls matched. Added opt-in DP1 output-count transport and canonical boundary-only seed/history restoration, preserving unchanged trace bindings and greedy no-op state behavior. Host gates: 64 sampling-contract tests and 25 plugin cache/scheduler tests pass. Live reruns remain parent-owned. See `AUTOFIX_sampling_lifecycle.md` and `readiness_vllm/seed_lifecycle_*` artifacts.


Decode seed repair passes the identical live before/after probe: allfour target/
short-peer pairs now exactly match both isolated seed6 controls. The seed-only
variant also passes allfour orderings without penalty history. Artifacts:
seed_lifecycle_after.json/.log and seed_lifecycle_after_seed_only.json/.log.
Server stopped cleanly via SIGINT52458; log saved as server_decode_seed_fixed.log
and ps showed no serving leftovers.

Independent rereview of this patch found reachable KV preemption/resumed-prefill
state loss: re-prefilling retained generated tokens must preserve their original
seed count and distinguish original-prompt repetition history from generated
presence/frequency history. This is a separate required repair in progress;
no context or preemption capability reduction is being used to bypass it.

## 2026-09-27: resumed-prefill sampling state

Fresh review found that supported preemption replays original prompt plus retained outputs, while sampler initialization treated that entire prefix as prompt and reset seed/output history. Four CPU negative controls verified producer, consumer, and actual canonical sampler state gaps. Extended existing opt-in count transport to prefill and restored original-prompt mask, suffix output counts, and base+k seed while preserving full model prefix. Full host contract: 77 passed in 33.40 s. Reduced same-logits device oracle prepared for parent execution; see `AUTOFIX_resumed_prefill.md`. Both earlier finishing-peer lifecycle HTTP probes now pass after the decode fix, with and without penalties.


Resumed adapter prefill now passes the reduced tracked same-logits oracle:
seed777317, prompt/output masks and output counts match, and token333 matches
reference333 (resumed_prefill_reduced.json/.log). The uninterrupted token349 is
informational because recomputed prefill and incremental decode have different
selected precision paths. CPU77-case suite passes.

Reviewer then identified a producer-side async-preemption reconciliation gap:
worker history can include a token the scheduler discarded. The next full-server
launch was cancelled before any request/check, preserving
server_resume_launch_cancelled.log. SIGINT53753 stopped runner/API startup but
left orphan EngineCore53830; SIGTERM53830 removed it. A subsequent ps audit found
no runner, EngineCore or vLLM entrypoints. This explicitly records the interrupted
startup cleanup rather than claiming runner-only shutdown was sufficient.


Post-cancellation health: bounded tt-smi list could not run because this image
has no tt-smi executable (exit127; post_cancel_device_list.log). A bounded
TTNN1x4 FABRIC_1D mesh open/close succeeded with MESH_SMOKE_OK and exit0
(post_cancel_mesh_smoke.log). No reset, profiler or dependency installation was
needed; all four devices opened and closed normally.

## 2026-09-27: authoritative retained history after preemption

Verified async-discard/resume mismatch with 9 failing CPU cases and 2 passing ordinary-decode controls. Shared cached-state updates now reconcile retained suffix only on resume, using scheduler full IDs/count; existing front-packed rows reload their full prefix/count/block mapping. Pending output drains precede reconciliation in normal/lane entry points. Final gates: 37 plugin retained-history/cache tests and 78 sampling-contract tests pass. See `AUTOFIX_resume_worker_history.md`; final device/server loading is parent-owned.


Final full30-layer server launched with `full_server_command.json` after all
resume-history fixes. Both final seeded lifecycle probes passed all four
finishing-peer orderings (with penalties and seed-only);
`seed_lifecycle_final.json` and `seed_lifecycle_final_seed_only.json`.
`requests_full_final.json` passes nine96-token requests:31/63/95-token prompts
isolated and concurrent, then33/1057/33-token prompt reuse.
The shared runner is executing `final_checks_command.json`, with default CI
burst enabled and exact4096/128/B1/C1 primary workload.


Final shared primary benchmark (4096 input/128 output/B1/C1) completed1/1:
TTFT P50/P99 2549.301076/2549.301076ms;
TPOT mean/P50/P99 19.712850/19.712850/19.712850ms;
ITL P50/P99 19.619414/27.752106ms; aggregate
25.330633tok/s;1000/meanTPOT=50.728332t/s/u.
Secondary CI100/100/32 burst completed32/32,3200outputtokens:
TTFT P50/P99 30815.582207/30816.835017ms;
TPOT mean/P50/P99 680.950320/677.478360/754.133229ms;
ITL P50/P99 677.253453/692.436601ms;
aggregate32.690414tok/s;TPOT-derived
1.468536t/s/u is secondary burst context only.
Exact commands/config/rawpaths are in vllm_benchmark.json and
vllm_ci_serving_benchmark.json; rawfiles vllm_result.json and
vllm_ci_serving_result.json.
Final12 qualitative outputs read; allsix greedy prefixes and two256-token
controls match, final degeneracy exit0/no findings. Fullsampling now running.


Final static checks: model Python Black check passed for13 changed/new files
(`format_final.log`, line-length120,targetpy310); git diff --check passed in both
repositories. Plugin Black check left9 checked files unchanged; it proposed
rewriting pre-existing assertion formatting in platform.py/input_batch.py
(`format_plugin_final.log`). Those unrelated lines were retained; no serving
source changed after the final source manifest. Python/docs changes require
no C++ build under AGENTS.md.


Final shared runner exited0: fullsampling72passed,1skipped,zero failed in
1123.65s. The existing all-vocabulary chat-logprob test skipped because of the
default max_logprobs20 cap; all ordinary top-N logprobs passed.
`final_validation.json`, `sampling_tests.log`, and `final_checks_runner.log`
record this final run. Both presence tests and the formerly failing mixed-params
request-isolation case pass in this fullprofile.
After checks, SIGINT55104 stopped the held shared server; runner exit0,
API55123 andEngineCore55165 exited. Final ps inspection found no serving
runner/API/EngineCore leftovers; `final_cleanup.json` records before/after.
No profiler/reset or device-health probe was used at final closure.


Shutdown audit detail: the installed API server uses abort/timeout=0 for this
runner stop. It reports force-stopping one remaining EngineCore, then records
engine-manager completion and FastAPI application shutdown. The process audit
confirms all three owned PIDs are gone; this is not a claim of graceful device
teardown. Nanobind reports105 instances/973 types/4455 functions at interpreter
exit, exactly the same counts as the prior `server_decode_seed_fixed.log`
shutdown after a much smaller workload. These are binding-reference teardown
warnings preserved in the logs, not leftover serving processes or a measured
per-request growth claim. No warning was suppressed and no additional reset or
profiling was performed.


Independent final $stage-review returned clean-pass with no required work:
`stage_review.md`; initial findings preserved in `stage_review_initial.md`.
The final source and artifact manifests match the measured path.
Post-review local checkpoint: vllm repo, branch gemma4-vllm-integration,
commit `7f72b1c6e905f5137fe3377f2e7b42738d3f271d`. No push. Commit identity was absent in this runtime;
used command-local Codex <codex@openai.com>, matching prior tt-metal automation
identity, without changing global Git configuration.


The tt-metal commit hook normalized final newlines in eight generated JSON
files and sorted test imports. Parsed JSON equality against staged originals
was verified (`post_hook_artifact_check.json`); final artifact/qualitative hashes
were refreshed. Four negative tests now use the repository expect_error fixture
with the same exception and regex. Post-hook targeted host rerun passed83/83
in35.43s (`post_hook_host_tests.log`). All11 measured runtime/policy hashes remain
unchanged. These closure-only changes were sent for independent review audit.


Final post-hook independent review retains clean-pass;83hosttests passed.
Local tt-metal source/evidence checkpoint: branch gemma-4-26b-a4b-it,
commit `d6a165c5ff66af6a40471770003a855373db73c4`. All applicable repository
pre-commit hooks passed on this commit; no hook was bypassed.
Local vllm checkpoint: branch gemma4-vllm-integration,
commit `7f72b1c6e905f5137fe3377f2e7b42738d3f271d`.
`checkpoints.json` records both. This following documentation-only checkpoint
records their SHAs; nothing was pushed. Stage09serving requirements are complete.
Rawlogs and tensor controls remain local under existing ignore policies; JSON
evidence and manifests are versioned. The unrelated operator intervention note
and embedded vllm directory were excluded from the tt-metal checkpoint.
