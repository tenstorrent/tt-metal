# Stage09 vLLM integration

Started from datatype-sweep clean-pass, tt-metal checkpoint fe82645a041.
Selected head_bfp8_hifi2 policy and native context262144 remain authoritative.
Headline target:4096 input/128 output/batch1/concurrency1. No serving measurement yet.

Enabled skills: vllm-integration, agentic-serving, tt-device-usage,
tt-enable-tracing, AutoDebug/AutoFix, model-bringup-telemetry.
Plugin inventory: .codex/config.toml. Environment setup validated with the
installed model-bringup0.1.21/scripts/environment.py and tt-autodebug0.1.8.

Startup device command:
`docker exec qwen38-functional-build timeout 60 tt-smi -ls --local`
exited0 and listed four Blackhole p300c devices. No serving processes found;
no reset performed. All device commands remain serialized. No serving profiler.

Serving source resolution:
`git ls-remote --symref https://github.com/tenstorrent/vllm-tt-plugin.git HEAD`
resolved main at1badbd02bb6911dfdb59c206bdc00c6c1e8dc0bf.
Checkout: bringup/dependencies/vllm-tt-plugin (separate repository).
Its README.md and docs/install-vllm-tt.sh recommend upstream
https://github.com/vllm-project/vllm tag v0.30.0, empty target source build.
Read docs/vllm-overrides.txt before installation. Source install runs in the
validated qwen38-functional-build container with `set -e; UV_NO_CACHE=1;
source docs/install-vllm-tt.sh`. Exact output: readiness_vllm/install.log;
pre-install inventory: readiness_vllm/environment_before.txt.

Initial interface inspection: canonical split sampler and device feedback exist,
but public low-level decode always reloads host token/position. Pinned plugin
negotiates decode_input_update_contract=1 with independent input, page-table,
sampling-param and sampling-state commands. Recurrent state slot ownership and
prefix hits also require explicit adaptation. AutoDebug launched inspection-only
against generator.py, model.py and plugin interface docs. Sandbox preflight failed
because this host disallows unprivileged namespaces; retried with the skill's
explicit AUTODEBUG_SKIP_CHILD_SANDBOX=1 under existing unrestricted authorization.
Evidence: readiness_vllm/autodebug_launch.log.

Local telemetry packet initialized from the supplied exact-identifier template;
unmeasured serving metrics remain null. No pass or performance claim yet.

## Contract repairs and reduced validation

AutoDebug completed at AUTODEBUG.md. AutoFix work is documented in
generator_autofix.md, plugin_state_autofix.md, mamba_cache_bridge.md and
state_pool_design.md. The shared generator now uploads inverse temperature,
separates sampler parameters from RNG reset, retains device token/position state
on explicit no-reload submissions, and moves recurrent/sampler state with slots.
The adapter uses absolute sampled-token position as its seeded RNG counter:
prefill offset=exclusive prompt end; decode reset offset=start_pos+1. This is
reproducible across resume and is not a claim of HF CPU RNG equivalence.

All device probes use this command prefix inside the existing container:
`docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh python -m models.autoports.qwen_qwen3_8_27b.tests.`
Modules and compact results under readiness_vllm:

- probe_serving_state: state_gather_probe.json, exact BF16/FP32 gather.
- probe_serving_generator_contract: generator_device_contract.json, pass;
  resident tokens/positions/all state match serial control, zero steady token
  and position refreshes, one actually consumed page change.
- probe_serving_adapter_contract: adapter_device_contract.json, pass after
  fixing use of gather's returned tensor. Initial adapter_device_contract.log
  failed the nonzero conv remap check; adapter_device_contract_retry.log passed.
  Optional gather out storage is transformed for non-last-axis gather; copying
  that original scratch instead of the returned tensor was incorrect. Delayed
  read N survives submission N+1, captures remain4 and host-logit reads0.
- probe_serving_state_pool: state_pool_probe.json, pass; exact two-layer
  synthetic conv/recurrent snapshots, independent peers, immutable prior
  snapshot rows and retained trace replay. First attempt hit TTNN Shape slicing
  in Python; tuple conversion fixed it before kernel execution.
- probe_serving_continuation: continuation_probe.json, pass for193 tokens split
  64/64/65 and129 split123/6. The short logits-only teacher-forced bridge reaches
  an aligned prefill boundary without replaying consumed recurrence. No sampler
  advancement or inactive-peer mutation. Numerical tolerances and all four
  device comparisons are in the artifact, not model accuracy metrics.

These are reduced or synthetic contract probes, never full-model serving
accuracy/performance. Tracking is1, program-cache exclusion0. No profiler.

Shared plugin pytest collection:150 tests collected using --tt-server-url,
--tt-model-name and --tt-max-num-seqs2; pytest_collection.log. Actual full server
sampling remains pending. pip_check.log records existing unrelated image/eval
package conflicts plus the plugin's documented OpenCV4.11/numpy<2 override;
imports.log confirms TTNN, generator, vLLM0.30.0 and standalone plugin imports.

First reduced server command used the shared runner with real layers0/3,
max_num_seqs1, max_model_len262144, block_size32 and tracking enabled. It reached
health, then a direct129-token/3-output greedy request returned500 because the
plugin expected the text MRoPE `(output,zero_deltas)` tuple. Fixed in adapter;
reduced_initial_server.log and reduced_non_aligned_initial.json preserve failure.
Runner/server/EngineCore all exited; process audit found no leftovers.

Hybrid prefix implementation uses upstream MambaSpec align checkpoints and
4096-token scheduler blocks expanded to32-token attention pages. Physical
snapshot copies retain exact FP32 recurrence and BF16 convolution. The planner
uses padded virtual pages; physical pools are separate typed allocations.
AtB32 the planned289 IDs consume21,425,766,400 bytes/device for attention plus
snapshot pools, before live state/weights/workspaces. This is an estimate;
full-model allocator fit and actual prefix reuse remain required. Native context
is unchanged; the intended32x32768 working set may exceed the serving pool.

The reusable launch command is tests/serve_vllm.sh, called via docker exec.
Current inner loop adds QWEN38_SERVING_LAYERS=0,3, QWEN38_HOST_SAMPLING=1,
QWEN38_SERVING_EVENTS=<readiness_vllm/reduced_events.jsonl>, --stages serve and
--max-num-seqs2. It preserves262144 context, uses scheduler block4096,
mamba-cache-mode align, mamba-block-size4096, token budget8192 and decode-only
tracing. The optional host mode supports shared-test controls and intermediate
prefill compatibility; measured greedy token-out must show zero host-logit reads.

Two startup-only failures were fixed: bare architecture override attempted
generic upstream model interface inspection (use the documented TT-prefixed
execution override instead); then upstream align validation rejected the
plugin's unconditional disable_chunked_mm_input flag. The latter is now cleared
only for resolved text-only align adapters, with19 passing host tests. Failure
logs are reduced_hybrid_registration_failure.log and
reduced_hybrid_alignment_failure.log. Process audits were clean after both.

### Reduced hybrid lifecycle and sampling acceptance (2026-10-25)

The same reduced layers0/3, batch2, native262144 server completed nonaligned129-token HTTP traffic, changed chunk offsets0/4096/8192/12288, concurrent decode/prefill and multi-turn prefix reuse. `readiness_vllm/reduced_lifecycle.json` records4 baseline/final captures,0 retirements,6 concurrent-prefill events, and8192/8260 reused multi-turn prompt tokens. These are reduced interface checks, not full-model quality/performance evidence.

Attached shared runner `--stages sampling --sampling-profile smoke --max-num-seqs 2 --server-url http://localhost:8000` passed3 targets; all-vocabulary logprobs was skipped because server max-logprobs defaulted to20; logs archived as `reduced_sampling_runner.log` and `reduced_sampling_tests.log`. Server log archived as `reduced_hybrid_server.log`. SIGINT to owned runner1548799 completed clean shutdown; container process audit found no EngineCore/API/runner processes. No device reset was required.

Full-model matching qualitative controls launched with `docker exec qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh python -m models.autoports.qwen_qwen3_8_27b.tests.run_vllm_qualitative_controls`; output `full_model_controls.json`, log `full_model_controls.log`.

### Full-model control and B32 allocation attempt

`full_model_controls.json` completed9/9 prompt-matched controls with64layers, selected precision, nativecontext262144 and canonical token-out generation. Direct output review is in `full_model_control_verdict.json`: coherent/topic-correct, no loops/gibberish/unrequested language drift/contamination;3 outputs exhaust256 tokens including checkpoint thinking, so are incomplete answers rather than serving failures.

Started full B32: `docker exec -e QWEN38_HOST_SAMPLING=1 -e QWEN38_SERVING_EVENTS=/workspace/models/autoports/qwen_qwen3_8_27b/readiness_vllm/full_b32_initial_events.jsonl qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/serve_vllm.sh --stages serve --max-num-seqs 32`. `full_b32_initial_runner.log`; active `server.log` will be archived. This allocation attempt uses existing4096physicalprefill buckets. Forecast from Stage08 allocator evidence:~30.3GB/device steady plus~10.1GB4096prefill scratch exceeds34.2GB capacity. This forecast is not a measured physical-limit proof; internal microchunks are the first repair candidate if allocation fails. No context or concurrency reduction is authorized by that estimate.

B32 attempt failed at `PREPARE_PREFILL1280` (10:12:24UTC):222822400-byte DRAM request;8banks each required27852800bytes but largestfreeblock27849920bytes, allocated4198954176bytes/bank. Exact evidence `readiness_vllm/full_b32_initial_server.log`. API/EngineCore/runner all exited; processaudit clean, noreset. Repair: Generator accepts bounded internal `prefill_chunk_size` default4096; B32serving chooses1024, allocates/prepares onlythatphysicaldomain, and internally splits arbitrarylogicalprompts. Native262144context,289schedulerblocks and32slots remainunchanged. Actual repaired fit/correctness pending.

Microchunk device check: `probe_serving_continuation` withphysicalchunk128 passed193/129logicalprompts, PCC0.9999768/0.9999599 andNRMSE0.003044/0.003341, exactinactivepeer andzeroRNGadvance; `continuation_probe.json` and `continuation_microchunk_probe.log`. Original4096-domain artifact retainedas `continuation_probe_4096.json`.

`probe_serving_generator_contract` rerun aftermasked RNGrepair passedallreplicas, includingUINT32values above2^24/2^31, exactuntouchedcounters, zeronewcaptures, andstale-token/position/page-onlyrefresh controls. Evidence `generator_device_contract.json`, `generator_rng_device_contract.log`; originalartifactretainedas `generator_device_contract_pre_rng.json`.

Full B32retry launched with1024physicalchunks and explicit `--async-scheduling`, allothercontext/cache settings unchanged. `full_b32_runner.log`, `full_b32_events.jsonl`, current `server.log`.

### Full-model serving evidence (batch 32)

The repaired server reached health at 10:22:35 UTC with all 64 layers, context 262144, 289 scheduler blocks, 1024-token physical chunks, and four captures. Post-preparation DRAM allocation was 29,313,353,216 bytes/device; free was 4,731,160,064 bytes/device (allocator checkpoint, not peak). Command: `readiness_vllm/full_b32_command.txt`; raw events: `full_b32_events.jsonl`.

Initial full-model smoke: 2 passed, 1 skipped, 1 reproducibility failure. `full_b32_initial_sampling_tests.log` shows mixed penalty/device batch RNG ownership changed across orderings. Event analysis in `generator_autofix.md` confirms first run was entirely host while the second sampled one first token on device before penalty companions forced host sampling. The existing explicit `seeded_sampling_policy=host` will be used for seeded shared-test compatibility; final greedy performance remains canonical device sampling. The all-vocabulary logprob skip was caused by default max20; subsequent server commands allow `--max-logprobs -1`.

`run_vllm_server --stages qualitative --server-url http://localhost:8000 --max-num-seqs 32` succeeded. All twelve outputs were read: coherent, on-topic, no repetition loops, gibberish, unintended language drift or visible request contamination. Story and thermodynamics are token-limited, matching standalone controls. Artifacts: `vllm_qualitative_outputs.json`, `full_b32_qualitative_outputs.json`, `vllm_qualitative_verdict.json`, `full_b32_qualitative_runner.log`. Mechanical gate passed: `degenerate_output_check.json`.

`run_sampling_route_controls` passed: three device-only seeded request orderings reproduced exactly with zero host-read delta; six first-token top20 distributions and greedy tokens matched exactly under reordered host-logprob controls. Artifact: `sampling_route_controls.json`. This proves route-stable sampling and logit reproducibility; mixed host/device RNG streams are not claimed equivalent.

Started `run_serving_lifecycle.py --max-concurrency 32` against this same server with native default32769-token agent turns; output `full_b32_lifecycle.json`, raw requests and events alongside.

Lifecycle harness corrections: the first host-side invocation could not reach the container-local server and was stopped (`full_b32_lifecycle.log`); it did not issue model requests. The initial container fixture with32769-token agent turns was stopped before acceptance because observed8192-token prefill latency (~30s at fixedB32) projected an hour-scale stress run, contrary to the short-lifecycle scope. Client termination cancelled its active HTTP request; metrics confirmed running=waiting=0. Its events remain in the untruncated server event history; no server restart or counter reset occurred.

Final bounded lifecycle invocation uses fresh unseen logical lengths137/138/139 and8195/12291,256-token decode anchors,32 conversations of4130 tokens plus33 appended tokens. This measures the recorded typical4096 context at target concurrency32. The32768-per-agent planning working set remains a separate memory-ledger pool shortfall, not a measured workload or reduced advertised context. Command: `docker exec qwen38-functional-build python models/autoports/qwen_qwen3_8_27b/tests/run_serving_lifecycle.py --server-url http://localhost:8000 --events-path models/autoports/qwen_qwen3_8_27b/readiness_vllm/full_b32_events.jsonl --output models/autoports/qwen_qwen3_8_27b/readiness_vllm/full_b32_lifecycle.json --max-concurrency 32 --agent-turn-length 4130`; log `full_b32_lifecycle_typical.log`.

### B1 serving repair and benchmark retry

The first B1 server failed during preparation because identity remapping gathered a singleton sampler tensor and TT gather rejected its rank. Exact evidence: `b1_remap_failure_server.log`. The validated B1 mapping can only be[0], so `remap_state_slots` now invalidates resident metadata and returns without movement; B>1 preparation is unchanged. Eight host tests pass. Reduced B1 device evidence `generator_b1_device_contract.json` proves exact identity state/RNG, canonical sampling, resident stale-input equivalence, one page-only refresh and zero steady token/position refreshes.

The failed server runner escalated SIGTERM to SIGKILL; subsequent process audit found no API/EngineCore/runner leftovers. No device reset. Private-container IPC audit found670 orphan MPI `sm_segment` files owned byUID4123 and no live FD/mmap references. Cleanup freed49,397,760 allocated bytes; TT UMD locks and device metadata were retained. Evidence: `orphan_mpi_cleanup.json`.

Retried primary benchmark with `docker exec -e TT_METAL_TRACE_ALLOC_TRACKING=0 qwen38-functional-build bash models/autoports/qwen_qwen3_8_27b/tests/serve_vllm.sh --stages serve,benchmark --max-num-seqs 1 --no-benchmark-ci-serving`. Host sampling and event logging are unset; source/environment manifest is recorded in `full_b1_command.json`. Exact target remains4096 input/128 output/one request/concurrency1.

### Primary measurement and synchronous-control repair

B1 exact4096/128/1/1 benchmark passed: TTFT P50/P99=443.344ms; TPOT mean/P50/P99=27.601952ms; ITL P50/P99=27.624621/28.604692ms; aggregate output32.411012 tokens/s; TPOT-derived36.229321 tokens/s/user. Raw counts4096 input and128 output,1/1 completed. Artifacts `full_b1_vllm_benchmark.json`, `full_b1_vllm_result.json`, `full_b1_vllm_benchmark.log`, `full_b1_server.log`, `full_b1_command.json`. Shutdown audit: host compatibility false, host-logit reads0,127 model replays,128 sample replays, token/position refreshes1 each, page refreshes2 total. No remaining server processes. A final repeat is planned after subsequent adapter-only synchronous-finalization repair; this is a valid observed target result, not yet the final-source measurement.

The B8 non-overlapped page-growth control exposed a real callback gap: plugin synchronous finalization passes raw device tokens with `read_from_device=False,async_read=False`; adapter host formatting previously assumed a prior async local-replica read. Multi-device `to_torch` failed before accepting the control. Artifacts prefixed `sync_readback_failure_`; process audit clean after engine/API exit. Fix routes device storage through existing minimal read_decode_output, while already-read HOST outputs are formatted without another read. Three host tests pass; reduced adapter device check pending. The page-growth harness also had a local variable named concurrent shadowing concurrent.futures; renamed before requests, initial traceback retained.

Matching full64/B32 standalone logit baseline launched with `run_standalone_route_logits`:512-token correctness-only allocated cache, native model config262144 unchanged, six raw test prompts, no profiling/performance claim. Output `standalone_route_logits.json`.

### Full-model allocator-growth control

The synchronous callback repair passed the reduced adapter device test (`adapter_device_contract.json`, `adapter_sync_finalization_device.log`): raw device finalization equals async host finalization, one direct read for synchronous output, zero extra reads for already-host output. Full64/B8 non-overlapped control then passed (`page_growth_sync_control.json`): three concurrent distinct prompts4080/4079/4078,4160 output tokens each, six actual allocator boundary changes across4096 and8192, all page-only refreshes,4152 unchanged-table steps, four stable captures/zero retirements and zero host-logit reads. Exact command/config in `full_b8_sync_fixed_command.txt`; event history `full_b8_sync_fixed_events.jsonl`. Root SIGTERM to serving runner1644663 stopped API/EngineCore; process audit found none remaining; no reset. Shutdown log archived `full_b8_sync_fixed_server.log`.

Matching B8 async server launched with the same cache/config and explicit `--async-scheduling`; command `full_b8_async_command.txt`. Verification compares two forward/reversed concurrent waves token-exactly to the control. Explicit host compatibility is enabled for subsequent shared sampling tests; greedy growth requests must retain zero host-logit reads.

Async allocator-growth verification passed on full64/B8 (`page_growth_async.json`, `.requests.json`, `full_b8_async_events.jsonl`, command/environment/source hashes `full_b8_async_command.json`). All six forward/reversed wave outputs of4160 tokens are exactly equal to non-overlapped control; twelve page-only refreshes,8304 unchanged-table steps, four stable captures, zero retirements and zero host-logit reads. Log confirms actual overlapping submissions. Full shared sampling profile launched against the same live server with `serve_vllm.sh --stages sampling --server-url http://localhost:8000 --max-num-seqs 8 --sampling-profile full`; explicit compatibility mode and seeded-host policy retained for shared-test parameters.

Full sampling profile succeeded:76 passed,74 skipped,no failures in311.59s.71 skips concern unconfigured speculative decoding;3 chunked-prefill opt-in tests lack runner budget flag, with separate actual scheduler/mixed-shape lifecycle evidence. All-vocabulary logprobs passed. Artifacts `full_b8_sampling_tests.log`, `full_sampling_runner.log`, `sampling_verdict.json`.

Sampling-log anomaly classification: `full_b8_sampling_tests.log:186` records `httpx2.AsyncClient.aclose()` after the pytest event loop had closed, captured during a passing repetition-penalty test. Stack is entirely HTTP client teardown; all76 assertions passed, later requests completed, and server remained healthy. No model/device failure or device-holder leak followed. The independent reviewer re-derived the six token-exact growth comparisons from raw arrays.

### Batch16 validation

`serve_vllm.sh --stages serve,benchmark --max-num-seqs 16 --benchmark-prompt-len 129 --benchmark-output-len 8 --benchmark-num-requests 16 --benchmark-concurrency 16 --no-benchmark-ci-serving` passed16/16 requests,2064 input and128 output tokens. Tracking/event logging enabled, no host compatibility. This is capacity/nonaligned correctness validation, not headline performance. OwnB16 configuration:4096 physical chunk,177 scheduler blocks,native262144 context,selected precision. Four captures,zero host-logit reads; archived `full_b16_*` command/source manifest, events, raw/normalized result and server log. Runner shutdown completed; no API/EngineCore processes remained.

B16 shutdown classification: upstream abort shutdown uses timeout0 and logs forced EngineCore termination. AUTOPORT_SHUTDOWN_AUDIT prints, but no completed shutdown_release event exists. The subsequent process audit establishes no API/EngineCore device holders; it does not establish graceful trace/mesh teardown.

### Batch32 scheduler admission repair

The final tracked server uses `--max-num-batched-tokens 131072 --long-prefill-token-threshold 4096` with unchangedB32/physical1024/native262144/selected precision/cache289. `run_serving_lifecycle.py --max-concurrency 32 --agent-turn-length 4130` passed112 requests on onePID, four stable captures/zero retirements/zero host-logit reads; actual scheduler continuation starts4096/8192/12288 with concurrent decode. Cold32×4130 wave fell524.390→33.763s and64→4 prefill calls; warm32×4163 remained3.446s with99.1768% reuse. Whole diagnostic traffic255.429s,TTFTp50/p95/p99=4970.906/32686.050/32686.979ms,completed throughput0.438478req/s,SSEpausep99/max425.512/31959.208ms. These synthetic host-wall lifecycle metrics are not device or headline performance. Long-prefill decode stalls remain visible; serial cold-request admission overhead is removed. See `full_b32_budget131072_lifecycle.json`, rawrequests/events/command/source manifest/serverlog and offline hosttimeline.

Stopped diagnostic server; API/EngineCore process audit clear. Launched sameB32 config via shared runner `--stages serve,benchmark`, `TT_METAL_TRACE_ALLOC_TRACKING=0`, events/host sampling unset, to collect secondary CI100/100/32 plus nonheadline4096/128/1 at fixedB32. Manifest `full_b32_performance_command.json`.

### Batch32 tracking-off measurements

Shared runner `--stages serve,benchmark --max-num-seqs 32` with validated131072/4096 scheduler flags completed primary regression1/1 and CI32/32. Tracking0, events unset,host compatibility unset; exactsource/environment `full_b32_performance_command.json`. CI100input/100output/32requests/no explicit concurrency cap: TTFT P50/P99=2395.195/2395.813ms;TPOT mean/P50/P99=112.060/112.047/112.359ms;ITL P50/P99=109.197/204.297ms;aggregate output237.215tokens/s;TPOT-derived8.9238tokens/s/user is secondary burst context, not headline. Canonical `vllm_ci_serving_benchmark.json`/`vllm_ci_serving_result.json` plus full_b32_performance_ archives. FixedB32 single-request4096/128 regression: TTFT15088.975ms,TPOT133.159ms,7.5098tokens/s/user,3.9999aggregate outputtokens/s; not B1headline. Shutdownprocess audit no API/EngineCore left; noreset. FinalB1source-identical benchmark launched withtracking0 andhost/eventsunset.

### Final single-user result and closure audit

Final-source exact4096input/128output/B1/concurrency1 passed1/1: TTFT P50/P99=445.545612ms; TPOT mean/P50/P99=27.591066ms; ITL P50/P99=27.616491/28.495863ms; output throughput=32.404760tokens/s; decode1000/meanTPOT=36.243616332tokens/s/user. Source/environment/argv `final_b1_command.json`; exact raw+normalized `final_b1_vllm_result.json`/`final_b1_vllm_benchmark.json` (also canonical names), runner+server logs final_b1_*. Single-request quantiles are not population tail estimates.

Teacher forcing203/100/B1 from Stage08 selected policy36.381tokens/s/user is only a decoder latency lower-bound reference, not a serving or comparable headline measurement. Matching Stage08 autoregressive4096/128/B1 median36.170903tokens/s/user,431.263307ms TTFT is separate token-out evidence (`doc/datatype_sweep/performance.json`, selected/benchmark_b1_perf.json). Final serving adds no measured material decode overhead relative to that token-out baseline; runtime audit documents removed padding and resident traced state.

`final_cleanup.json`: no live serving processes or open TT device descriptors in the container; host accessible process/descriptor scan also empty,670host process directories inaccessible. One earlier defunct EngineCore entry adopted by containerPID1 holds no descriptors; it cannot hold devices. Upstream abort/force termination is recorded, not represented as graceful teardown. No resets or profiler runs. Initial independent stage-review found only then-pending final evidence; final rereview follows report reconciliation.

### Independent acceptance and local checkpoints

Fresh xhigh Stage Review returned **clean-pass** after final raw/count/source/telemetry/cleanup rereview; `stage_review.md`. Initial findings were missing final artifacts and stale metadata, all reconciled without runtime changes. Plugin pre-commit passes; source/doc whitespace check passes (raw logs preserve original whitespace). No additional model rerun requested.

Plugin checkpoint: repo `bringup/dependencies/vllm-tt-plugin`, branch `main`, commit `d5ee19c6754905e7a9a69c79d9e2ee4b99b84295`. Explicit pre-commit run before commit. Repository had no configured author; per-command Codex <codex@localhost> matches prior tt-metal bringup commits; no global identity changed. Nothing pushed.

tt-metal checkpoint: repo `/home/mvasiljev/tt-metal`, branch `exp26-qwen38-bringup`, commit `425000e4bd982a48db43a6963e03359d07b509b0`. Stage-owned implementation, tests, reports and raw readiness evidence committed locally after clean-pass. This following documentation-only receipt records the checkpoint SHA. Plugin commit `d5ee19c6754905e7a9a69c79d9e2ee4b99b84295`. Neither repository was pushed. Mandatory telemetry packet updated with final measurements and these checkpoint references.

### Runner gate remediation (2026-10-25)

Reproduced the supplied 09-vllm.check.sh failure with MODEL_DIR=models/autoports/qwen_qwen3_8_27b and HF_MODEL=Qwen/Qwen3.8-27B: exit 2; readiness_vllm/runner_gate_reproduced.log. The three critical findings all refer to context_length=256 inside preparation_events_host_check.json. Its explicit scope is host-only AST-extracted prepare_decode with fake tensor/trace operations, one fake layer, and no hardware validation. These are synthetic preparation-event fixtures, not served-context settings. The existing preparation_events.md independently describes that origin.

The checker recursively treats every context key in readiness_vllm JSON as serving configuration and does not inspect scope. Concrete counterevidence: final_b1_command.json argv sets --max_model_len 262144; final_b1_server.log records the full model configuration; doc/context_contract.json records supported_context=262144. No production context reduction occurred. Preserved the fixture byte-for-byte at tests/evidence/preparation_events_host_check.json (SHA256 c5aec0171f39c95b35bbd59b98a3391ea557a7fe4b190949f3efbe361c3b7a3e), and updated both documentation references. This separates synthetic test evidence from serving measurements without altering its values, deleting failure evidence, changing runtime code, or weakening the installed checker. Existing plugin_state_after.log advisory caps are host plugin tests, not final server configuration; retained unchanged.

Validation command: export TT_AUTODEBUG_ROOT=/home/mvasiljev/tt-metal/.codex/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.8; eval "$(python .codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21/scripts/environment.py)"; MODEL_DIR=models/autoports/qwen_qwen3_8_27b HF_MODEL=Qwen/Qwen3.8-27B bash "$TT_MODEL_BRINGUP_ROOT/prompts/model_bringup_multigoal/09-vllm.check.sh". Enabled dependency verified in .codex/config.toml. No hardware commands, new benchmarks or profiling: runtime implementation and existing full-model acceptance evidence are unchanged.

Reverification result: unchanged 09-vllm.check.sh exits **1 (advisory)**, not 0; readiness_vllm/runner_gate_remediated.log. Degeneracy and context checks pass; no critical findings remain. check_agentic_serving.py explicitly returns 1 for documented copy_reason and pool_shortfall_reason. Existing six duplicate-weight-family reasons and the configured 786432-token pool versus estimated 32x32768 working set cause that advisory result. The agentic-serving skill explicitly permits explained shortfalls as advisory. These are retained honestly rather than changing estimates, removing weight accounting or pretending the pool is larger. This remediation uses the user-authorized concrete-refutation route for the original critical report, not a claim that the aggregate checker returned 0.

Additional verification: final_b1_server.log lines 22 and 48 independently record API max_model_len and EngineCore max_seq_len of 262144; full_b32_performance_command.json also uses 262144. `git show HEAD:<original fixture path>` equals the relocated bytes exactly. Raw final_b1_vllm_result.json confirms one completed request, 4096 input and 128 output tokens. Wrote required local packet `telemetry/packets/b0421e58-420d-49e8-acf6-5bb002034c03.json` under the supplied run directory using its exact template identifiers, existing 445.545612 ms TTFT and 36.243616 tokens/s/user; device-time/roofline values remain null with reasons.

Fresh independent xhigh stage-review returned **clean-pass** for the remediation: `runner_remediation_review.md`. Reviewer independently ran the unchanged checker (exit1, zero critical findings), verified fixture byte identity, both native-context server manifests and all44 source hashes in each, and raw-derived telemetry/template identity. No runtime rerun required. This is acceptance of the concrete refutation, not an aggregate exit0 claim.
