# Qwen3.8-27B vLLM integration — stage 9

**Primary native single-user S128/G128/N1: TTFT P50 85.82 ms; decode 41.3053
tokens/s/user.** Max sequences1, concurrency 1, greedy temperature 0, ignore EOS,
one unmeasured same-shape warmup. Full sampling compatibility profile: **73/73
passed** at max sequences 32. Native qualitative outputs are coherent and on topic;
sampled haiku meter remains imperfect, as in the prior selected-policy control.
See [the inspected quality report](qualitative_review.md).

| Primary metric | S128/G128/N1, max sequences 1, concurrency 1 |
| --- | ---: |
| TTFT P50 / P99 | 85.8247 / 85.8247 ms |
| TPOT P50 / mean / P99 | 24.20998 / 24.20998 / 24.20998 ms |
| ITL P50 / P99 | 24.19049 / 25.33166 ms |
| Aggregate output throughput | 40.4966 tokens/s |
| Decode `1000 / mean TPOT` | 41.3053 tokens/s/user |

One measured request completed all 128 requested output tokens. TTFT and TPOT
P99 therefore equal the single observed value; these are not population tail
estimates. Raw measurements are `readiness_vllm/vllm_result.json`; the exact
benchmark command, config and normalized metrics are in
`readiness_vllm/vllm_benchmark.json`, with console `vllm_benchmark.log`.
`readiness_vllm/primary_run_config.json` records server command, source hashes,
precision, native-only environment and warmup count. No host compatibility,
allocation tracker or profiler is enabled in this primary result.

The previous full-model teacher-forcing B1/S203/G100 result (40.878 tokens/s,
76.607 ms TTFT) is only a decoder/generator latency lower-bound reference; its
workload and output boundary differ from serving. The more closely shaped
canonical token-out B1/S128/G128 reference is 41.0955 tokens/s, 58.4179 ms TTFT
without per-token readback, or 41.0750 tokens/s, 58.9856 ms TTFT with deferred
complete-token delivery (`doc/datatype_sweep/selected_token_out.json`). These
are different harnesses, not a claim that serving improved model speed. The
primary decode rate is within about 0.6% of those canonical references; serving
TTFT also includes request handling and eager prefill. The audited path retains
nonblocking split traces and device token/position/seed feedback, with minimal
async token reads and no unchanged-table device copies. No avoidable vLLM-specific
decode overhead remains identified in this measured path. No decoder retuning or
serving profiler collection was needed.

## Serving configuration

All 64 layers use selected `head_bfp4_lofi` policy from
[the datatype selection](../datatype_sweep/selected_precision_config.json), pinned
HF revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The mesh is Blackhole
TP4 `(1,4)` (`P300x2` CLI), with 8192-byte fabric payload and decode traces.

| Policy group | Selected value |
| --- | --- |
| Attention/output/gate/up/down/head weights | BFP4 |
| Those six compute fidelities | LoFi, FP32 destination accumulation enabled |
| Attention/MLP activations, residuals, CCL | BF16 |
| Attention KV cache | BFP8 |
| Logits, sampling, norms, embeddings, convolution | BF16 |
| Recurrent state | FP32 |
| Sensitive operations / final norm | HiFi4 / HiFi2 |
| Layer exceptions | Empty, exactly as selected |
| Token feedback | UINT32 |

The served per-request context is **262144**, matching
[context_contract.json](../context_contract.json); there is no capability reduction.
The B32 server allocates 8224 attention pages of 32 tokens, including 32 extra allocation-slack
pages, for a shared 263168-token physical pool. This supports one maximum-context
request or many shorter requests; it does not promise 32 simultaneous full-context
allocations. Full-context execution evidence belongs to the prior full-model
contract. This stage exercises serving with non-aligned prompts and up to 32
concurrent short requests.

`TTQwen38ForCausalLM` is registered in the sibling vLLM repository's
`plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models()`.
The adapter delegates prefill, decode, sampling parameters, recurrent-state
reset/remap, and trace lifecycle to the full-model generator. vLLM calls cache
allocation and passes that exact `ModelCache` object through every request; serving
never calls standalone `_ensure_cache`. Attention pages are scheduler-owned.
Constant-size convolution/recurrent state follows requests through fixed slots.

Capabilities are `supports_async_decode=True`, `supports_sample_on_device=True`,
and `supports_prefix_caching=False`. Native decode replays the canonical model
and split sampling traces nonblocking, writes sampled tokens back to the persistent
device token input, and advances device positions/seeds once per emitted token.
Only one replica's 32 UINT32 token vector is read for scheduler output. There is
no native host argmax, full-logit readback, replacement sampler, or Python token
read/write feedback loop. Page values are compared every call because allocator
growth can occur with `reset_batch=False`; unchanged tables cause no device copy.

## Explicit sampling modes

| Environment | Behavior and intended evidence |
| --- | --- |
| `QWEN_VLLM_HOST_COMPATIBILITY` unset | Native traced performance path; unsupported host-only requests fail explicitly |
| `QWEN_VLLM_HOST_COMPATIBILITY=1` | Supported batches remain native; unsupported top-k/penalties/logprobs/structured requests use existing plugin host compatibility |
| `QWEN_VLLM_HOST_COMPATIBILITY=all` | Explicit host-only shared compatibility suite; one RNG backend prevents batch neighbors from changing seeded sampling algorithms |

All server commands retain TT `sample_on_device_mode=all`. The optional model
capability hook routes compatibility requests to the plugin's existing host
sampler. Host-only suite results are not native performance measurements.
Native stochastic sampling supports k1..32; greedy requests use canonical k1.
Explicit native seeds are normalized below the int32 limit and anchored to absolute
output positions only at authoritative refreshes. Steady device seed progress is
preserved through parameter changes. This does not promise identical random
streams between TT and PyTorch samplers; see [AUTOFIX_native_sampling.md](AUTOFIX_native_sampling.md).

## Correctness evidence

All paths below are relative to `models/autoports/qwen_qwen3_8_27b/`.

| Evidence | Result and scope |
| --- | --- |
| `readiness_vllm/sampling_tests.log` | Final full shared profile, 73 passed in 973.88s; explicit optional all-host compatibility, max sequences 32 |
| `doc/vllm_integration/adapter_host_after_seed_fix.log` | 39 host tests pass, with 16 subtests; real adapter/plugin prefill, cache, async and seed boundaries |
| `doc/vllm_integration/adapter_device_after_seed_fix.json` | Traced layers 0/3 structural control passes 68 steps, stale tokens/positions, changed/unchanged pages, drained remap at step 36, inactive-state preservation and owned pending snapshots; allocation tracker enabled |
| `readiness_vllm/full_concurrent_control.json` | All 64 layers; four distinct chat prompts S67/69/66/69,G100; all 12 text and exact token-ID streams match native async, synchronous host-greedy control, and reordered native repeats; page boundaries96/128/160 |
| `readiness_vllm/native_seed_continuity.json` | Native k5,T0.7,p0.9,seeds42/43; A(S67,G100),B(S69,G47) alone, B(G33)+A(G100), A(G100)+B(G47); all6 token-ID comparisons pass |
| `doc/vllm_integration/native_seed_server.log` | No explicit-host marker during that native stochastic run |
| `readiness_vllm/full_logit_determinism.json` | Full-vocabulary normalized prefill logits: two prompts, repeats and reversed concurrent submission, four maximum differences0 |
| `readiness_vllm/standalone_logit_control.json` | Same selected all64-layer model; explicitly swapped slots0/1, standalone cache128; all four serving-vector differences0 |
| `readiness_vllm/full_lifecycle.json` | Full-model S31/33/127/129/31/33,G70 warm/new/repeated shapes all complete; serving cache usage0 before/after |

The reduced active streams collapse to token220, so their equality alone is a
weak numerical oracle. Their state/counter assertions remain useful; the diverse
full-model exact-token comparison supplies request-sensitive serving evidence.
HTTP submission order does not force internal row placement. Direct adapter tests
force the permutation; the standalone logit control explicitly swaps slots.
The full-logit comparison covers prefill, while shared logprob tests and the
100-token synchronous control exercise decode.

The final device probe performs 68 model/sampling replays and 68 minimal reads,
with only 2 token/position/RoPE/seed refreshes (initial binding and drained remap),
6 changed-page refreshes, and 14 distinct owned snapshots among 68 retained snapshots.
No unchanged-step seed, token, position or page-table device copy is required.

## Secondary CI serving-burst performance

These are native, uninstrumented results at `max-num-seqs=32`, S100/G100/N32,
unbounded burst admission, zero warmup requests, greedy temperature 0, and
`ignore_eos=True`. All 32 requests and 3200 requested output tokens completed.
This is capacity/nightly context, not headline single-user decode throughput.

| Metric | S100/G100/N32, max sequences 32 |
| --- | ---: |
| TTFT P50 / P99 | 6450.50 / 6451.94 ms |
| TPOT P50 / mean / P99 | 200.61 / 201.67 / 224.13 ms |
| ITL P50 / P99 | 186.79 / 638.75 ms |
| Aggregate output throughput | 121.62 tokens/s |
| `1000 / mean TPOT` | 4.9586 tokens/s/user, burst affected |

Exact command and workload configuration:
`readiness_vllm/vllm_ci_serving_benchmark.json`. Raw measurements:
`readiness_vllm/vllm_ci_serving_result.json`; console:
`readiness_vllm/vllm_ci_serving_benchmark.log`. This run has vLLM chunked prefill disabled, and S100 is below the internal
4096-token chunk bound. Admission, first-use work and fixed-batch32 decode affect
this burst result; do not compare its derived value to batch1 teacher-forcing
throughput. Chunked prefill can additionally affect other burst configurations.

A secondary cold single-user capacity control, S128/G128/N1 at max sequences 32,
records TTFT623.12ms, mean TPOT224.03ms, ITL P50/P99212.30/213.26ms, aggregate
4.4024tokens/s and TPOT-derived4.4637tokens/s/user. It includes first-use shape/
trace work and is preserved as `readiness_vllm/vllm_batch32_single_user_*`.
The primary measurement uses max sequences 1 and one unmeasured
same-shape warmup; the shared benchmark's default endpoint check skips warmup.

## Commands and ownership audit

Source `../run-env.sh` from the tt-metal root. The stage-owned wrapper only sets
the pinned environment and invokes the installed, unmodified shared runner through
`tests/vllm_process_guard.py`. Native B32 serving used:

```bash
QWEN_VLLM_MAX_SEQS=32 QWEN_VLLM_HOST_COMPATIBILITY=1 \
 bash models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh \
 --stages serve --sampling-profile full
```

Final full compatibility and native B1 commands:

```bash
QWEN_VLLM_MAX_SEQS=32 QWEN_VLLM_HOST_COMPATIBILITY=all \
 bash models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh \
 --stages serve,sampling --sampling-profile full

unset QWEN_VLLM_HOST_COMPATIBILITY QWEN_VLLM_TEST_LAYERS
unset TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_TRACEBACKS
export QWEN_VLLM_MAX_SEQS=1
bash models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh \
 --stages serve --sampling-profile full
# Attach from another shell using the same environment:
bash models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh \
 --stages qualitative --server-url http://localhost:8000 --sampling-profile full
python models/autoports/qwen_qwen3_8_27b/tests/check_vllm_qualitative_extended.py \
 --output models/autoports/qwen_qwen3_8_27b/readiness_vllm/qualitative_extended.json
bash models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh \
 --stages benchmark --server-url http://localhost:8000 --sampling-profile full \
 --no-benchmark-ci-serving --additional-benchmark-args='--num-warmups 1'
```

The native B32 CI benchmark was attached with `--stages benchmark --server-url
http://localhost:8000 --sampling-profile full` and default CI settings; its raw
summary records both the exact CLI and S100/G100/N32 workload.

The wrapper passes `--mesh-device P300x2 --max-model-len 262144 --block-size 32`,
`--max-num-seqs <selected value>` and this TT configuration (the shared runner adds
`sample_on_device_mode: all`):

```json
{"trace_region_size":134217728,"fabric_config":"FABRIC_1D_RING","fabric_max_packet_payload_size_bytes":8192,"trace_mode":"decode_only"}
```

Additional server arguments are
`--hf-overrides '{"architectures":["TTQwen38ForCausalLM"]}' --no-enable-prefix-caching --max-logprobs -1`.
Exact expanded server/runner commands, source hashes and environment are in
`readiness_vllm/full32_run_config.json`. No Tracy, watcher or device profiler is
used for serving performance; allocation tracking is disabled in benchmark runs.

The shared runner's startup-cancel path could orphan EngineCore. The local guard
marks only its own descendants, uses process start times/pidfds, adopts/reaps
orphans, and applies bounded INT/TERM cleanup without touching unrelated processes.
Seven guard host tests include cancellation, actual packaged-shutdown reproduction,
PID reuse, unrelated sentinel survival, and a TERM-resistant report. The TT worker
now drains async work, closes the model, then closes its mesh explicitly; six
worker tests cover normal, exceptional and partial initialization shutdown.

Serving cache usage returns to 0 after full-model lifecycle and concurrent controls.
Attention/recurrent buffers are fixed for the configured serving pool; request-local
prefill activations are transient and chunked at 4096. Compiled programs and their
persistent buffers warm by shape, and new shapes release conflicting traces before
warmup. These counters are vLLM cache occupancy, not a full TT DRAM allocator audit;
short repeats do not prove every possible long-running shape sequence is bounded.

The final host-profile log also emits the generic active-trace allocation warning
(`full_host_profile_server.log:166`). Allocation alone is not proof of corruption:
request-local prefill intermediates are released before decode replay. Persistent
first-use packed-prefill program buffers were a real tracker-detected defect and
are now warmed after releasing existing traces; see
[AUTOFIX_prefill_sampling_trace.md](AUTOFIX_prefill_sampling_trace.md). The repaired
representative smoke and final 68-step adapter probe pass with trace allocation
tracking enabled; earlier all-layer tracked sampling reached 52 passing cases
before an external interruption. The final 73-test all-layer suite is uninstrumented
and passes, but is not a complete all-shape allocation-tracker audit. The warning
is retained as an evidence limit, not silently declared proof of safety.

Earlier startup Ethernet failures were confounded by host telemetry ownership.
The operator paused kubelet for this reservation; subsequent explicit shutdown and
mesh reopen succeeded without reset. Do not infer that the worker fix alone proved
the heartbeat fault's cause. Historical recovery and failed triage-capture details
remain in [work_log.md](work_log.md) and the linked AutoFix reports. Nanobind prints
reference-leak warnings at Python exit; observed worker close markers, process
exit, and empty device-owner lists are the cleanup evidence.

The benchmark CLI now tolerates missing distribution metadata in a source checkout
by falling back to the package version string. The first metadata-only setup attempt
found missing `setuptools_scm`; no compiler or dependencies were installed.

## Final gates and artifacts

The installed `09-vllm.check.sh` exits 0: no degenerate shared outputs and context
contract 262144 preserved (`stage_check.log`). The native B1 server finishes with
zero host-fallback markers and serving cache occupancy 0. Both native B32 and B1,
the all-host full-profile server, standalone logit control and reduced device
probe close their devices. `readiness_vllm/final_process_audit.json` records
empty device-owner files and no surviving B1 guard/runner/API/EngineCore PIDs.
Held-server wrappers exit 130 after intentional guard SIGINT; check runners and
the full launch/sampling/shutdown flow exit 0. Worker close markers and guard
cleanup-complete markers are preserved in the archived server/runner logs.
Final bounded device list and exact ring mesh-open/close evidence are in
`final_device_list.log` and `final_mesh_smoke.log`; no final reset is needed.

The independent [stage review](stage_review.md) returns **clean-pass**, with no
required work. Local commit provenance is recorded in [work_log.md](work_log.md).
Raw logs/JSON/vectors remain in ignored persistent artifact directories. Source,
tests and compact Markdown reports are committed; operator-owned root
`AUTODEBUG.md` and `PIPELINE_BLOCKERS.md` are excluded. This stage never pushes.
