# Qwen3.8-27B optimized vLLM — stage10

**Primary native S128/G128/N1: warmed TTFT 62.95 ms and decode 41.313 tokens/s/user.**
Max sequences1, concurrency1, greedy temperature0, ignoreEOS, one same-shaped
warmup. Matched baseline: 81.37 ms and 41.322 tokens/s/user. TTFT falls 22.6%;
decode is effectively unchanged and within 0.6% of the selected full-model
S128/G128/B1 token-out reference. All 64 layers run through the real TT vLLM
plugin and `tt/generator_vllm.py`.

**73/73 full sampling compatibility tests pass** at max sequences32. Native
qualitative, concurrent async/seed/lifecycle checks and both benchmark profiles
pass. Final process audit finds no vLLM/EngineCore/owned runner left; the bounded
health listing sees all four chips. Independent [stage review](stage_review.md) returns **clean-pass**, with no
required work.
See [the inspected quality report](qualitative_review.md).

| Metric | Before: S128/G128/N1, max sequences 1 | After: S128/G128/N1, max sequences 1 |
| --- | ---: | ---: |
| TTFT P50 / P99 ms | 81.37337 / 81.37337 | 62.95475 / 62.95475 |
| TPOT mean / P99 ms | 24.20018 / 24.20018 | 24.20525 / 24.20525 |
| ITL P50 / P99 ms | 24.19366 / 24.87232 | 24.18633 / 24.59560 |
| Aggregate output tokens/s | 40.56899 | 40.79966 |
| Decode 1000 / meanTPOT, tokens/s/user | 41.32200 | 41.31335 |

One measured request delivered128 tokens after one unmeasured warmup. TTFT and
TPOT P99 equal that sole request; they are not population-tail estimates. Raw
[before](before/primary_vllm_result.json) and [after](after/primary_vllm_result.json)
JSON, normalized `primary_vllm_benchmark.json`, console logs and server logs are
retained in those directories. Command arrays and workload configs match exactly.
The earlier `candidate_primary/` run is intermediate evidence, not the final result.

## Serving contract and implementation

All runs retain context 262144, block32, Blackhole TP4 mesh `(1,4)` / CLI `P300x2`,
Ring fabric payload8192, trace reservation134217728bytes/device, `trace_mode=decode_only`
and `sample_on_device_mode=all`. Primary max sequences1 and CI max sequences32
are separate profiles with matched before/after settings. HF revision is
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

The selected `head_bfp4_lofi` policy remains unchanged: attention/output/gate/up/down
and head weights BFP4 with LoFi/FP32 destination; activations/residuals/CCL BF16;
KV BFP8; recurrence FP32; norms/embeddings/logits/sampling/convolution BF16;
sensitive operations HiFi4 and final norm HiFi2. Runtime logs print the policy;
[retained stage8 runtime tensors](../datatype_sweep/selected_confirmation.json),
source consumption and unchanged model/decoder establish the actual uploaded
policy. The confirmation was restored byte-for-byte from local stage8 git object
`3f48cac393a`; [restoration provenance](restored_evidence.json) records its hash.
`propagation_check.json` is an inherited summary, not a per-layer upload audit.

The generator now owns serving prefill output and first-token sampling traces.
One retained prompt shape through 4096 at slot0/start0 reuses the full-model
prefill trace and a fourth canonical sampling trace. Long, multirow and nonzero-slot
prefills use existing chunked computation and pack logits into one persistent
sampling input. All temporary outputs die before sampling replay. Warmed sampling
is traced in both branches; new shapes/programs warm once before capture. Public
independently owned logits APIs retain their ownership contract.

Decode still replays separate model and canonical split-sampling traces with
`blocking=False`. Physical TopK32 and gathered candidates implement semantic
greedy k1,p0,T1; no adapter argmax or generic greedy sampler is introduced.
Token/position/RoPE/cache/page/sampler tensors stay persistent. Scheduler changes
refresh authoritative inputs; device feedback advances tokens, positions and seeds.
Unchanged page tables cause no device copy. Serving binds the exact external vLLM
cache and never allocates a standalone pool.

Actual native logs enable `async_scheduling=True`. The adapter returns device
tensors for `read_from_device=False`, defers one replica's32 UINT32 token read
with `async_read=True`, and formats host output after its completion event.
The primary warmup+measurement counters show254 model and254 sampler decode
replays, one cold prefill/sample followed by one replay of each, two token/position/
RoPE refreshes and256 minimal token reads. No native full-logit read or host
compatibility marker occurs. First-use eager warmup is outside the primary measurement.

## Secondary CI serving-burst capacity

| Metric | Before: S100/G100/N32, max sequences 32 | After: S100/G100/N32, max sequences 32 |
| --- | ---: | ---: |
| TTFT P50 / P99 ms | 3287.20327 / 3288.33937 | 3372.12224 / 3373.81323 |
| TPOT mean / P99 ms | 201.80040 / 221.78248 | 200.82763 / 221.32456 |
| ITL P50 / P99 ms | 186.79139 / 647.94197 | 186.89214 / 609.08773 |
| Aggregate output tokens/s | 138.09036 | 138.16995 |
| Decode 1000 / meanTPOT, tokens/s/user | 4.95539 | 4.97939 |

The unbounded burst completes32/32 requests and3200 tokens before and after.
Both runs first execute the runner's cold B32 S128/G128/N1 check, then the burst
with zero explicit warmups. Aggregate throughput is effectively unchanged;
cold-burst TTFT rises 2.58%. This is capacity/nightly-parity evidence, not a burst
speedup or headline single-user decode rate. Cold first-use prefill/sampling work
is included here; steady decode sampling remains traced. Raw artifacts:
`before/` and `after/batch32_vllm_ci_serving_{result,benchmark}.json`, corresponding
logs and `batch32_server.log`. The preceding B32 single-request result is also
preserved, but is not used as the primary latency result.

## Full-model comparison

| Selected full 64-layer reference | S128/G128/B1 TTFT ms | S128/G128/B1 decode tokens/s/user |
| --- | ---: | ---: |
| Queued token-out, final synchronization, final token checked outside timing |58.41793|41.09546|
| Deferred complete delivery, final history transfer included |58.98559|41.07500|
| Native vLLM, streaming minimal async reads |62.95475|41.31335|

Reference: [selected_token_out.json](../datatype_sweep/selected_token_out.json).
These share selected math and sampling but differ in prompt data, cache geometry,
request handling and delivery boundary. They establish comparable decode speed,
not a measured model-speed improvement from vLLM. Teacher-forcing S203/G100 has
reference-token upload and is intentionally excluded from the headline comparison.

## Validation, limits and reproducibility

- Full 64-layer tracked exact eager/traced prefill logits and tokens pass at31,33,128,129,
  4095,4096,4097, including changed prompts/pages and stable persistent identities:
  [prefill_full.json](prefill_full.json).
- Reduced B4 mixed rows31/45 in slots1/3 and S4097 fallback prove warm sampler
  reuse and temporary lifetime; separate four-chip Watcher checks pass:
  [fallback_reduced.json](fallback_reduced.json), [watcher_prefill.json](watcher_prefill.json).
- The68-step reduced adapter test covers changed token/current position, physical
  page growth/remap, pending async output ownership and inactive state:
  [adapter_final.json](adapter_final.json). Reduced outputs are structural evidence,
  not model quality.
- Full 64-layer B32 tracked HTTP controls pass12/12 distinct native/synchronous/reordered
  comparisons and6/6 seeded continuations. Sequential and concurrent nonaligned
  lifecycle checks each complete6/6 requests and leave running/waiting/KV usage0:
  `concurrent_control.json`, `seed_continuity.json`, `lifecycle*.json`.
- All 59 host tests pass; Black checks and Python compilation pass. This is a
  Python/docs-only change; no C++/CMake build is needed.
- [context_contract.json](../context_contract.json) preserves262144 with no public
  alignment rule. At most8,421,376 additional persistent bytes/device and one
  retained trace shape fit the existing capacity budget. The shared vLLM pool
  supports one maximum-context request or many shorter requests, not32 simultaneous
  full-context allocations. Full-capacity execution is inherited from the unchanged
  full-model context path; this stage validates the added buffers and nonaligned tails.
- No Tracy, tt-perf-report, live-server/adapter device profiling or ReadDeviceProfiler
  was collected. Device time and roofline/utilization remain unreported.
- Native stochastic sampling supports k1..32 without penalties/logprobs. Explicit
  compatibility modes retain the completed integration's broader API behavior;
  their results must be separated from native sampling and performance evidence.
- Baseline and final B32 nanobind teardown warnings occur after mesh close; they
  are not a claim of a leak-free allocator. Active-trace allocation advisories are
  covered by separate allocation tracking and ownership tests. See warning
  classification in [work_log.md](work_log.md).

Exact launch/test commands, rejected options and artifacts are in
[work_log.md](work_log.md). The [serving checklist](serving_checklist.md) maps
relevant optimize requirements to evidence; [perf_summary.json](perf_summary.json)
retains normalized matched measurements. [AutoDebug](AUTODEBUG_prefill.md) and
[AutoFix](AUTOFIX_prefill.md) record the ownership diagnosis and repair experiments.

Full sampling logs: `full_sampling_tests.log` and `full_sampling_runner.log`,
exit0,73 passed with no skips/xfails; the three pytest warnings concern the
unpackaged source version and SWIG deprecations. The explicit all-host mode is
a compatibility check, while every reported performance number is native.
Final [process audit](final_process_audit.json) and `final_device_health.log`
record clean shutdown and the bounded four-chip health listing.
