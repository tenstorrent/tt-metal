# Qwen3.6-27B vLLM integration

**Primary single user: TTFT275.75 ms; decode37.63 tokens/s/user —128 input /128 output /1 request, concurrency1, greedy, all64 layers.**
Decode trace replay is enabled; allocation tracking and detailed telemetry are
disabled for performance. All serving acceptance checks pass after the canonical
sampling repair. Stage 9 is complete: independent [round 2 review](stage_review_round2.md)
returned **clean-pass**, with no required work remaining. The aggregate gate has
no critical failures (exit1: advisory-only serial-reference/history diagnostics).

|Workload(input/output tokens per request)|Completed|TTFT P50/P99 ms|TPOT mean/P50/P99 ms|ITL P50/P99 ms|Output tok/s|1000/meanTPOT t/s/u|
|---|---:|---:|---:|---:|---:|---:|
|Primary128/128,1 request, concurrency1|1|275.75/275.75|26.572/26.572/26.572|26.568/27.369|35.060|37.633|
|Secondary CI100/100,32 requests, uncapped admission|32|3759.77/6121.49|100.748/100.262/123.880|56.229/1141.899|229.094|9.926|
|Serial prefill128/32,32 requests, concurrency32|32|3985.53/6704.32|169.077/166.326/248.821|56.262/1584.187|107.533|5.914|

Every profile uses random prompts, temperature0 and ignore_eos=True. The primary
has one measured request, so request-level P50/P99 coincide; it is not a tail
latency study. CI and prefill-burst TPOT include admission and competing chunked
prefill. Their derived t/s/u values are secondary, not headline decode speed.
Commands, configs and raw paths are in [primary](../../readiness_vllm/vllm_benchmark.json),
[CI](../../readiness_vllm/vllm_ci_serving_benchmark.json) and
[prefill burst](../../readiness_vllm/vllm_prefill_burst_benchmark.json).
Raw files: [primary](../../readiness_vllm/vllm_result.json),
[CI](../../readiness_vllm/vllm_ci_serving_result.json),
[prefill](../../readiness_vllm/vllm_prefill_burst_result.json).

The prior full-model128/128/B1 generator reports75.38 ms TTFT and37.82 t/s/u.
Its teacher-forced37.65 t/s/u is only a lower bound on decoder cost, not a serving
throughput comparison. Current serving decode is37.63 t/s/u; serving TTFT includes
eager scheduler prefill. The canonical sampler is shared with the full generator.
The [performance audit](../../readiness_vllm/p2_perf/benchmark_audit.json) records
282 device/0 host forwards, no OOV or errors. There is no measured-path host
argmax, full-logits readback, Python token-feedback loop or recapture. Resident
decode refreshes page tables only when mappings change. No additional decode
optimization sweep was performed after closing the required gate.

The final configuration is all64 layers, P300x2/1×4 Blackhole chips0–3,
max-model-len262144, max-num-seqs32, page32,8224 physical pages, chunk budget512,
serial prefill, async scheduling, trace region100000000, FABRIC_1D, trace mode
all, device sampling all and approved sampler ownership policy request.
Checkpoint defaults are T1/top_p.95/top_k20. On-device stochastic top-k1..64 is
now supported; greedy normalizes to1. Optional QWEN_VLLM_HOST_COMPATIBILITY=1
admits unsupported shared-test/diagnostic features. Performance stays on device.

The [adapter](../../tt/generator_vllm.py) delegates preparation, prefill, decode,
sampling and cleanup to QwenGenerator. supports_async_decode=True uses deferred
host readback while recurrent token feedback stays resident. vLLM owns the
supplied attention KV cache; its identity and mappings pass through unchanged.
Fixed-size GDN state remains model-owned. The entire
[selected precision](../datatype_sweep/selected_precision_config.json) is loaded:
BF4 MLP/BF8 attention and head, BF16 activations/explicit CCL, BF8 KV, selected
LoFi/HiFi2 fidelities and layer0/63 exceptions. The current
[runtime check](../../readiness_vllm/p2_final_async/runtime_contract.json)
confirms the complete config equality,64 runtime layers, cache and registration.

The stack is [tenstorrent/vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin/tree/042a94863ad27d88207bcf71abdf67122c6f4c27)
and upstream [vLLM0.31.0](https://github.com/vllm-project/vllm/tree/v0.31.0),
as recommended by that plugin revision's docs/install-vllm-tt.sh; installed
version0.31.0+empty. Upstream vLLM is unchanged. Runtime class override
TTQwen3_5ForConditionalGeneration=models.autoports.qwen_qwen3_6_27b.tt.generator_vllm:QwenVLLM
resolves in EngineCore to this autoport. See [environment](environment.json).
Plugin commits ea251ee,1e35b65,cf2ce6d499040f9bce0c0a0e263329da7771f1e7 and
 da1587235b84b13f2826cfd7e8b448b340322a54 implement PC03,PC01/02,PC04 and PC05.
[plugin_changes.md](plugin_changes.md) records exact approvals and validation.
The P2 repair changes tt-metal only; no additional plugin interface was added.

P2 now passes all15 checks, with unchanged four profiles, users1/32,128 output
tokens and ratio limit1.15. The formerly failing top-k40/T1/p.95/seed42 load is
78.238609 ms/token versus82.333068 ms/token plain at32users: ratio0.950269573.
All four profiles use device sampling; [P2 audit](../../readiness_vllm/p2_final_async/p2_audit.json)
records2617 device/0 host operations. The shared P2 regex misses the adapter's
operation=... mode=device marker; the byte-scoped audit supplies actual routing.
[Current P2](../../readiness_vllm/eval_probes/p2_repair/harness_requests.json)
and [raw runs](../eval_readiness/harness_requests/runs.json) replace the old4.275×
failure without a waiver or threshold change. The old evidence is preserved in
[p2_repair/prior_artifacts](p2_repair/prior_artifacts).

The fix retains SamplingGenerator→TTSampling→ttnn.sampling. Local64 candidates
per shard are compacted to global64 on device with matching IDs; an opt-in
width64 TTNN mode sorts and samples both tiles. Default32 remains unchanged.
Public temperature uploads now convert T to1/T exactly once. Evidence includes
[negative control](p2_repair/legacy_width_retry.json),
[wide distribution/trace probe](p2_repair/width64.json),
[TP4 adapter probe](p2_repair/adapter_width64.json), and
[AutoDebug/AutoFix](p2_repair/AUTOFIX.md). Strict allocation tracking includes
program-cache allocations and a failing unsafe-allocation positive control.

Final full sampling:67 passed/75 documented skips, no failures. The skips are
71 speculative cases,3 test-budget-dependent chunk cases and1 all-vocabulary
logprob case. Current mixed-shape lifecycle independently covers chunk behavior.
All12 [qualitative outputs](../../readiness_vllm/vllm_qualitative_outputs.json)
were read: coherent and on topic, no mechanical repetition, gibberish, unwanted
language drift or contamination. Six greedy texts exactly equal retained TT
controls. Stories retain a pronoun shift; verbose answers truncate at256 tokens,
including incomplete thermodynamics coverage. No complete-answer or population
quality claim is made. [Verdict](../../readiness_vllm/qualitative_verdict.json)
and [prompt format](../../readiness_vllm/qualitative_prompt_format.json) record
controls and settings; sampled requests use T.7/p.9/checkpoint k20, unseeded.

The current [lifecycle](../../readiness_vllm/p2_final_async/lifecycle_summary.json)
passes17 requests and16 exact synchronous greedy controls, with nonaligned lengths
129/130/131/1027/1539. Its [exact log interval](../../readiness_vllm/p2_final_async/lifecycle_interval.json)
contains27 chunks at starts0/128/256/512/640/768/1024/1152/1536,759 decodes,
41 page updates and two witnesses of decode while another prefill is pending.
Five prepared signatures retain IDs0–4: no workload capture/retirement,24888
stable program-cache entries,775 device/0 host forwards. There are31 consumed
new-page examples,26 page-only updates and718 unchanged-table steps. This mixed
cold-shape correctness workload has TTFT P50/P99=668.59/38214.53 ms and ITL
P50/P99=55.70/469.46 ms across17 requests; it is separate from warmed benchmarks.
[Fixed logits](../../readiness_vllm/p2_final_async/logits.json) and
[request-stable sampler ownership](../../readiness_vllm/p2_final_async/topk_route.json)
pass. The late unsupported companion now uses top-k128 because64 is supported.
Representative stale-token/current-position/inactive-row/remap/deferred-read
checks remain in [adapter_prefix_final.json](adapter_prefix_final.json).

The once-only [max-context request](../../readiness_vllm/pc05_final_async/max_context.json)
passed262111 prompt+33 output=262144 total, prompt modulo32=31. No context
reduction. Its client parsing time is not performance evidence. P3's15 checks
and P5's9 needle cases at8K/32K/128K and depths.1/.5/.9 pass in
[retained eval evidence](../../readiness_vllm/eval_probes/final_pass/status.tsv).
That directory's P2 result is historical; use the repaired P2 above. P3/P5 use
greedy. Model/cache/layout/positions/precision are unchanged, the current server
allocates the same full cache pool and all current greedy controls match; these
unaffected checks and capacity acceptance were not repeated.

Prefix caching has the documented recurrent_state exclusion;
[prefix_cache_check.json](../../readiness_vllm/prefix_cache_check.json) is an
exclusion, not a cache-hit measurement. Batched prefill stays disabled; the
serial burst is a baseline, with no off/on speedup claim. Untried policy
comparison, chunk-budget/batching/overlap tuning and capacity-client parsing
improvements remain hand-off notes in [work_log.md](work_log.md).

Reproduction (from tt-metal; use fresh output directories):

```bash
source models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/env.sh
QWEN_VLLM_HOST_COMPATIBILITY=1 TT_METAL_TRACE_ALLOC_TRACKING=1 \
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  bash models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/launch_server.sh \
  models/autoports/qwen_qwen3_6_27b/readiness_vllm/p2_final_async 32
# Attached client; exact P2 arguments are in eval_probes/p2_repair/command.json:
python models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/run_p2_acceptance.py
# After draining/stopping acceptance, separate performance server:
QWEN_VLLM_HOST_COMPATIBILITY=1 QWEN_VLLM_TELEMETRY=0 TT_METAL_TRACE_ALLOC_TRACKING=0 \
  bash models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/launch_server.sh \
  models/autoports/qwen_qwen3_6_27b/readiness_vllm/p2_perf 32
bash models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/check_server.sh \
  models/autoports/qwen_qwen3_6_27b/readiness_vllm 32 benchmark,prefill_burst full \
  models/autoports/qwen_qwen3_6_27b/readiness_vllm/p2_perf/server.log request
```

[Acceptance command](../../readiness_vllm/p2_final_async/command.json),
[phase commands](../../readiness_vllm/p2_final_async/acceptance_commands.json),
[performance command](../../readiness_vllm/p2_perf/command.json), and
[current run summary](../../readiness_vllm/vllm_run_summary.json) retain exact
configuration. The [preserved sampling/qualitative summary](../../readiness_vllm/p2_final_async/sampling_qualitative_run_summary.json)
contains merged stage metadata, including historical benchmark/prefill entries.
Use the phase commands and completed phase log for sampling/qualitative provenance,
and `p2_perf` for the current measured benchmarks.

Both servers drained to0 running/0 waiting, then their API/EngineCore processes
exited and fuser found no device holders. No reset was used. Upstream's
zero-timeout shutdown force-killed EngineCore; graceful trace teardown is not
claimed. [Acceptance cleanup](../../readiness_vllm/p2_final_async/cleanup.json)
and [performance cleanup](../../readiness_vllm/p2_perf/cleanup.json) record this.
The tracking-disabled preparation warning is checked against the corresponding
strict tracked acceptance, including program-cache allocations.

[Current acceptance summary](final_acceptance_summary.json) and
[current artifact manifest](p2_artifacts.json) name exact evidence and reuse
scopes. The PC05 manifest and round1 review are historical. Raw logs remain at
the recorded local paths; no per-artifact rehashing/compression was added.
Local commit SHAs and final review disposition are recorded in the work log.
Nothing was pushed.
