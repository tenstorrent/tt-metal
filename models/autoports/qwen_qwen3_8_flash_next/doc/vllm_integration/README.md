# Qwen3.8-Flash-Next vLLM integration

**Primary single-user: TTFT6.473s; decode9.676tokens/s/user** — 128 input / 128 output / 1 request, concurrency1, greedy, resident capacity2.
**Full sampling passes:67 passed,75 skipped; run_vllm_server exit0.** Matched qualitative acceptance passes12/12. P2/P3 critical checks, all9 P5 cases, and full262144-token context acceptance pass. Aggregate gate exit1 is advisory only; independent review returned **clean-pass** with no required work. Nothing has been pushed.

| Serving workload | TTFT P50 / P99 (ms) | TPOT mean / P50 / P99 (ms) | ITL P50 / P99 (ms) | Output throughput (tokens/s) | TPOT-derived tokens/s/user |
| --- | --- | --- | --- | --- | --- |
| Primary:128in/128out/1request, concurrency1 | 6472.729 / 6472.729 | 103.351 / 103.351 / 103.351 | 103.474 / 104.377 | 6.531068 | 9.675793 |
| Secondary CI:100in/100out/32requests, uncapped client concurrency | 214314.488 / 420849.723 | 201.663 / 201.660 / 202.124 | 201.813 / 203.119 | 7.259288 | 4.958759 |
| Serial prefill baseline:128in/32out/32requests, concurrency32 | 125835.494 / 242367.825 | 201.555 / 201.523 / 202.155 | 201.583 / 202.851 | 4.118953 | 4.961431 |

All profiles are greedy, ignore EOS, use selected precision, C262144/B2/page32/chunk256, async scheduling and tracking0. Completed requests:1/1,32/32,32/32; no missing output tokens. The primary has one request, so its TTFT and TPOT percentiles coincide. No benchmark generation warmups were requested; generator preparation precedes requests. The CI and prefill burst values include admission waiting at resident capacity2 and are secondary serving measurements, not headline decode speed. Decode tokens/s/user is `1000 / mean_tpot_ms`.

Raw results: [primary](../../readiness_vllm/vllm_result.json), [CI burst](../../readiness_vllm/vllm_ci_serving_result.json), [prefill burst](../../readiness_vllm/vllm_prefill_burst_result.json). Normalized results have matching `*_benchmark.json` names (primary `vllm_benchmark.json`). Exact commands and workload settings are embedded in each normalized result; [benchmark runner log](logs/benchmark_runner.log) records successful completion. [Performance process provenance](../../readiness_vllm/performance_server_provenance.json) records the actual launch command and environment. Batched prefill remains disabled, so the prefill burst is the serial baseline and has no invented enabled-arm ratio.

## Shared serving path

The [adapter](../../tt/generator_vllm.py) delegates allocation, prefill, decode,
canonical split sampling, async reads and cleanup to the full-model generator.
vLLM owns attention KV allocation and passes that cache through the low-level
API. Constant-size recurrent state remains model-owned. `supports_async_decode`
is true, decode trace is enabled, and the server uses `--async-scheduling`.
The device path reuses canonical token-out feedback, nonblocking trace replay,
and persistent device token/position state. It has no host argmax, full-logits
readback or Python token-feedback loop. Explicit optional host compatibility is
used for documented unsupported sampling features and diagnostic logprobs;
performance uses greedy on-device sampling with `sample_on_device_mode=all`.

Serving loads selected `combined_shared4_mixed`: expert BFP4/LoFi; interior
shared weights BFP4/LoFi with layers0/47 BFP8/HiFi4; QSA/head BFP8/HiFi2;
GR/GDN/ZAB BFP8/LoFi; activations, residuals, CCL and KV cache BF16.
The runtime [construction record](../../readiness_vllm/acceptance_tracking_v2/resolved_adapter.json)
verifies all48 layers, precision and registration. The
[serving contract](../full_model/serving_contract.json) records supported decode
buckets and capability boundaries. A model-private config normalization maps
vLLM's equivalent attention labels; [AutoFix](AUTOFIX.md) records validation.

Stack: standalone [tenstorrent/vllm-tt-plugin base042a948](https://github.com/tenstorrent/vllm-tt-plugin/tree/042a94863ad27d88207bcf71abdf67122c6f4c27)
and [upstream vLLM v0.31.0](https://github.com/vllm-project/vllm/tree/v0.31.0),
as pinned by the plugin's `docs/install-vllm-tt.sh`. Runtime plugin commit
`edb95ca695d719a5340d919ae3d6fc2f6dd219a8` implements the owner-approved
`supports_prefill_sample_mask`, `prefill_sample_mask`, and all-intermediate
no-output semantics. Test-only follow-ups end at
`629f07cb385494b14d0ee3d138de039d71d09546`. Upstream vLLM is unchanged.
[Plugin-change justification and validation](plugin_changes.md),
[stack provenance](serving_stack.json), and [local commits](commits.json) give
exact sources and commands. Host validation passed2343/16skip.
Registration maps `TTQwen4ExpForConditionalGeneration` to
`models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm:QwenFlashNextForCausalLM`.

The exact [server command](run_server.sh) uses P300x2, mesh1x4, TP4/EP4,
C262144/B2/page32, token budget256, FABRIC_1D_RING, trace_mode all,
trace_region_size0, qwen3 reasoning and qwen3_xml tool parsing. Acceptance
uses `TT_METAL_TRACE_ALLOC_TRACKING=1` and skip-program-cache0; final performance
used tracking0. No live-serving profiler is used. Environment and attached
probe commands are in [acceptance commands](acceptance_commands.md).

## Context, correctness and traces

Supported/served context remains **262144**, matching
[context_contract.json](../context_contract.json). [Maximum-context acceptance](../../readiness_vllm/max_context_acceptance.json)
passed once:262136 nonaligned input plus8 output tokens, exact total262144,
10315.503seconds. The repeated-token capacity stress makes no quality claim. [Full48 capacity evidence](capacity_full_v1.json) records8195
scheduler-owned blocks,32,662,724,096 allocated bytes/rank and898,926,080 bytes
of observed runtime growth, with a2GiB reserve. B32 non-KV resident state alone
exceeds physical34,178,731,008 bytes/rank. B2 is the largest tested serving
configuration, not a claimed largest physically feasible batch.

[Latest full48 mixed-shape run](../../readiness_vllm/acceptance_tracking_v2/mixed.json)
passes ten requests:129/130/131/257/513/515 input tokens,96 output tokens,
concurrency2. Actual chunk starts0/256/512 overlap active decode. Its202
execution signatures each capture once, with zero retirements, unsafe
allocation findings, intermediate outputs and OOV device tokens. The artifact
records TTFT percentiles, stream gaps and request throughput. Its optional
non-overlapped control is null; the generic correctness_scope text mentioning
that control must not be read as a passed comparison. Separate controls below
supply matched-history evidence. Initial full48 mixed evidence had199
signatures; this differs from the latest server's broader prior workloads.

[Direct adapter transitions](adapter_async_transitions_v2.json) establish stale
input handling, page growth, live remapping, retirement/reuse and two queued
reads with exact token/state/RNG agreement. The post-repair
[adapter check](adapter_async_table_snapshot_v1.json) also passes. Recorded real
plugin scheduler histories in both arrival orders compare immediate and queued
execution exactly: [forward](../../readiness_vllm/recorded_history_v1/recorded_queue_v2.json)
200 history/307 output checks,103 commits; [reverse](../../readiness_vllm/recorded_history_reverse_v1/recorded_queue.json)
202 history/310 output checks,104 commits. Both observe91 queued overlaps and
zero unsafe overlaps, versus zero immediate overlaps. These reduced controls
are correctness diagnostics, not performance. Full48
[concurrent repeats](../../readiness_vllm/acceptance_tracking/history.json) pass
six exact96-token responses under row reversal and automatic page growth.

Earlier different-history sync/async comparisons failed6/8 greedy outputs.
They remain failed evidence. [Diagnosis](AUTODEBUG_async.md) and
[history manifest](masked_mixed_history_manifest_v1.json) establish differing
prior widths/slots, without proving the numerical cause. Matched-history checks
above test async correctness without that confound; no checkpoint-quality
explanation or retrospective pass is claimed.

The exact-source [standalone full48 baseline](../../readiness_vllm/logit_baseline_table_snapshot_v2.json)
passes full-logit repeat/cross-position/solo-batch checks and canonical masked
token/RNG/no-intermediate-output checks. The [served comparison](../../readiness_vllm/acceptance_tracking_v2/logit_determinism.json)
passes repeat/reversed top20 selected logprobs at both physical B2 positions,
matching the baseline within1e-5. This host-logprob diagnostic does not claim
full-vocabulary served readback or performance.

## Sampling, quality and eval readiness

[Final full sampling](../../readiness_vllm/sampling_full_acceptance.json):67pass,
75skip,1109.31seconds, runnerexit0. The shared suite and assertions are unchanged
apart from documented general test workload repairs. Its existing optional
`--tt-reasoning-token-budget=1528` adds room to reasoning chat tests while keeping
prompts, seeds and served thinking defaults. Exact log:
`readiness_vllm/sampling_tests.full_budget1528_v1.log`; runner command/output:
`logs/sampling_full_budget1528_v1.log`; complete observed chat responses:
`readiness_vllm/sampling_chat_full1528_v1/`. Initial7failed/60pass/75skip is
preserved. The configured logprob ceiling20 causes the existing all-vocabulary
probe to skip; it is not full-vocabulary evidence.

The [native allowance control](hf_reasoning_budget_outcome.md) remains **8/9,
failed**: one sample stopped before the reasoning-end marker. Eight completed
paths motivate the conditional allowance; they do not establish a universal
budget. Native constrained outputs also include language/topic drift. The
[independent diagnosis](AUTODEBUG_reasoning_budget.md) found no concrete source
defect; terminal logits/RNG were not measured. All eleven served focused chat
responses stopped normally and satisfied original assertions; seed0 required112
tokens, exceeding the original100. Full-profile responses were also read.
Constrained greetings include altered spellings and a one-character answer;
this is not the accepted qualitative suite. See [AutoFix record](AUTOFIX_reasoning_budget.md).

[Matched qualitative acceptance](../../readiness_vllm/quality_served_v1/qualitative_review.json)
passes12/12: six shared prompts, greedy and sampled, seed42, thinking off,
max256. Every served text equals its full-model TT control; HF/server/actual
input token IDs match all six rendered prompts. Root read HF, TT and served
outputs for coherence, topic, repetition, gibberish, language drift and request
contamination. Outputs are coherent/on-topic without observed contamination or
mechanical degeneration. French is explicitly requested. Long answers truncate
at256 and sampled haiku meter is imperfect. Served defaults remain unchanged.

P2 passes all critical checks after the decode-only unchanged-table validation
repair. At B2,128output tokens, reasoning-host TPOT219.0ms versus greedy198.9ms
is1.101, below1.15. B1,128output host122.1ms versus103.5ms is1.179 advisory.
[Exact P2 evidence](P2_table_snapshot_v2.json) records all accepted request
profiles and device routes. P3 passes critical chat/tool/thinking/history/stream
checks; `render_c07` is advisory because `/tokenize` treats deprecated
reasoning_content differently while actual chat retains the history.
[P3 audit](P3_render_audit.md) limits that conclusion precisely. [P5 accuracy](../eval_readiness/long_context_accuracy.json) passes all9 cases at8K/32K/128K, depths.1/.5/.9, with complete raw requests and responses under
`readiness_vllm/P5_accuracy_raw/`. Every response was read and exactly recalls its requested passphrase; all stop normally. Actual prompt lengths range8166–130899 tokens; thinking off, temperature0, max512 output. The unchanged shared scorer exited0.

Prefix caching is excluded by documented `recurrent_state`; the
[prefix artifact](../../readiness_vllm/prefix_cache_check.json) records exclusion,
not a successful cache-hit test. Batched prefill is disabled; its burst reports
the serial baseline only.

## Performance scope and closure

Full-model teacher forcing9.776829665936t/s/u (AIME,100output) and resident
B1 generator9.801183224430t/s/u (128input/128output, TTFT4711.342390ms) are
lower-bound context only, not serving results. Primary serving decode9.675793tokens/s/user (128/128/1) is within1.3% of the comparable resident generator bound. Teacher-forcing workload differs and is not a serving performance verdict. The measured path has canonical device feedback, nonblocking traces, deferred token readback and scheduler-triggered metadata refresh; no identified avoidable decode overhead remains. P2 removed avoidable host-table
normalization without changing canonical device sampling. Untried host-DMA,
position-upload and optional batched-prefill ideas are carried in the work log.

[Final cleanup](../../readiness_vllm/cleanup.json) records graceful shutdown, zero device holders and no remaining owned API/EngineCore/runner processes. No reset was needed. The [performance route audit](../../readiness_vllm/performance_sampling_route_audit.json) records2207 device decode submissions,35 device prefill submissions, zero host-sampled steps and zero OOV tokens. Of decode submissions,2123 needed neither input nor page-table reload;51 refreshed only page tables and33 reloaded inputs at scheduler transitions. Tracking-only diagnostic fields are absent. Every one of198 prepared/captured execution signatures was captured once; shutdown releases are recorded separately.
Historical reduced-server summaries are preserved byte-identically under
`doc/vllm_development/`; its migration manifest distinguishes diagnostic contexts
from current acceptance. The context checker passes without reducing capability.

[Work log](work_log.md) records commands, failures, owner decisions and SHAs.
The aggregate [stage check](logs/stage9_final_check.log) completed with exit1: no critical failures, only historical reduced-context log mentions, optional batched-prefill comparison, P2 B1 host-route ratio and P3 tokenize-only history advisories. Independent [stage review](stage_review_round1.md) returned **clean-pass** in round1, with no required work. [Commit records](commits.json) identify local checkpoints. No push.
