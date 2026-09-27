# Gemma 4 optimized vLLM

Primary single-user **TTFT 2498.28 ms; decode 50.64 tokens/s/user**
(4096 input / 128 generated tokens, batch 1, concurrency 1, greedy on device).
Before: 2642.79 ms TTFT and 50.85 tokens/s/user for the identical workload and
server configuration. Decode is effectively flat; this is not a decode speedup
claim. The change removes one eager device-operation dispatch per emitted token
by including the existing public-token copy in the canonical sampler trace.

Full sampling: **72 passed, 1 documented logprob-cap skip, zero failures**.
Qualitative, primary/CI benchmarks, context/degeneracy and async contract checks
pass. Independent final review returned clean-pass; checkpoint SHAs are recorded in checkpoints.json.

| Primary workload | TTFT P50/P99 ms | TPOT mean/P99 ms | ITL P50/P99 ms | Aggregate output tokens/s | TPOT-derived tokens/s/user |
| --- | --- | --- | --- | --- | --- |
| Before:4096/128/B1/C1 | 2642.793 / 2642.793 | 19.667 / 19.667 | 19.597 / 24.302 | 24.899 | 50.847 |
| After:4096/128/B1/C1 | 2498.285 / 2498.285 | 19.747 / 19.747 | 19.598 / 27.104 | 25.567 | 50.640 |

Each primary result has one completed request; percentiles do not establish a
population distribution or statistical significance. Mean ITL 19.747424 ms agrees
with final TPOT 19.747422 ms. First-use candidate TPOT 17.7136 ms disagreed with
ITL 19.5620 ms, consistent with streaming chunk/coalescing timing; it is retained
in candidate_first_use/ and excluded from speedup claims. Warmed timing resolves
the discrepancy. TTFT was 5.47% lower in the final run, without a causal or
statistical speedup claim; no prefill math changed.

## Secondary CI capacity

| CI workload | TTFT P50/P99 ms | TPOT mean/P99 ms | ITL P50/P99 ms | Aggregate output tokens/s | TPOT-derived burst tokens/s/user |
| --- | --- | --- | --- | --- | --- |
| Before:100/100/32-request burst | 14702.597 / 14703.811 | 681.289 / 752.464 | 677.224 / 729.325 | 39.111 | 1.468 |
| After:100/100/32-request burst | 14511.987 / 14513.015 | 681.787 / 752.376 | 677.243 / 735.973 | 39.178 | 1.467 |

Both burst runs complete 32/32 requests with 3200 input and 3200 output tokens,
no client concurrency cap. Admission and prefill affect burst per-user rates;
they are not headline decode rates.

## Serving and trace contract

P300x2, mesh 1x4, TP4/DP1, max_num_seqs=32, max_model_len=262144,
block_size=32, trace_region_size=1000000000, FABRIC_1D, decode_only tracing,
async scheduling, sample_on_device_mode=all. No prefix caching or scheduler
chunked prefill. Exact commands: server_command.json, candidate_server_command.json,
baseline_command.json, final_checks_command.json. The only environment path
difference is writable log placement. OMP_NUM_THREADS=8 and the selected
head4_inner_all4_shared_down4 precision policy are unchanged.

The measured path is the real TT plugin through tt/generator_vllm.py and the
30-layer generator. The model-local SamplingGenerator subclass delegates all
sampling, seeds and penalty bookkeeping to the existing canonical implementation.
It captures the existing persistent public-token copy after sampling, eliminating
the eager post-replay slice. No new tensor storage or third replay is introduced.
Both model and sampling trace replay use blocking=False. Deferred readback copies
only the local device token tensor; host formatting follows its completion event.
No host greedy argmax/full-logits readback is used by these benchmarks. Optional
host compatibility remains enabled for shared constrained/logprob tests only.

The selected full-model autoregressive control measures 51.6473 tokens/s/user
at 4096/128/B1/C1 (../datatype_sweep/selected/performance.json). Serving reaches
98.05% of that rate. Both include token-out sampling/feedback, but use different
prompts and harness timing boundaries; this is a comparable-work reference,
not a matched speedup experiment or a teacher-forcing performance result.

## Correctness and limits

84 host sampling-contract tests pass. Reduced stale-token/position/page tests
pass with allocation tracking. Multirow cache/token equivalence passes under
TT_METAL_WATCHER=10 with ETH instrumentation disabled: full ETH instrumentation
requires 28464 bytes versus 26624 bytes available. No compute-core assertion was
disabled. Full nine-request nonaligned/concurrent/lifecycle output matches the
prior validated token streams (requests_final.json). Two queued async reads pass, with independent host storage and correct output
values/shapes after a batch rebind (adapter_queued_reads.json). Existing paged-cache ownership and context 262144 are preserved.

All six greedy shared outputs match the validated prior serving control exactly;
all six sampled outputs were read with no new serving anomaly. See
qualitative_verdict.md and qualitative_comparison.json for controls and limits.
Long outputs truncate at 256 tokens, the inherited sampler caps stochastic top-k
at 32, and the controlled malformed phrase “own-contained” remains. This is a
serving regression check, not broad model accuracy certification.

No Tracy, tt-perf-report, serving profiler, or ReadDeviceProfiler was collected.
Device latency and device roofline fields are unknown intentionally. Rejected
options and source diagnosis are in AUTODEBUG.md/AUTOFIX.md; operation audit
and the applicable optimize checklist are in work_log.md/checklist.md.
Cleanup passed with no owned serving processes remaining. Independent final
review returned clean-pass; local checkpoint SHAs are recorded in checkpoints.json.

Benchmark archives under before/, after/ and candidate_first_use/ preserve raw
JSON and normalized shared-runner output. Normalized `raw_result_file` and
command paths retain their original execution location in readiness_vllm;
the adjacent archived raw JSON is authoritative for this stage comparison.
