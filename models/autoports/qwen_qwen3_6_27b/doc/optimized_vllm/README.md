# Qwen3.6-27B optimized vLLM — final measurements and independent acceptance

**Primary single-user TTFT: 291.277 ms; decode: 37.625576 tokens/s/user — 128 input / 128 output / 1 request, concurrency 1, greedy, all 64 layers.** These are the final post-quality benchmark results from real vLLM TT-plugin serving through `tt/generator_vllm.py`. Full sampling, qualitative smoke, synchronous/async lifecycle controls, adapter checks, long-context timing, long-answer load, runtime audit and cleanup pass. Independent [stage review](stage_review_round1.md) returned **clean-pass**, with no required work. Local checkpoint history is recorded in [commits.json](commits.json); no push is performed.

Decode remains close to the standalone generator. The retained implementation improves a measured mixed-continuation workload, while the final primary TTFT is **5.162% higher** than baseline. This report uses the reproduced final number rather than the earlier, faster C3 candidate TTFT of 277.926 ms at the same 128/128/1 workload. One measured request per primary run does not establish a latency distribution or a statistically significant improvement.

## Same-configuration primary results

All rows use 128 input / 128 output / 1 request, concurrency 1, random prompts, explicit temperature 0, ignore_eos, serial TP4, context 262144, max_num_seqs 32 and scheduler budget 512. Allocation tracking and detailed telemetry are disabled. Decode rate is 1000 / mean TPOT; output throughput includes request overhead. Request-level median and P99 coincide because each run measures one request.

| Source; workload 128/128/1 | TTFT median/P99 ms | TPOT mean/P99 ms | ITL median/P99 ms | Output tok/s | Decode t/s/u |
|---|---:|---:|---:|---:|---:|
| Before, d6780795 | 276.979/276.979 | 26.573071/26.573071 | 26.566793/27.194867 | 35.047923 | 37.632082 |
| Final C1+C2+C3, after quality checks | 291.277/291.277 | 26.577666/26.577666 | 26.570258/27.321591 | 34.905595 | 37.625576 |

The final server launched at checkpoint `8df7e416f5c8bf097ae84f1a8660cb0ef4e22951`; runtime C1/C2/C3 was frozen at `d0a39b7d`. Exact launch/configuration and phase commands are in [final source](../../readiness_vllm/optimized_final/source.json), [live process argv](../../readiness_vllm/optimized_final/server_argv.json) and [runner summary](../../readiness_vllm/optimized_final/vllm_run_summary.json) and [acceptance commands](../../readiness_vllm/optimized_final/acceptance_commands.json). Baseline and final use the same serving/workload configuration. Primary raw evidence: [before](../../readiness_vllm/optimized_baseline512/vllm_result.json), [final](../../readiness_vllm/vllm_result.json), [normalized final](../../readiness_vllm/vllm_benchmark.json). The post-quality [benchmark audit](../../readiness_vllm/optimized_final/final_benchmark_audit.json) records 282 device and 0 host forwards across primary, CI and burst measurements.

The fresh [standalone comparison](full_model_comparison.json) measures 37.702224 t/s/u, 26.523634 ms/token and 75.333871 ms TTFT at 128 input / 128 output / B1, context 262144. Serving achieves 99.797% of that decode rate, a 0.054032 ms/token difference. Standalone timing replays resident model/sampling traces with one synchronization per 127-step window and no per-token readback; vLLM streams tokens through scheduler/plugin/HTTP. These are like logical workloads with different measurement boundaries, not an identical-harness optimization comparison. The analytical primary functional floor is 9.185971 t/s/u at 128/128/B1; final serving exceeds it. See [functional_floor.json](functional_floor.json).

## Secondary capacity measurements

CI uses 100 input / 100 output / 32 requests with no explicit client concurrency cap; the server permits 32 sequences. Its TPOT includes admission and competing prefill and is not headline per-user decode speed. The separate burst uses 128 input / 32 output / 32 requests at concurrency 32. Both are greedy on the same server configuration.

| Workload and source | Completed | TTFT median/P99 ms | TPOT mean/P99 ms | ITL median/P99 ms | Output tok/s |
|---|---:|---:|---:|---:|---:|
| CI 100/100/32 before, uncapped | 32 | 3764.805/6118.651 | 100.565408/123.711529 | 56.212523/1133.620407 | 229.325791 |
| CI 100/100/32 final, uncapped | 32 | 3814.207/6197.995 | 101.056727/124.577216 | 56.225626/1151.054108 | 227.731638 |
| Burst 128/32/32 before, concurrency 32 | 32 | 3924.778/6588.572 | 167.918460/245.881359 | 56.286939/1562.256804 | 108.626447 |
| Burst 128/32/32 final, concurrency 32 | 32 | 4002.460/6727.738 | 170.045541/249.496937 | 56.226111/1595.222883 | 107.038529 |

CI TPOT-derived rates, secondary only, are 9.943777 → 9.895432 t/s/u for 100/100/32. Serial-prefill burst median/P99 TTFT ratios are 1.019793/1.021122 for 128/32/32, final divided by baseline. Batched prefill is disabled in both; these are off/off ratios, not a batched-prefill speedup or batched-equals-solo claim. Final raw [CI](../../readiness_vllm/vllm_ci_serving_result.json) and [burst](../../readiness_vllm/vllm_prefill_burst_result.json), with normalized [burst ratios](../../readiness_vllm/vllm_prefill_burst_benchmark.json), preserve exact values.

## Mixed-shape continuation tradeoffs

One paired, untracked, telemetry-enabled trial used the same 17-request workload before and after C1: prompts 129/130/131/1027/1539, 64 output tokens except a 256-token concurrent decode request, changed batch order, sampled/penalty/greedy switches and repeated prompts under load. All 16 greedy request token streams match exactly. This is separate from fixed-shape warmed benchmarks. [mixed_c1_comparison.json](mixed_c1_comparison.json) retains per-request shapes and values.

| Metric for the 17-request mixed workload | Before | C1 |
|---|---:|---:|
| TTFT median ms | 652.123 | 698.567 |
| TTFT P95 ms | 7362.055 | 2982.683 |
| TTFT P99 ms | 7362.703 | 3461.149 |
| ITL median ms | 55.669 | 55.689 |
| ITL P99 ms | 217.885 | 473.853 |
| Maximum ITL ms | 3639.968 | 1231.591 |
| Duration s | 38.794 | 35.661 |
| Completed requests/s | 0.438208 | 0.476716 |

C1 improves this trial's high TTFT, worst pause and completion rate, while median TTFT and P99 ITL worsen. One pair is not a distribution guarantee. The final telemetry-disabled [latency-only replay](../../readiness_vllm/optimized_final/mixed_latency_summary.json) completes the same 17 requests with TTFT median/P95/P99 646.196/2995.918/3467.203 ms, ITL median/P99/max 55.676/471.011/1119.700 ms and 0.477764 requests/s. Its telemetry regime differs from the paired trial; it is not another matched before/after result and deliberately does not claim trace-contract acceptance.

## Completed correctness and output checks

The full [sampling log](../../readiness_vllm/sampling_tests.log) reports **67 passed, 75 skipped**, no failures. The [sampling audit](../../readiness_vllm/optimized_final/sampling_audit.json) records 1947 device and 477 explicit host-compatibility forwards: unsupported-feature compatibility tests intentionally use the host path. These counts are not the device-only benchmark path. The 75 skips are 71 speculative-dummy-server cases, three chunk-recall cases whose explicit opt-in budget flag was not supplied by the shared sampling runner, and one unsupported full-vocabulary logprob case. The three skipped recall cases are not claimed as passed; current mixed-shape token controls, continuation probes and P5 timing provide separate supported-chunk evidence. See [sampling_summary.json](sampling_summary.json) for exact case names and reasons.

All 12 final [qualitative outputs](../../readiness_vllm/vllm_qualitative_outputs.json) were reviewed with live prompt-token validation and fresh standalone controls. The [verdict](../../readiness_vllm/qualitative_verdict.json) passes the serving qualitative smoke: all six greedy texts equal the TT controls after removing only the terminal control token, with no mechanical looping, gibberish, wrong-language drift or request contamination observed. The served artifact lacks completion token IDs, so this is exact text agreement, not a token-equality claim. Both stories have a pronoun-continuity defect; the greedy defect reproduces in fresh TT output. Longer explanations/stories remain incomplete at the 256-token cap, including incomplete thermodynamics coverage. These are disclosed observations, not an HF-reproduced waiver or population-quality claim. The [qualitative audit](../../readiness_vllm/optimized_final/qualitative_audit.json) records 1979 device and 0 host forwards. [Prompt metadata](../../readiness_vllm/qualitative_prompt_format.json) identifies chat format, thinking disabled and the actual sampled recipe.

The final all-layer [synchronous lifecycle](../../readiness_vllm/optimized_sync/lifecycle_summary.json) and [tracked async lifecycle](../../readiness_vllm/optimized_tracked/lifecycle_summary.json) pass. The latter matches 16 synchronous greedy controls, retains five prepared trace signatures with no retirements, holds 24888 program-cache entries, and exercises 27 prefill chunks, concurrent decode, 31 consumed allocator-growth cases, 26 page-only updates and 718 unchanged-table decode steps. Actual continuation offsets and batch orderings are preserved in its raw artifact. Allocation tracking includes program-cache allocations; latency is measured separately without tracking.

The final reduced layers 0/3/4 [adapter receipt](adapter_optimized_final.json), at full context allocation 262144, passes complete preparation, stale-input handling, page-only refresh, dirty full-table flush, inactive-state preservation, deferred-read sentinels, bucket transitions and retained-trace continuation. It includes the unsafe-buffer positive control. The reduced adapter and full-server lifecycle establish complementary evidence; neither is substituted for the other. Sync/tracked servers have their own cleanup receipts. Final eval-server [cleanup](../../readiness_vllm/optimized_final/cleanup.json) confirms API and EngineCore exited, no device holders, no OOV lines and no reset. Upstream force-terminated EngineCore during shutdown; graceful trace teardown is not claimed.

## Configuration and runtime contracts

Selected serving configuration: all 64 layers, P300x2/1×4 Blackhole chips 0–3, max-model-len 262144, max-num-seqs 32, page size 32, physical KV pool 8224 pages, scheduler budget 512, serial prefill, async scheduling, trace region 100000000, FABRIC_1D, trace mode all, sample_on_device_mode all and existing approved seeded-sampler policy request. The [context contract](../context_contract.json) remains unchanged. Prefix caching keeps the documented recurrent_state exclusion: GDN state cannot be reconstructed from attention blocks.

The baseline budget sweep uses 31 decoders plus 4096/32768-token arrivals. Scheduler budgets 512/2048/4096 produce worst ITL 966.6/1982.7/3394.6 ms respectively, selecting 512 as the largest within the 1000 ms bound in those runs. The final-source 512 confirmation measures worst ITL 974.3 ms, still within the bound, with 4096/32768-token prompt TTFT of 3010.8/24039.4 ms under 31 decoding users. Larger-budget rows remain the baseline-source measurements, not new final-source reruns; [budget_path_audit.json](budget_path_audit.json) verifies that the measured device branches are unchanged. Higher budgets improve long-prompt TTFT but exceed the bound. Internal model chunks remain at most 512 tokens. [baseline_summary.json](baseline_summary.json) and [final P5 timing](../eval_readiness/long_context_timing.json) preserve this provenance and select the served budget 512.

The [stage8 policy](../datatype_sweep/selected_precision_config.json) is unchanged: inner MLP BFP4/LoFi with layer 0/63 exceptions; attention/head BFP8; GDN attention HiFi2, full attention LoFi, head HiFi2; BF16 activation/residual/CCL; BFP8 KV. The final [runtime contract](../../readiness_vllm/optimized_final/runtime_contract.json), extracted by [summarize_runtime.py](summarize_runtime.py), passes exact selected-policy equality, all 64 layers, 256 observed weights and their fidelities, boundary/cache dtypes, full context, 32 slots and async/device-sampling configuration. [perf_summary.json](perf_summary.json) and [final_metrics.md](final_metrics.md) reconcile exact primary/CI/burst metrics, the analytical floor and the standalone boundary comparison; device-profiler fields are null with the prohibited-profiler reason.

The adapter uses the generator's canonical width64 split sampler with semantic greedy k1/p0/temp1. Model/sampling traces replay nonblocking over persistent token, position, RoPE, page-table and sampler inputs; token feedback and position advancement remain device-owned. The async adapter returns device tensors, queues minimal token readback and formats after completion. vLLM owns the exact attention cache and page mappings; GDN state stays model-owned. [trace_async_decisions.md](trace_async_decisions.md) documents source and ownership boundaries.

No Tracy, tt-perf-report, ReadDeviceProfiler, live-server profiling or adapter profiling is used. Device-op context comes from earlier non-serving stages; current serving claims use benchmark JSON and functional/trace/sampling evidence.

## Eval readiness and forecast

[P5 long-context timing](../eval_readiness/long_context_timing.json) passes all four checks: requests complete, no observed TPOT cliff, nonaligned 127/128-token TTFT ratio passes, and served budget equals the sweep pick. The [timing audit](../../readiness_vllm/optimized_final/timing_audit.json) records 494 device and 0 host forwards. At one user and 32 output tokens per timing request, measured rows are:

| Prompt tokens / output tokens / users | TTFT ms | Decode TPOT ms |
|---|---:|---:|
| 1024 / 32 / 1 | 541.6 | 26.69 |
| 8192 / 32 / 1 | 4160.3 | 27.13 |
| 32768 / 32 / 1 | 15812.3 | 27.65 |
| 131072 / 32 / 1 | 98123.3 | 29.63 |

The separate single-user boundary comparison measures TTFT 286.5 ms at 127 prompt tokens versus 288.2 ms at 128 prompt tokens. Context remains 262144 throughout; the timing rows do not replace the inherited maximum-context capacity evidence.

[P4 long-answer load](../eval_readiness/long_answer_load.json) passes all **11 checks** with the `lm_eval_reasoning` profile, 32 users and 6000 output tokens per user. All 192000 output tokens complete in 769.2 s, giving 249.6 aggregate tokens/s. Per-user throughput is 7.76 t/s/u in answer positions 0–1000 and 7.83 t/s/u in positions 5000–6000, above the probe's 7.37 t/s/u functional floor. The same 32-user, 256-output-token eval/plain probe measures TPOT 126.75/116.57 ms, ratio 1.087 against the 1.15 bound. The [load audit](../../readiness_vllm/optimized_final/long_answers_audit.json) records 6522 device and 0 host forwards.

P4 records no new trace captures, engine errors or preemptions; peak KV usage is 73.54%, and running/waiting requests and KV return to zero with `/health` 200. `ignore_eos=True` forces the 6000-token output window. This is load, sampling-route and resource-lifecycle evidence, **not a quality verdict or evidence that 32 useful reasoning answers were completed**.

The P4 planning forecast uses the packaged default question counts and typical answer lengths at concurrency 32, not a model-specific release plan or completed evaluations:

| Planning workload | Questions | Typical answer tokens | Forecast hours | Scope |
|---|---:|---:|---:|---|
| GPQA Diamond | 198 | 12000 | 2.63 | Extrapolates beyond the measured 6000-token window |
| MMLU Pro subset | 280 | 3000 | 0.93 | Uses measured answer-length buckets |
| IFEval | 541 | 1500 | 0.90 | Uses measured answer-length buckets |

The artifact forecasts 4.47 h total against an 18 h job cap, below its 14.4 h planning threshold. GPQA's 12000-token estimate reuses the last measured bucket beyond 6000 tokens and is optimistic if later decode slows. Replace these defaults with the actual release question counts and answer lengths before scheduling evaluations. Exact probe commands, exits and audits are in [acceptance_commands.json](../../readiness_vllm/optimized_final/acceptance_commands.json) and [final_pipeline_commands.json](final_pipeline_commands.json). The first acceptance attempt stopped at an idle assertion while the chunk probe's requests were cancelling; a bounded drain wait fixed phase handoff, and acceptance resumed at timing. The resumed acceptance, runtime audit and final shutdown all exit 0; the original failed attempt remains in the command history.

## Decisions and optimization stop

- **C1 retained:** vectorized full attention for tile-aligned physical single-row continuation, preserving original 512-token global partitions. Other valid starts retain the accepted path.
- **Rejected global 32-token partition:** changed GDN arithmetic and failed real-weight comparison; original partitions restored.
- **Rejected uniform multi-row extension:** failed the real-logit agreement gate at start 2016; prior multi-row implementation restored without weakening the bar or restricting advertised lengths.
- **C2 retained:** compute maximum page ID once per authorized refresh while validating all cache bounds.
- **C3 retained:** guard diagnostic page-diff/JSON work with telemetry; functional reload semantics and sampling-route logs remain.

[AUTODEBUG.md](AUTODEBUG.md) and [AUTOFIX.md](AUTOFIX.md) preserve failures and adapted attempts. Stop Rule 2 ended optional search after C2 and C3 primary 128/128/1 throughput changes of −0.000921% and +0.003401%, each below 3%. These are tiny measurements, not established speedups. [stop_rule.json](stop_rule.json) and [candidate_summary.json](candidate_summary.json) retain search evidence; the final headline uses the later post-quality run. [checklist.md](checklist.md) lists untried page sharing, intermediate buckets, additional prefill traces, optional batching, selected-policy geometry/readers and terminal/collective work. The stop does not waive final gates or review findings.

## Completed closure

Independent [review](stage_review_round1.md) returned clean-pass with no required work. All required checks, artifact self-checks and cleanup are complete. Evidence checkpoint `0b2d5ae5ae8c808b16af7e0753f4f0a4782d536e` is local; [commit_receipt.json](commit_receipt.json) records the reviewed runtime, gate status and repository SHAs. Nothing is pushed.

## Commands and artifact map

Run from tt-metal after sourcing this model's `doc/vllm_integration/env.sh`. Device jobs remain serialized and bounded. Exact executed commands and source receipts are authoritative; do not start a second job on an occupied mesh.

| Purpose | Command or evidence |
|---|---|
| Selected server | `QWEN_VLLM_HOST_COMPATIBILITY=1 QWEN_VLLM_TELEMETRY=0 TT_METAL_TRACE_ALLOC_TRACKING=0 bash models/autoports/qwen_qwen3_6_27b/doc/optimized_vllm/launch_server.sh <fresh-output-dir> 32 --async-scheduling request 512` |
| Attached primary/CI/burst | `bash models/autoports/qwen_qwen3_6_27b/doc/optimized_vllm/check_server.sh <output-dir> 32 benchmark,prefill_burst full <server.log> request` |
| Final phases and routing | [acceptance_commands.json](../../readiness_vllm/optimized_final/acceptance_commands.json), [final benchmark audit](../../readiness_vllm/optimized_final/final_benchmark_audit.json), [sampling audit](../../readiness_vllm/optimized_final/sampling_audit.json), [qualitative audit](../../readiness_vllm/optimized_final/qualitative_audit.json) |
| Lifecycle/control drivers | [run_lifecycle.py](run_lifecycle.py), [summarize_lifecycle.py](summarize_lifecycle.py), [final pipeline commands](final_pipeline_commands.json) |
| Serial final acceptance | [finish_pipeline.py](finish_pipeline.py) and [run_final_suite.py](run_final_suite.py); driver existence alone is not a passing result |
| CPU summaries | [summarize_runtime.py](summarize_runtime.py), [summarize_final.py](summarize_final.py) |
| Search and runtime contract | [candidate_summary.json](candidate_summary.json), [stop_rule.json](stop_rule.json), [checklist.md](checklist.md), [topology audit](topology_audit.md), [trace/async decisions](trace_async_decisions.md) |

C1/C2/C3 introduce no serving-plugin or vLLM interface changes. Existing stage9 PC01–PC05 approvals in [plugin_changes.md](../vllm_integration/plugin_changes.md) remain authoritative. Repository state and local checkpoint SHAs are recorded in [stack_state.json](stack_state.json) and [commits.json](commits.json). Independent review returned clean-pass.

[artifact_manifest.json](artifact_manifest.json) distinguishes committed evidence from local raw transcripts. Original server logs remain byte-identical for phase-offset audits; oversized server logs also have committed `.log.gz` copies under the model-bringup server-log exception. Smaller raw server logs stay in the persistent workspace beside committed lifecycle, runtime and routing receipts. Routine hook/driver transcripts and the large AutoDebug transcript remain local; their compact reports and exact command receipts are committed.
