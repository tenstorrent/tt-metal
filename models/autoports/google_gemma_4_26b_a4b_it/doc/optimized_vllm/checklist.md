# Serving optimization checklist

This stage preserves selected decoder math/layout/dtypes from stages05/07/08.
No new matmul/precision/topology candidate is claimed. The serving change is
one existing output-copy operation moved into the canonical sampler trace.

| Applicable item | Evidence/status |
| --- | --- |
| Exact primary workload | before/vllm_result.json:4096/128/B1/C1; after/vllm_result.json passes |
| Secondary capacity | before/ and after/vllm_ci_serving_result.json:100/100/32,32complete |
| Selected datatype | ../datatype_sweep/selected_precision_config.json; constructor calls precision_summary; no dtype edit |
| Traced decoder, nonblocking replay | generator._replay and SamplingGenerator._execute_trace; both blocking=False |
| Canonical split greedy | _Gemma4SamplingGenerator delegates to SamplingGenerator; max_top_k32, force_argmax disabled; original sampler math unchanged |
| Persistent inputs and minimal deferred reads | adapter_changed_pages.json; generator_vllm.read_decode_output reads one device token tensor and records event |
| Changed/unchanged pages; stale tokens/positions | adapter_changed_pages.json and host_sampling_contract.log; requests_final.json: actual allocator growth and concurrency match prior and isolated controls |
| Topology audit, best correct baseline | work_log.md, AUTODEBUG.md and README.md; serving98.05% of selected full-model reference |
| Batch preserved | no batch1 specialization; batch_cache_watcher.json and final32-request burst pass |
| Nonaligned prompt/lifecycle coverage | reduced33-token probe passes; requests_final.json: full31/63/95 and33/1057/33 requests pass |
| Watcher | ACTIVE_ETH instrumentation exceeds config buffer; scoped retry passes, no asserts disabled on compute cores |
| Functional and stress/replay correctness | 84 host tests; reduced device token equivalence passes; qualitative passes; full sampling72passed,1documentedskip,zero failures |
| Runtime fallback audit | no adapter edits, no added host conversion; model host_sampling compatibility remains explicit; measured on-device mode requires all |
| Context/capacity | context_contract.json:262144, persistent tensor payload delta0, same trace region; no capability reduction |
| Device-time/roofline | intentionally unavailable: vllm_serving_profiler_disabled_to_protect_hardware |
| Final default | after/ final benchmarks; source hashes unchanged; decode effectively flat, observed TTFT lower |
| Queued async output ownership | adapter_queued_reads.json: two queued reads, stable device tensors, distinct host storage, correct old output after batch rebind |
| Cleanup | cleanup.json: no owned serving process remains |
| Independent review, commits | clean-pass in stage_review.md; local SHAs in checkpoints.json |

The inherited decoder/kernel checklist evidence is in
../optimized_multichip_decoder/README.md, ../optimized_full_model/README.md,
and ../datatype_sweep/README.md. It includes packed QKV, residual/norm layout,
SDPA/cache configs, reduced-weight/fidelity candidates, DRAM-sharded projections,
active-expert decode, persistent CCL buffers, and terminal split sampling.
Those unchanged lower-level paths are not reprofiled during this serving stage.
