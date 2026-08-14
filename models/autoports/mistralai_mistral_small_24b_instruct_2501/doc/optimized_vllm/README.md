# Optimized vLLM serving: Mistral Small 24B Instruct 2501

## Primary single-user result

**Real vLLM TT-plugin serving through `tt/generator_vllm.py`, 128-token prompt / 128-token generation / one request / concurrency 1:** TTFT P50/P99 **578.776/578.776 ms**, mean TPOT **18.925 ms**, ITL P50/P99 **17.756/18.843 ms**, aggregate output throughput **42.915 token/s**, and TPOT-derived decode **52.839 token/s/user**. The request completed 128/128 output tokens. This final-default decode rate is within **3.0%** of the comparable optimized full-model traced token-out result, 54.452 token/s/user at 128 decode steps.

The same-harness baseline was 128/128/1 at TTFT P50/P99 568.942/568.942 ms, mean TPOT 18.928 ms, ITL P50/P99 17.762/19.390 ms, aggregate output throughput 43.053 token/s, and 52.833 token/s/user. Final-default TPOT improved by 0.012% and decode rate by 0.012%; TTFT moved by +1.7%. No speculative implementation change was kept because the inherited integration was already at the optimized full-model lower bound.

## Serving contract and gates

- Model: `mistralai/Mistral-Small-24B-Instruct-2501`; TP4 `1x4` Blackhole p300c mesh; `FABRIC_1D`; `max-num-seqs=32`; block size 32; served context 32,768.
- TT config: `{"sample_on_device_mode":"all","trace_region_size":200000000,"fabric_config":"FABRIC_1D"}`. The server log reports `trace_mode=all`, `sample_on_device_mode=all`, `async_scheduling=True`, and the exact selected `bfp4_lofi_bfp8kv_bf16ccl` runtime policy.
- Decode returns device tensors from `decode_forward(..., read_from_device=False)`. The plugin calls `read_decode_output(..., async_read=True)` for the minimal sampled-token read and then `process_decode_output_host(...)`. `supports_async_decode=True` is exercised by the live async scheduler.
- Persistent replicated device tensors hold token feedback, signed cache/SDPA positions, RoPE positions, page tables, sampling parameters, KV caches, and split-sampler output. Model and Sampling1D traces replay with `ttnn.execute_trace(..., blocking=False)`. On steady async decode, stale host token/position inputs are ignored; unchanged page tables are not recopied. Reset, changed-page-table, unchanged-page-table, slot-remap, and fresh-prefill tests pass.
- Greedy serving uses the full-model split Sampling1D path (`k=1,p=0,temp=1`) with local top-k candidate tensors and `tt_out_tok` feedback. The production path has no adapter argmax, full-logits readback, generic sampling fallback, or Python token-feedback loop.
- Production sampling with host compatibility disabled passed a 192-token prompt / 128-token stochastic decode at temperature 0.7, top-p 0.9, top-k 32, followed by two eight-request mixed greedy/stochastic waves in reversed slot order. All 17 requests returned HTTP 200, health remained 200, and opt-in live routing logs record `perform_device_sampling=True` at stochastic prefill, decode, and mixed decode. The full compatibility profile separately passed **72 tests with 1 expected skip** for host-only API features; its stochastic rows are not counted as device-path proof.
- The earlier compatibility-disabled smoke began with a mixed batch containing top-k 100, explicit seeds, and penalties, all host-only for this adapter. Its saved EngineCore fatal is therefore not a supported top-k-32 sampler failure; the original overwritten server log prevents a stronger attribution. The focused supported production rerun above is the disposition evidence.
- Qualitative: six chat-template prompts, with greedy and sampled production completions, are coherent and on-topic. The sampled completions used the traced device Sampling1D route with host compatibility disabled. No repetition loop, gibberish, wrong-language drift, control-token leakage, doubled subwords, or request contamination was observed, and the automated degenerate-output check passes. A `learning,,,` completion seen only in the earlier compatibility-enabled artifact did not reproduce in the final production suite, two isolated repeats, a repeated transition sequence, or an eight-way identical-prompt batch. Matching HF/full-model controls remain under `doc/optimized_full_model/qualitative_suite/`.
- A direct logical 37-token prompt succeeds, preserving non-aligned serving. `doc/context_contract.json` remains at 32,768 with no capability reduction.

## Secondary CI serving-burst evidence

**100-token prompt / 100-token generation / 32 requests / unbounded admission (observed concurrency 32):** 32/32 completed; TTFT P50/P99 **1192.476/1193.635 ms**; mean TPOT **19.887 ms** (P99 24.216 ms); ITL P50/P99 **17.939/76.613 ms**; aggregate output throughput **1018.207 token/s**. The TPOT-derived 50.283 token/s/user is capacity/nightly-parity context, not the headline decode result.

The effective-workload baseline was TTFT P50/P99 1191.704/1192.844 ms, mean TPOT 19.618 ms, ITL P50/P99 17.935/67.958 ms, aggregate output throughput 1026.974 token/s, and 50.973 token/s/user. The baseline command explicitly set concurrency 32 while the final command admitted all 32 submitted requests without an explicit cap; both raw results report observed concurrency 32. The final run followed the full sampling stress suite; its 1.4% TPOT variance does not change the single-user verdict.

## Candidate and topology audit

| serving boundary | current operation | candidate/action | result |
|---|---|---|---|
| model decode | persistent-input model trace, nonblocking replay | retained | required full-model optimized graph; no serving-local math duplication |
| terminal sampling | vocab-sharded split Sampling1D trace with device feedback | retained | full-model evidence: 0.339 ms versus 1.261 ms force-argmax; no full-vocab gather |
| scheduler state | device token/position advance; page-table copy on change | retained | stale-input/reset/page-table tests pass |
| sampled-token read | async read of replicated mesh tensor | read only rank-0 replica | rejected: primary 52.833 to 52.800 token/s/user and CI 50.973 to 50.937 token/s/user |
| decoder/collective topology | selected 40-layer BFP4/LoFi, BFP8 KV, BF16 CCL, 11-core sharded residual | inherited and retained | serving is within 3.0% of full-model token-out; reopening device math was not indicated |

The on-device topology, dtype/fidelity, matmul geometry, fused/packed projection, persistent CCL, LM-head, and sampler checklist items are inherited from the completed optimized-full-model and datatype-sweep stages and are referenced rather than re-profiled. Live-vLLM Tracy, `tt-perf-report`, watcher, adapter profiler, device profiler, and `ReadDeviceProfiler` were intentionally not run.

## Artifacts

- Final: `final/vllm_benchmark.json`, `final/vllm_result.json`, `final/vllm_ci_serving_benchmark.json`, `final/vllm_ci_serving_result.json`, `final/sampling_tests.log`, `final/vllm_qualitative_outputs.json`, `final/vllm_qualitative_prompt_format.json`, `final/non_aligned_prompt_check.json`, and `final/degenerate_output_report.json`.
- Production device-sampling proof: `final/production_device_sampling_evidence.json` and `final/server_production_device_sampling.log`; final adapter and cleanup evidence: `final/adapter_unit_tests.log` and `final/cleanup.log`.
- Recoverable pre-initialization device transition: `final/server_production_sampling_startup_failure.log`; the bounded reset and 1x4 mesh-open control passed before the successful production run.
- Rejected rank-0-only async-read candidate: `candidates/shard0_read/`.
- Full-model comparison: `../optimized_full_model/perf_summary.json` and `../optimized_full_model/README.md`.
- Commands, recovery evidence, audit details, limitations, and commit record: `work_log.md`.
