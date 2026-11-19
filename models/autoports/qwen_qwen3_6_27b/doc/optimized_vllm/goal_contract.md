Use $vllm-integration, $optimize, and $tt-device-usage for Qwen/Qwen3.6-27B vLLM serving in this tt-metal checkout.

Subagents are authorized when a selected skill requests or recommends them.

Start from the completed vLLM integration and selected datatype-sweep config; optimize serving in place.

Goal completion requirements:
- The measured path is real vLLM TT plugin serving through tt/generator_vllm.py.
- Plugin or vLLM changes meet the $vllm-integration plugin-change rules.
- vLLM sampling tests, qualitative outputs, primary single-user and CI serving-burst benchmarks pass.
- Warmed primary single-user TTFT, TPOT, ITL, throughput, and TPOT-derived t/s/u are measured before and after with the same run_vllm_server workload and server config (mode, max-num-seqs, max-model-len, mesh, sampling, TT config).
- CI serving-burst metrics are measured before and after as capacity evidence, never as headline t/s/u.
- Keep `max_model_len`, benchmark and eval context at doc/context_contract.json unless a hard physical limit is proven.
- Keep valid non-aligned prompt-length serving; never trade it for an aligned-only faster path.
- Repeat the mixed-shape trace reuse check from $vllm-integration after trace or scheduler changes, with supported chunked-prefill paths and mixed-traffic latency, not fixed-shape warmed benchmarks.
- Optimized vLLM decode is about as fast as optimized full-model traced decode for like work.
- Async decode: `decode_forward(..., read_from_device=False)` returns device tensors; deferred reads minimal; `supports_async_decode` is true only when vLLM exercises it. Decode reuses persistent trace inputs with ttnn.execute_trace(..., blocking=False); sync or host reads only after the async boundary, per the plugin contract.
- sample_on_device_mode=all keeps sampling on device with full-model greedy split sampling: no host argmax, full-logits readback, eager, force-argmax or slow generic sampling. If sampler work dominates decode, fix the generator LM-head/sampling contract, not the adapter.
- Token, position, RoPE, page-table, cache, and sampler trace inputs are persistent device tensors refreshed only on scheduler-state changes; stale-input tests cover changed token/position and changed/unchanged page tables.
- No Tracy, tt-perf-report, ReadDeviceProfiler or live-server/adapter profiling; use benchmark JSON, sampling, stale-input, async and no-host-fallback evidence.
- Clean runtime fallback and cleanup: no unnecessary torch/from_torch/to_torch, tilize/untilize, reshard, blocking readback, standalone-cache assumption, or leftover vLLM/EngineCore process.
- Eval readiness ($vllm-integration eval-readiness-load): sweep 3 max_num_batched_tokens values, serve the largest within the ITL bound; `run_long_context --part timing` and `run_long_answer_load` (32 users, eval profile) pass, or findings are in limitations.json.
- Batched prefill is optional: record the burst ratio; adopt only with unchanged decode and trace retention and passing batched-equals-solo evidence.
- optimized_vllm README/work_log record commands, before/after single-user and CI burst metrics, full-model comparison, eval forecast, trace/async decisions, rejected options, limitations, artifacts.
- Relevant $optimize checklist items have serving-path evidence until the $optimize Stop Rule (checklist done, last 2 kept <3%, or wall cap); untried items are hand-off notes.
- Use $tt-enable-tracing for trace capture/nonblocking replay, async readback, stale inputs and serving trace correctness.
- README.md leads with primary single-user TTFT and decode t/s/u; CI burst metrics secondary; workload shapes next to every number.
- $stage-review returns `clean-pass` (or `pass-with-carried` after 3 rounds); findings fixed/rereviewed. Local commit stage-owned touched-repo changes; never push; log SHAs.

Unmet requirements, review findings, failed gates: work. Stop only after `$autofix` fails or a plugin change needs authorization.
