# Qwen3.6-35B-A3B vLLM Integration

Status: `Qwen/Qwen3.6-35B-A3B` serves through the shared TT vLLM path on P300C 2x2 with vLLM-owned attention KV cache, on-device traced sampling, and `max_model_len=262144`.

Primary single-user result, workload `128 input / 128 output / 1 request / max concurrency 1 / temperature 0.0 / max_num_seqs 32`: TTFT P50/P99 `7517.3 ms / 7517.3 ms`; TPOT mean/P99 `945.3 ms / 945.3 ms`; ITL P50/P99 `919.2 ms / 1232.0 ms`; output throughput `1.0034 tok/s`; TPOT-derived decode `1.0579 t/s/u`. Raw artifact: `readiness_vllm/vllm_result.json`; normalized artifact: `readiness_vllm/vllm_benchmark.json`.

Secondary CI serving-burst result, workload `100 input / 100 output / 32 requests / unbounded concurrency / temperature 0.0 / max_num_seqs 32`: TTFT P50/P99 `156247.2 ms / 156248.2 ms`; TPOT mean/P99 `1003.5 ms / 2002.9 ms`; ITL P50/P99 `919.2 ms / 3851.7 ms`; output throughput `12.7538 tok/s`; TPOT-derived decode `0.9965 t/s/u`. Raw artifact: `readiness_vllm/vllm_ci_serving_result.json`; normalized artifact: `readiness_vllm/vllm_ci_serving_benchmark.json`. This profile is CI parity and burst-serving capacity evidence, not the headline decode t/s/u.

## Serving Path

- Adapter: `models/autoports/qwen_qwen3_6_35b_a3b/tt/generator_vllm.py::Qwen3_5MoeForConditionalGeneration`.
- Plugin registration: `/localdev/vkovacevic/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models()` registers `TTQwen3_5MoeForConditionalGeneration` and overrides upstream `Qwen3_5MoeForConditionalGeneration` to the TT adapter.
- Cache ownership: vLLM allocates and owns the attention KV cache. The adapter accepts `kv_cache` and page tables from vLLM, normalizes vLLM one-based physical block IDs to TT zero-based cache IDs, and stores only model-owned linear-attention state with the cache object.
- Sampling path: `sample_on_device_mode=all`; prefill and decode reuse the full-model generator's split token-output sampler. The measured adapter has no separate sampling path, host argmax path, full-logits readback path, generic top-k greedy fallback, or Python readback/writeback token-feedback loop.
- Capability flags: `supports_async_decode=True`, `supports_async_decode_overlap=False`, `supports_sample_on_device=True`, `supports_prefix_caching=False`.
- Decode trace: enabled in the adapter. Trace capture now resets token/current-position tensors and executes the captured trace before returning the first traced step, so the first replayed token uses the current request rather than capture-time dummy inputs.

Final readiness command:

```bash
env TT_METAL_HOME=/localdev/vkovacevic/tt-metal TT_METAL_WATCHER_DISABLE_ETH=1 TT_READINESS_TRACE_REGION_SIZE=384000000 \
PYTHONPATH=/localdev/vkovacevic/tt-metal:/localdev/vkovacevic/vllm/plugins/vllm-tt-plugin/src:/localdev/vkovacevic/vllm:${PYTHONPATH:-} \
LD_LIBRARY_PATH=/localdev/vkovacevic/tt-metal/build_Release/lib:${LD_LIBRARY_PATH:-} \
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve,sampling,benchmark \
  --model-dir models/autoports/qwen_qwen3_6_35b_a3b \
  --hf-model Qwen/Qwen3.6-35B-A3B \
  --mesh-device P300C \
  --max-num-seqs 32 \
  --max-model-len 262144 \
  --block-size 32 \
  --port 8011 \
  --server-timeout 2400 \
  --tt-config '{"trace_region_size":384000000,"fabric_config":"FABRIC_1D_RING"}' \
  --additional-server-args=--async-scheduling
```

Runner output: `readiness_vllm/run_vllm_server_final_sampling_benchmark_trace_replay_patch.log`. Server log: `readiness_vllm/server.log`. The merged TT config was `sample_on_device_mode=all`, `trace_region_size=384000000`, and `fabric_config=FABRIC_1D_RING`. The plugin disabled scheduler overlap because this model reports `supports_async_decode_overlap=False`; model async decode support remains enabled.

## Precision And Context

Serving uses `doc/datatype_sweep/selected_precision_config.json` `baseline_default` through the full-model generator load path:

- weights: embedding, norms, and router BF16; attention, linear-attention, shared-MoE, and LM-head BF8; routed-MoE BF8 on linear-attention layers and BF4 on full-attention layers;
- activations, residual, CCL, KV cache, linear state, and logits BF16; token sampling output uint32;
- compute fidelities: TTNN defaults; layer exceptions: none.

Served `max_model_len=262144`, matching `doc/context_contract.json` `supported_context=262144`. No serving context reduction was applied. Non-aligned prompt evidence: `readiness_vllm/non_aligned_prompt_check_trace_replay_patch.json` records a direct chat request with server-side prompt length `26`, non-divisible by `16, 32, 64, 128, 1072, 2048`, and HTTP `200`.

## Correctness Evidence

- Adapter and sampling-contract tests: `readiness_vllm/adapter_tests.log`, plus latest local rerun `5 passed`.
- Full TT plugin sampling suite against the final non-debug server: `readiness_vllm/sampling_tests.log`, `72 passed, 1 skipped, 2 warnings in 3579.31s`.
- Targeted request-isolation plus seeding reproducer after the lane greedy fix: `readiness_vllm/targeted_request_isolation_then_seeding_after_lane_greedy_patch.log`, `2 passed`.
- No-thinking qualitative degeneracy check: `readiness_vllm/check_vllm_no_think_degenerate_output_trace_replay_patch.log`, `No degenerate output detected`.
- Stale-token/current-position/page-table coverage is in `models/autoports/qwen_qwen3_6_35b_a3b/tests/test_generator_vllm.py`: trace input reset before replay, capture-step replay before return, vLLM page-table normalization, page-table slot scatter, async capability, context contract, and no adapter logits sampling path.

Any host-side sampling compatibility used by shared tests is explicit and optional in the plugin. It does not replace the `sample_on_device_mode=all` performance path.

## Qualitative Verdict

Accepted qualitative artifact: `readiness_vllm/vllm_chat_no_think_qualitative_outputs.json`; runner log: `readiness_vllm/run_no_think_chat_qualitative_trace_replay_patch.log`; prompt/control verdict: `readiness_vllm/qualitative_no_think_control_verdict.json`.

Judgment: no-thinking chat requests with `chat_template_kwargs={"enable_thinking": false}` are coherent and topical. The story prompt now begins `a young shepherd discovered a hidden door...`, the explanation and thermodynamics prompts stay on topic, the French translation stays in French, and the code prompt emits a plausible Fibonacci function. No repetition loop, gibberish, wrong-language drift, or request contamination was observed in the accepted no-thinking outputs. Some 64-token-capped responses truncate naturally.

Raw completion and default chat artifacts remain as cautionary evidence: `readiness_vllm/vllm_qualitative_outputs.json`, `readiness_vllm/vllm_chat_qualitative_outputs.json`, and `readiness_vllm/qualitative_verdict.json` show Qwen thinking scaffolding or request-analysis contamination when the request does not disable thinking. Those modes are not the accepted clean qualitative mode.

## Performance Notes

Full-model teacher-forcing lower bound from the selected datatype sweep is TTFT `9079.1 ms` and decode `16.3785 t/s/u` in `doc/datatype_sweep/selected_precision_config.json`. The primary vLLM serving result is TTFT `7517.3 ms` and TPOT-derived decode `1.0579 t/s/u` for the `128/128/1` workload on a server configured for `max_num_seqs=32`.

The measured serving path has removed the known avoidable adapter fallbacks: no adapter host argmax, no full-logits readback, no Python token readback/writeback feedback loop, no generic top-k greedy fallback in the performance path, no stale trace input return on first capture, no one-based vLLM page-table mismatch, and no per-token page-table copy when unchanged. A temporary async-overlap experiment (`readiness_vllm/vllm_result_async_overlap.json`, `readiness_vllm/vllm_benchmark_async_overlap.log`) improved one TTFT run but did not materially improve TPOT, so overlap remains disabled until fully proven. The remaining decode gap versus full-model teacher-forcing is a serving-path limitation; the CI burst TPOT is recorded only as secondary capacity evidence.

## Cleanup

Runtime cleanup was audited after the final serve and benchmarks:

- `readiness_vllm/process_cleanup_check.log`: `NO_MATCH` for `run_vllm_server`, `vllm.entrypoints.openai.api_server`, and `VLLM::EngineCore`.
- `readiness_vllm/tt_smi_list_after_cleanup.log`: all four P300C chips were visible and resettable.
- `readiness_vllm/mesh_smoke_after_cleanup.log`: after a board reset recorded in `readiness_vllm/tt_smi_reset_after_cleanup_smoke_timeout.log`, 2x2 mesh open/synchronize/close completed with `MESH_SMOKE_OK`.

The server and benchmark logs print nanobind leak diagnostics during Python shutdown, but the process cleanup, `tt-smi`, and final mesh smoke show no leftover serving process holding devices.
