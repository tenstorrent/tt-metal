# Gemma 4 12B vLLM Integration

Batch-1 serving summary: full vLLM sampling passed, qualitative output is not production-quality, TTFT P50/P99 is 40003.7 ms / 40203.0 ms, ITL P50/P99 is 44.0 ms / 88.2 ms, aggregate decode throughput is 21.3 tok/s, and mean per-user decode throughput is 21.8 t/s/u.

Benchmark workload: `prompt_len=128`, `output_len=128`, `num_requests=32`, `concurrency=8`, `max_num_seqs=1`, `max_model_len=4096`, T3K, `sample_on_device_mode=all`, `trace_region_size=100000000`, `fabric_config=FABRIC_1D_RING`.

## Status

- Adapter: `models/autoports/google/gemma-4-12B/tt/generator_vllm.py`.
- Registration: `/localdev/moconnor/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py` registers `Gemma4UnifiedForCausalLM`, `Gemma4UnifiedForConditionalGeneration`, `TTGemma4UnifiedForCausalLM`, and `TTGemma4UnifiedForConditionalGeneration`.
- Starting baseline: `../optimized_full_model/README.md` reports the completed optimized full model on T3K at TTFT 121.93 ms for the 149-token AIME24 prompt and traced on-device decode replay at 23.08 tokens/s/user.
- Sampling: full TT plugin profile passed: `71 passed, 1 skipped`.
- Serving artifacts: `models/autoports/google/gemma-4-12B/readiness_vllm/`.
- Qualitative verdict: outputs are often on-topic only as fragments, with repetition loops, prompt contamination, blank sampled completions on two prompts, code/request drift, and no reliable French translation. This validates the shared vLLM path, but not model quality.

## Serving Commands

Server launch used for the final full run:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir models/autoports/google/gemma-4-12B \
  --hf-model google/gemma-4-12B \
  --mesh-device T3K \
  --max-num-seqs 1 \
  --max-model-len 4096 \
  --server-timeout 1800 \
  --tt-config '{"trace_region_size": 100000000, "fabric_config": "FABRIC_1D_RING"}' \
  --additional-server-args '--chat-template models/autoports/google/gemma-4-12B/doc/vllm_integration/simple_chat_template.jinja --generation-config vllm'
```

Final full checks against that server:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages sampling,qualitative,benchmark \
  --server-url http://localhost:8000 \
  --model-dir models/autoports/google/gemma-4-12B \
  --hf-model google/gemma-4-12B \
  --max-num-seqs 1 \
  --max-model-len 4096 \
  --sampling-profile full \
  --tt-config '{"trace_region_size": 100000000, "fabric_config": "FABRIC_1D_RING"}'
```

The final check command exited 0. The held server was terminated after the run; no `vllm`, `api_server`, or `EngineCore` process remained except the unrelated repo multigoal launcher whose command line contains `vllm`.

## KV Cache Ownership

vLLM owns the serving KV cache. The adapter builds the generator with `allocate_standalone_cache=False`, allocates per-layer cache tensors from vLLM KV specs, and passes the exact `kv_cache`, `page_table`, and hybrid per-layer page tables through to the generator/model prefill and decode paths.

The standalone generator path is still separate. In vLLM mode, hidden fallback to generator-owned cache is rejected.

## Evidence

- `readiness_vllm/sampling_tests.log`: full sampling pass, `71 passed, 1 skipped`.
- `readiness_vllm/vllm_qualitative_outputs.json`: six greedy/sampled prompt outputs; judged poor quality with repetition and request contamination.
- `readiness_vllm/vllm_benchmark.json`: benchmark metrics above.
- `readiness_vllm/server.log`: server startup, warmup, request traffic, and benchmark traffic.
- `AUTODEBUG.md`: AutoFix diagnosis for stale traced decode behavior during bringup.

## Limitations

- `max_num_seqs=1` only; the adapter rejects larger vLLM batch sizes today.
- Prefix caching is disabled.
- Async decode is not advertised.
- `test_chat_logprobs_all_vocab` is skipped by the plugin suite.
- Text quality remains poor and should be treated as an optimized full-model/model-quality limitation, not as proof of a production-ready model.
