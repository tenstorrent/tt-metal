# Qwen3-4B vLLM Integration

## Headline

Primary single-user vLLM TT serving, final production path:

| Workload | TTFT P50/P99 | TPOT mean/P99 | ITL P50/P99 | Throughput | Decode t/s/u |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 prompt tokens, 128 output tokens, 1 request, concurrency 1, temperature 0.0, ignore EOS | 225.6 / 225.6 ms | 13.3 / 13.3 ms | 10.3 / 10.7 ms | 67.0 output tok/s | 75.4 |

CI serving-burst evidence, secondary only:

| Workload | TTFT P50/P99 | TPOT mean/P99 | ITL P50/P99 | Throughput | TPOT-derived t/s/u |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 prompt tokens, 100 output tokens, 32 requests, vLLM burst admission, temperature 0.0, ignore EOS | 1893.0 / 2869.6 ms | 33.2 / 48.4 ms | 17.7 / 295.8 ms | 586.3 output tok/s | 30.2 |

The CI burst profile is not the headline decode rate because burst admission and chunked prefill can affect TPOT. The headline decode value is the primary single-user `1000 / mean_tpot_ms`.

## Serving Status

- Model: `Qwen/Qwen3-4B`
- vLLM adapter: `models/autoports/qwen_qwen3_4b/tt/generator_vllm.py`
- vLLM plugin registration: `/home/ubuntu/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`, `TT_QWEN3_TEXT_VER=autoport_qwen3_4b`
- Mesh: `P150x4`, tensor parallel degree 4
- Max model length: `40960`, matching `doc/context_contract.json` `current_supported_context`
- Max num seqs: `32`
- KV cache block size: `32`
- Sampling profile: `full`
- TT config: `{"sample_on_device_mode":"all","trace_region_size":134217728,"fabric_config":"FABRIC_1D_RING"}`
- Selected precision: `baseline_bfp4_lofi_bf16kv_bf16ccl`
- Precision policy: BFP4 LoFi attention/MLP/LM-head weights; BF16 embeddings/norms, activations, residual, CCL, logits, and KV cache; no layer exceptions
- Async decode: `supports_async_decode=True`, async scheduling enabled in the benchmark server
- Decode trace: enabled through the adapter's traced token-out path

The adapter binds vLLM-owned KV cache allocation and passes that cache through to the generator. It does not allocate a hidden standalone serving cache. The performance path delegates to the full-model generator's traced token-out methods and canonical split-sampling token-out path.

## Commands

Final shared serving, sampling, qualitative, and benchmark run:

```bash
TT_QWEN3_TEXT_VER=autoport_qwen3_4b \
VLLM_PLUGINS=tt,tt_model_registry \
QWEN3_4B_AUTOPORT_DIR=/home/ubuntu/tt-metal/models/autoports/qwen_qwen3_4b \
PYTHONPATH=/home/ubuntu/vllm:$PYTHONPATH \
python -m models.common.readiness_check.run_vllm_server \
  --stages serve,sampling,qualitative,benchmark \
  --model-dir models/autoports/qwen_qwen3_4b \
  --hf-model Qwen/Qwen3-4B \
  --mesh-device P150x4 \
  --max-num-seqs 32 \
  --max-model-len 40960 \
  --block-size 32 \
  --sampling-profile full \
  --server-timeout 1800 \
  --tt-config '{"trace_region_size": 134217728, "fabric_config": "FABRIC_1D_RING"}'
```

Non-aligned prompt-length check against the same shared vLLM path:

```bash
python_env/bin/vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-4B \
  --base-url http://localhost:8000 \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 37 \
  --random-output-len 8 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result \
  --result-filename models/autoports/qwen_qwen3_4b/readiness_vllm/non_aligned_prompt_len37_result.json \
  --temperature 0.0
```

Adapter contract and stale-token/current-position/page-table checks:

```bash
pytest -q models/autoports/qwen_qwen3_4b/tests/test_full_model_contract.py --tb=short
```

## Artifacts

- `readiness_vllm/sampling_tests.log`: full shared TT sampling suite, `72 passed`, `1 skipped`
- `readiness_vllm/vllm_qualitative_outputs.json`: six qualitative prompts, greedy and sampled completions
- `readiness_vllm/vllm_result.json`: raw primary single-user benchmark
- `readiness_vllm/vllm_benchmark.json`: normalized primary benchmark summary
- `readiness_vllm/vllm_ci_serving_result.json`: raw CI serving-burst benchmark
- `readiness_vllm/vllm_ci_serving_benchmark.json`: normalized CI serving-burst summary
- `readiness_vllm/non_aligned_prompt_len37_result.json`: prompt length 37, output length 8, completed `1/1`
- `readiness_vllm/non_aligned_prompt_len37.log`: non-aligned request log
- `readiness_vllm/adapter_runtime_tests.log`: adapter stale-token/current-position/page-table runtime contract log
- `readiness_vllm/trace_warning_audit.log`: active-trace warning classification for the final non-aligned rerun
- `readiness_vllm/server.log`: vLLM server log from the final runs
- `generated/test_reports/most_recent_tests.xml`: latest adapter contract test report

## Qualitative Verdict

I read all six prompt outputs in `vllm_qualitative_outputs.json`. Greedy and sampled completions are coherent and on topic for haiku, supervised versus unsupervised learning, story continuation, thermodynamics, French translation, and Python Fibonacci code. There is no material repetition, gibberish, wrong-language drift, or request contamination in the final outputs. The degenerate-output checker also reports no findings for the final vLLM qualitative artifact.

## Comparison

Optimized full-model teacher-forcing is only a lower bound, not the serving metric. `doc/optimized_full_model/perf_summary.json` reports `96.9 t/s/u` steady token-out and `63.95 t/s/u` traced teacher-forcing decode. The vLLM single-user serving path reports `75.4 t/s/u` from mean TPOT with TTFT and API serving overhead measured separately. The measured path uses async decode and traced on-device sampling; the TP4 pair-reducer bug that caused avoidable vLLM-specific decode overhead and bad greedy output was fixed before this benchmark.

## Trace Lifecycle

The final non-aligned prompt rerun still emits TT-Metal's active-trace allocation warning once after the last decode signpost. `readiness_vllm/trace_warning_audit.log` classifies it: there are zero `PERF_MULTICHIP_DECODE` signposts and zero `execute_trace` mentions after the warning, and the next server lines are `/metrics`, `Running: 0 reqs`, and shutdown. The TT vLLM runner now calls the model `reset_warmup_state()` hook when the scheduler is idle with no pending sample/async completions; for this adapter that releases decode and sampling traces before any future request recaptures or executes traces.

## Limitations

- Prefix caching is disabled.
- On-device sampling is greedy-only for the performance path. Shared tests that require host-only stochastic sampler semantics remain explicit compatibility behavior and do not replace the traced on-device serving path.
- CI burst metrics are secondary only.
