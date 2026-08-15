# Qwen3.6-27B vLLM integration

**Primary single-user serving (max-num-seqs 1; 128 input / 128 output, 1 request, concurrency 1): TTFT P50/P99 4,139/4,139 ms; TPOT mean/P50/P99 70.7/70.7/70.7 ms; ITL P50/P99 55.9/57.5 ms; TPOT-derived decode 14.14 t/s/u; output throughput 9.75 tok/s.**

Secondary capacity/CI serving burst (max-num-seqs 32; 100 input / 100 output, 32 requests, unconstrained admission): TTFT P50/P99 165,477/165,478 ms; TPOT mean/P50/P99 280.1/254.2/824.7 ms; ITL P50/P99 244.0/578.1 ms; output throughput 16.78 tok/s. The burst TPOT-derived 3.57 t/s/u is not the headline decode number because admission and batched prefill affect it.

## Status

The shared vLLM TT path serves `Qwen/Qwen3.6-27B` on four Blackhole p300c devices (`P300x2`, physical mesh 1x4). The adapter declares `supports_async_decode=True`, uses traced on-device decode sampling, delegates to the full-model generator's canonical split token-out methods, and passes vLLM-owned paged K/V through without a standalone cache. Full plugin sampling passed 72 tests with 1 skip. A direct 65-token non-aligned prompt completed normally. The served `max_model_len=262144` exactly matches `doc/context_contract.json`.

The primary max-num-seqs-1 server allocates 1,727,200 cache tokens (2,159 page-800 blocks including vLLM lookahead), providing 6.58x the full context. The capacity profile at max-num-seqs 32 allocates 1,752,000 tokens (2,190 blocks). No advertised context reduction is used.

## Configuration

- Primary server: `max_model_len=262144`, `max_num_seqs=1`, block size 64, `FABRIC_1D_RING`, trace region 200,000,000 bytes. The secondary CI capacity profile uses `max_num_seqs=32`.
- Sampling: `sample_on_device_mode=all`; optional host sampling is used only for unsupported shared-test features and preserves slot remaps. It is not the performance path.
- Precision: selected datatype-sweep policy `full_attention_bfp4_lofi`; BFP4 projection/MLP weights, BF16 activations and CCL payloads, BFP8 KV cache and linear recurrent state, LoFi projections/attention/MLP, HiFi2 recurrent update, and BFP8/HiFi2 LM head. The selected policy has no layer exceptions.
- Linear prefill uses 32-token recurrent scan chunks. This halves fragmented DRAM temporaries while preserving token-by-token recurrent equivalence; decode behavior and full context are unchanged.

## Commands

Full sampling launch (sampling passed; the benchmark phase in this combined run
then exposed the prefill allocator failure described in `AUTOFIX.md`):

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages serve,sampling,benchmark \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --hf-model Qwen/Qwen3.6-27B --mesh-device P300x2 \
  --max-num-seqs 32 --max-model-len 262144 \
  --sampling-profile full \
  --tt-config '{"trace_region_size": 200000000, "fabric_config": "FABRIC_1D_RING"}'
```

After AutoFix, the successful primary and CI benchmark launch was:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages serve,benchmark \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --hf-model Qwen/Qwen3.6-27B --mesh-device P300x2 \
  --max-num-seqs 32 --max-model-len 262144 \
  --sampling-profile full \
  --tt-config '{"trace_region_size": 200000000, "fabric_config": "FABRIC_1D_RING"}'
```

The final headline capacity-isolation/fixed configuration launch was:

```bash
python -m models.common.readiness_check.run_vllm_server \
  --stages serve,benchmark \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --hf-model Qwen/Qwen3.6-27B --mesh-device P300x2 \
  --max-num-seqs 1 --max-model-len 262144 \
  --sampling-profile full --no-benchmark-ci-serving \
  --tt-config '{"trace_region_size": 200000000, "fabric_config": "FABRIC_1D_RING"}'
```

The exact `vllm bench serve` child commands and workload shapes are retained in
the two normalized benchmark JSON files.

Host contracts:

```bash
pytest -q models/autoports/qwen_qwen3_6_27b/tests/test_full_model_public_contract.py \
  models/autoports/qwen_qwen3_6_27b/tests/test_vllm_adapter_contract.py
```

## Evidence

- `readiness_vllm/sampling_tests.log`: final full profile, 72 passed / 1 skipped.
- `readiness_vllm/vllm_result.json` and `vllm_benchmark.json`: primary max-num-seqs-1 raw and normalized results.
- `readiness_vllm/capacity_ab/maxseq{1,32}/`: same-path fixed-capacity A/B server logs and raw/normalized benchmark artifacts.
- `readiness_vllm/vllm_ci_serving_result.json` and `vllm_ci_serving_benchmark.json`: secondary CI burst.
- `readiness_vllm/non_aligned_prompt_65_tokens.json`: exact 65-token prompt, 4-token response, normal length finish.
- `readiness_vllm/vllm_chat_qualitative_outputs.json`: six chat-template-correct prompts, greedy and sampled.
- `readiness_vllm/vllm_chat_prompt_metadata.json`: rendered prompts, exact token IDs, tokenizer snapshot/class, and datatype/full-model control links.
- `readiness_vllm/degenerate_output_check.json` and `.log`: final vLLM-scope qualitative gate, passed.
- `doc/vllm_integration/logs/reduced_target_stale_input.log`: direct adapter trace gate for stale token/current-position/unchanged-page-table inputs and slot remapping.
- `readiness_vllm/server.log`: selected precision, trace mode, cache capacity, and serving lifecycle.
- `doc/vllm_integration/AUTOFIX.md`: isolated hypothesis/fix/refutation history.

## Qualitative verdict and limitations

All 12 chat outputs were read. They are coherent, on-topic, non-repetitive, grammatical, and free of gibberish, wrong-language drift, or request contamination. However, with a 128-token output cap every response exposes a structured reasoning preamble and reaches the length limit before its final answer. This is a real instruction-fulfillment/presentation limitation for short response budgets, not a model-collapse signal. The older `vllm_qualitative_outputs.json` used raw completions without the checkpoint chat template and is retained only as invalid-run diagnostic evidence.

The targeted same-path A/B changed only fixed execution capacity. At max-num-seqs 32, the 128/128 single-user profile measured 251.7 ms TPOT and 3.97 t/s/u; at max-num-seqs 1 it measured 70.7 ms and 14.14 t/s/u. The sampler remains padded to its canonical 32 rows in both arms, max context remains 262,144, and the cache difference is only 31 lookahead blocks. This isolates inactive-slot model trace work as the dominant avoidable overhead. The primary configuration therefore uses max-num-seqs 1; max-num-seqs 32 remains the explicit capacity/CI profile. Primary serving now exceeds the 6.96 t/s/u full-model teacher-forcing lower bound and reaches 80.9% of the 17.467 t/s/u canonical batch-1 split-token path. The residual is the required async serving/event boundary; there is still no second sampler, host token feedback, or logits readback.

Server shutdown completed cleanly after the successful benchmarks and final evidence session. No live vLLM or EngineCore process was left holding the devices.
