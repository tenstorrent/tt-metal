# Gemma 4 26B A4B vLLM integration

**Primary single-user serving (prompt 128, output 128, 1 request, concurrency 1): TTFT P50/P99 532.8/532.8 ms; decode 21.78 t/s/u from mean TPOT 45.92 ms; ITL P50/P99 44.08/45.66 ms.**

Status: ready on a 1x4 P300C mesh through the shared vLLM TT plugin. The server advertises the context-contract maximum of 262,144 tokens, uses `max_num_seqs=32`, asynchronous decode, traced model decode, and the canonical full-model split-sampling path.

## Final server configuration

```text
python -m vllm.entrypoints.openai.api_server --model google/gemma-4-26B-A4B-it --block_size 64 --max_num_seqs 32 --port 8000 --max_model_len 262144 --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":220000000,"fabric_config":"FABRIC_1D_RING"}}'
```

The selected precision policy is BF16 activations/residual/CCL/KV cache, BFP8 attention/dense/expert weights, BF16 norms/embedding/lm-head, FP32 routing, packed decode MLP BFP4, sliding attention HIFI2, and full attention/dense/expert compute LOFI. There are no layer exceptions.

## Evidence

- Sampling profile `full`: 72 passed, 1 skipped in 732.98 s on the final adapter. The performance path samples on device; host sampling remains explicit and optional for unsupported shared-test features.
- Non-aligned serving: exact 47-token prompt and a 2051-token prompt crossing the 2048 prefill chunk boundary both passed.
- Qualitative: six chat-templated prompts, greedy and sampled, are coherent and on-topic with no repetition collapse, gibberish, wrong-language drift, or request contamination. The haiku exactly matches the accepted full-model TT result. Mechanical degeneracy check: clean.
- CI serving burst (prompt 100, output 100, 32 requests, unconstrained admission): 32/32 complete; TTFT P50/P99 3949.9/5956.4 ms; TPOT P50/P99 424.7/450.7 ms; ITL P50/P99 369.2/1520.8 ms; aggregate output throughput 69.6 tok/s. This burst TPOT is secondary and is not the headline decode t/s/u.
- Full-model canonical token-out is 23.76 t/s/u and teacher forcing is 25.40 t/s/u; vLLM's 21.78 t/s/u is within 8.4% of canonical token-out. The prior padded-batch path was fixed, and no avoidable vLLM-specific decode graph overhead remains. No host logits readback/token-feedback loop or generic greedy fallback is present in the measured path.
- Context capacity: a live request with 262,143 input tokens plus one output token completed in 317.6 s (`vllm_near_max_context_result.json`). The server's 44,032-token cache log is a per-hybrid-group accounting summary; its reported 1.80x concurrency at 262,144 and this direct request prove one full-contract request fits.
- Fallback audit: the final qualitative and benchmark runner used `TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}'`; the raw server log records the setting and every request passed.
- Async overlap: chat-templated isolated greedy controls exactly match staggered concurrent outputs for a coherent 96-token decode crossing a 64-token page boundary and a two-word request that reaches EOS after 3 tokens. The runner rejects doubled tokens, repeated phrases, and dominant-token collapse; all four outputs pass. The retained SHA-256 pairs prove stable device feedback, exactly-once position advance, page-table refresh correctness, and deferred-read fencing (`async_overlap_state_test.json`).
- Final runner cleanup terminated the server cleanly. No EngineCore/vLLM process retained the devices, no TT fatal occurred, and the final log contains no unsafe-active-trace allocation warning.

Artifacts are under `readiness_vllm/`: `vllm_result.json`, `vllm_benchmark.json`, `vllm_ci_serving_result.json`, `vllm_ci_serving_benchmark.json`, `vllm_near_max_context_result.json`, `async_overlap_state_test.json`, both benchmark logs, `sampling_tests.log`, `vllm_qualitative_outputs.json`, `degeneracy_check.log`, and `non_aligned_and_chunked_requests.json`.

Limitations: prefix caching is disabled by the shared TT backend. Top-k above 32 and features requiring full host logits use the explicit compatibility mode and are excluded from performance claims.
