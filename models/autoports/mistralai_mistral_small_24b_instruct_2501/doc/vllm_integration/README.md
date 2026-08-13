# Mistral Small 24B vLLM integration

## Primary result

**Single-user 128-token prompt / 128-token generation, one request at concurrency 1:** TTFT P50/P99 **568.9/568.9 ms** and decode **52.8 tokens/s/user** from mean TPOT **18.928 ms**. ITL P50/P99 was **17.762/19.390 ms**, output throughput was **43.05 tokens/s**, and 1/1 request completed. This is the headline serving result.

The model serves through the shared OpenAI-compatible vLLM path on one TP4 P300 mesh. Decode uses the full-model generator's split Sampling1D token-out trace, with `supports_async_decode=True`; the production path has no adapter-local sampler, full-logits transfer, host argmax, generic top-k greedy fallback, or Python token-feedback loop.

## Configuration

- Model: `mistralai/Mistral-Small-24B-Instruct-2501`
- TT mesh: `P300x2` / one 1x4 TP mesh (four Blackhole devices)
- Advertised and served context: 32,768 tokens
- `max-num-seqs`: 32; page/block size: 32
- TT config: `{"sample_on_device_mode":"all","trace_region_size":200000000,"fabric_config":"FABRIC_1D"}`
- Production sampling: device split Sampling1D, greedy encoded as top-k 1 / top-p 0 / temperature 1, and sampled qualitative requests at top-k 32. The shared full compatibility suite may explicitly enable `MISTRAL_SMALL_24B_VLLM_HOST_SAMPLING_COMPAT=1`; this optional mode is not used for performance.

Server command:

```bash
TT_MISTRAL_TEXT_VER=mistral_small_24b_autoport \
PYTHONPATH="$PWD/vllm/plugins/vllm-tt-plugin/src:$PWD/vllm:$PWD" \
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir models/autoports/mistralai_mistral_small_24b_instruct_2501 \
  --hf-model mistralai/Mistral-Small-24B-Instruct-2501 \
  --mesh-device P300x2 --max-num-seqs 32 --block-size 32 \
  --max-model-len 32768 \
  --tt-config '{"sample_on_device_mode":"all","trace_region_size":200000000,"fabric_config":"FABRIC_1D"}'
```

The selected datatype policy is `bfp4_lofi_bfp8kv_bf16ccl`: BFP4_B attention and MLP weights for layers 0-39, BF16 embeddings/norms/LM head and activations/residuals, BFP8_B paged KV cache, and BF16 prefill/decode CCL payloads and workspace. Attention/MLP matmuls use LoFi, SDPA uses HiFi4, and the LM head uses HiFi2. There are no layer exceptions. The adapter constructs the canonical full-model generator, so serving consumes this policy rather than redefining it.

vLLM allocates and owns the serving cache. Every prefill and decode call must pass the exact cache object allocated by vLLM; the adapter rejects missing or substituted standalone caches. The scheduler's page table, current positions, reset information, and slot remaps are forwarded into the generator.

## Correctness and quality

The final full sampling profile passes **72 tests with one expected skip** on the explicit compatibility server; production performance remains on-device. Focused adapter/plugin tests cover stale host token/current-position rejection during async overlap, scheduler resets, changed page tables, caller-owned padded page tables, safe page-table tails after row reuse, trace capture/replay, and compatibility/production sampling routing. Decode trace capture is enabled in production.

The full 40-layer logit gate is also clean (`readiness_vllm/logit_determinism.json`). With a fresh caller-owned cache each time, production traced decode logits were bit-exact across two runs and at batch positions 0 and 1 (identical SHA256, max absolute difference 0, PCC 1.0, argmax token 1278). The standalone eager full-model baseline selected the same token and matched at PCC 0.999852 (max BF16-path difference 0.375). The diagnostic full-logit readback exists only in `run_logit_determinism.py`; it is not part of serving or benchmark execution.

A direct 37-token request (not divisible by the 32-token page size or tile/trace sizes) completed successfully. Realistic non-aligned chat prompts of 185, 186, 188, 191, and 200 tokens also generated correctly. The decisive fix was to stop compile/trace warmup from writing the real serving cache: eager compilation uses a disposable, exact-geometry scratch cache, capture records against the vLLM cache without replay, serving state is recopied, and the first real decode performs the first trace execution. Startup then resets the vLLM cache before serving.

All 12 final qualitative generations (six prompts, greedy and sampled) were read. They are coherent and on topic: valid ML haiku, accurate simple ML explanations, sensible story continuations, scientifically recognizable thermodynamics summaries, correct French translation, and useful Python Fibonacci implementations. None shows repetition loops, gibberish, wrong-language drift, request contamination, tokenizer markers, or mojibake. The sampled thermodynamics answer starts a fourth item after listing zeroth/first/second and is cut at 256 tokens; this is a mild count/ambiguity issue rather than serving corruption, and is recorded in the detailed verdict at `readiness_vllm/qualitative_verdict.md`.

`doc/context_contract.json` advertises 32,768 and provides the physical TP4 proof: all weights, BFP8 cache pairs, trace reservation, and a 1.5 GiB/rank runtime reserve fit, with a calculated ceiling of 34,464 and a physical decode at position 32,767. No hard limit requires reducing vLLM's advertised capability.

## Performance evidence

Primary single-user command and raw result are recorded in `readiness_vllm/vllm_benchmark.json` and `readiness_vllm/vllm_result.json`.

Secondary CI serving burst, **100-token prompt / 100-token generation, 32 requests at concurrency 32:** 32/32 completed; TTFT P50/P99 **1191.7/1192.8 ms**, TPOT P50/P99 **19.419/23.825 ms** (mean **19.618 ms**), ITL P50/P99 **17.935/67.958 ms**, output throughput **1026.97 tokens/s**, and mean-TPOT-derived **51.0 tokens/s/user**. These are capacity/CI metrics, not the headline decode rate: burst admission and chunked prefill can affect TPOT. Raw evidence is `readiness_vllm/vllm_ci_serving_benchmark.json` and `readiness_vllm/vllm_ci_serving_result.json`.

The optimized full-model teacher-forcing token-out measurement is **54.45 tokens/s/user** (18.365 ms/token), while its composed stack-plus-terminal accounting is **55.79 tokens/s/user** (17.925 ms/token). These are lower bounds only, not serving benchmarks. The **52.83 tokens/s/user** primary vLLM result leaves no material avoidable vLLM-specific per-token overhead in the measured decode path.

## Artifacts and limitations

- `readiness_vllm/sampling_tests.log`: final full sampling profile
- `readiness_vllm/vllm_qualitative_outputs.json`, `vllm_qualitative.log`, `vllm_qualitative_prompt_format.json`: final qualitative evidence
- `readiness_vllm/non_aligned_prompt_check.json`: direct non-aligned request
- `readiness_vllm/unsupported_sampling_survival.json`: production rejection/health survival audit
- `readiness_vllm/logit_determinism.json`: traced repeat/batch-position and standalone-baseline logit gate
- `readiness_vllm/vllm_benchmark.json`, `vllm_benchmark.log`, `vllm_result.json`: primary single-user benchmark
- `readiness_vllm/vllm_ci_serving_benchmark.json`, `vllm_ci_serving_benchmark.log`, `vllm_ci_serving_result.json`: CI serving burst
- `readiness_vllm/vllm_chat_template_exact_match.json`: exact model tokenizer/template validation

The optional host-sampling compatibility mode supports shared tests for features outside the device sampler's top-k-32/no-penalty/no-seed contract and gives unseeded stochastic compatibility tests fresh vLLM host RNG state. It is explicit, off by default, and never replaces traced on-device sampling for production serving or performance. Unsupported production requests are rejected before EngineCore admission; supported requests and `/health` remain live afterward. Final teardown stopped the owning runner cleanly and found no vLLM/API/EngineCore process. A recurring device-0 ERISC heartbeat timeout then required one bounded reset; all four boards listed and a 1x4 mesh opened and closed successfully afterward.
