# Phi-3.5-mini vLLM Integration

Batch-1 vLLM serving status: **passes shared runner with full sampling profile**. Prefill TTFT P50/P99 is **170.8 ms / 223.0 ms** and decode is **51.4 t/s/u mean per user** on prompt_len=128, output_len=128, 32 requests, concurrency=1. ITL P50/P99 is **19.3 ms / 21.1 ms** and aggregate output throughput is **48.4 tok/s**.

## Result

- Model: `microsoft/Phi-3.5-mini-instruct`
- Adapter: `models/autoports/microsoft_phi_3_5_mini_instruct/tt/generator_vllm.py`
- vLLM registration: `/localdev/moconnor/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`, arch `TTPhi3ForCausalLM`
- Max model length: `512`
- Max num seqs: `1`
- Block size: `32`
- Mesh/config: T3K, `FABRIC_1D_RING`, `trace_region_size=200000000`, `sample_on_device_mode=all`
- Sampling result: `59 passed, 13 skipped` in `readiness_vllm/sampling_tests.log`
- Qualitative verdict: coherent and mostly on-topic; no mechanical repetition, gibberish, or wrong-language drift. Limitation: several completions continue into extra instruction/answer text, and one sampled Fibonacci/translation output contains typo-like wording.
- Degeneracy check: pass, `No degenerate output detected`

## Command

```bash
VLLM_TT_SKIP_HOST_ONLY_SAMPLING_TESTS=1 python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/microsoft_phi_3_5_mini_instruct \
  --hf-model microsoft/Phi-3.5-mini-instruct \
  --mesh-device T3K \
  --max-num-seqs 1 \
  --max-model-len 512 \
  --block-size 32 \
  --sampling-profile full \
  --server-timeout 1800 \
  --benchmark-prompt-len 128 \
  --benchmark-output-len 128 \
  --benchmark-num-requests 32 \
  --benchmark-concurrency 1 \
  --tt-config '{"trace_region_size": 200000000, "fabric_config": "FABRIC_1D_RING"}'
```

The runner launched:

```bash
/localdev/moconnor/tt-metal/python_env/bin/python -m vllm.entrypoints.openai.api_server \
  --model microsoft/Phi-3.5-mini-instruct \
  --block_size 32 \
  --max_num_seqs 1 \
  --port 8000 \
  --max_model_len 512 \
  --plugin-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 200000000, "fabric_config": "FABRIC_1D_RING"}}'
```

## Precision Policy

Serving uses `doc/datatype_sweep/selected_precision_config.json`, config `c004_default_weights_bf16_ccl`:

- attention qkv/o weights: BF8 decode and prefill
- MLP gate_up/down: BF4 decode, BF8 prefill
- embeddings, norms, lm_head: BF16
- activation/residual/logits: BF16
- KV cache: BF8
- CCL: sync all-reduce with BF16 decode/prefill dtype
- compute fidelities: decode matmul/LM head/SDPA LoFi; prefill matmul/LM head HiFi2; norm and prefill SDPA HiFi4
- layer exceptions: none

## Adapter Contract

`Phi3ForCausalLM` is a thin vLLM wrapper around `Phi35MiniGenerator`. It constructs the generator with `allocate_standalone_cache=False`, so vLLM owns attention KV cache allocation. `allocate_kv_cache()` allocates the serving cache in selected BF8 KV dtype and the adapter passes that same cache object through `prefill_forward_token_out()` and `decode_forward_token_out()`.

Serving decode uses the full-model canonical split-sampling path: traced model replay, common TT sampler, `tt_out_tok` device feedback, and split async read/host formatting. The adapter has no `argmax`, no host sampler, no full-logits conversion/readback, and no generic top-k greedy fallback. Host-only sampling fallback is rejected; the host-only plugin tests are skipped under `VLLM_TT_SKIP_HOST_ONLY_SAMPLING_TESTS=1`.

Capabilities:

- `supports_async_decode=True`
- decode trace enabled by plugin default `trace_mode=all` plus `trace_region_size=200000000`
- `supports_topk_logprobs=True` using the shared on-device top-k logprob path
- `supports_prefix_caching=False`
- `tt_async_decode_allows_overlap=False`

`tt_async_decode_allows_overlap` stays false because vLLM still builds step N+1 inputs from host-side scheduler state (`token_ids_cpu`, positions, and block tables). The plugin now gates steady async overlap on this separate capability. No overlap proof was run, so overlap remains disabled by design.

## Evidence

Artifacts:

- `readiness_vllm/server.log`
- `readiness_vllm/sampling_tests.log`
- `readiness_vllm/vllm_qualitative_outputs.json`
- `readiness_vllm/vllm_benchmark.json`
- `readiness_vllm/adapter_contract_checks.json`
- `doc/vllm_integration/check_degenerate_output.log`

Sampling skips are expected for this batch-1, device-only profile:

- host-only compat tests: `min_p`, `bad_words`, `logit_bias`, `allowed_token_ids`, `min_tokens`
- all-vocab chat logprobs: skipped by server max-logprobs cap
- multi-request variety/penalty tests whose assertions require more than one active batch slot

The adapter contract check passed and verifies prefill/decode delegation, vLLM-owned KV-cache pass-through, page-table/current-position pass-through, decode trace flag pass-through, split-read flag pass-through, and host fallback rejection.

## Performance

Serving benchmark (`readiness_vllm/vllm_benchmark.json`):

| Metric | Value |
|---|---:|
| Completed requests | 32 |
| Elapsed | 83.67 s |
| TTFT P50 / P99 | 170.79 / 222.99 ms |
| ITL P50 / P99 | 19.33 / 21.11 ms |
| Output throughput | 48.38 tok/s |
| Mean per-user decode | 51.41 t/s/u |

Full-model lower bounds for comparison only:

- selected datatype token-out no-readback: TTFT `214.10 ms`, decode `56.37 t/s/u`, zero full-logits decode readbacks
- selected datatype teacher-forcing gate: TTFT `226.88 ms`, decode `40.34 t/s/u`

Serving decode is close to the token-out lower bound and above the teacher-forcing lower-bound number, with no avoidable host sampling/full-logit readback path in the measured serving route.

## Qualitative Verdict

Readout over the six vLLM prompt pairs:

- coherence/topic: greedy outputs are coherent and on-topic; sampled outputs are mostly coherent and on-topic
- repetition: no loops; degeneracy checker found no degenerate output
- gibberish: none
- wrong-language drift: none in the French translation; sampled output contains a typo-like phrase but remains in French for the translated section
- request contamination: present in several outputs as extra `Instruction 2`, `Answer`, or unrelated follow-on prompt text after the requested answer

The full-model stage control reported coherent English free-running output with no repetition or wrong-language drift. The vLLM continuation artifacts should be tracked as a qualitative serving limitation, not as mechanical token-feedback degeneration.

## Cleanup

The runner terminated the server cleanly. A post-run process audit found no leftover `vllm`, `EngineCore`, `api_server`, or `run_vllm_server` processes holding devices; only the unrelated multi-goal orchestration process matched the broad `vllm` grep pattern.
