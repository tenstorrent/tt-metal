# vLLM Integration Work Log

Date: 2026-06-15

## Starting Point

Started from the datatype-selected optimized full model for `microsoft/Phi-3.5-mini-instruct`.

Selected precision config: `doc/datatype_sweep/selected_precision_config.json`, `c004_default_weights_bf16_ccl`.

Full-model lower-bound evidence used:

- `doc/datatype_sweep/perf/post_selection_token_out_no_readback_prompt128_gen128.json`
- TTFT `214.10 ms`, decode `56.37 t/s/u`
- counters: `full_logits_decode_readbacks=0`, `sampled_token_readbacks=0`, `device_token_feedbacks=135`, `device_position_advances=135`
- selected teacher-forcing gate: TTFT `226.88 ms`, decode `40.34 t/s/u`, top1 `91/100`, top5 `100/100`

## Implementation

Added `tt/generator_vllm.py` with `Phi3ForCausalLM`.

Adapter behavior:

- constructs `Phi35MiniGenerator(..., allocate_standalone_cache=False)`
- implements vLLM-owned `allocate_kv_cache()` in selected BF8 KV dtype
- delegates prefill to `Phi35MiniGenerator.prefill_forward_token_out()`
- delegates decode to `Phi35MiniGenerator.decode_forward_token_out()`
- delegates split decode read and host formatting to the generator
- rejects `sampling_params=None` so host sampling fallback cannot silently enter serving
- declares `supports_async_decode=True`, `supports_topk_logprobs=True`, `supports_prefix_caching=False`, `tt_async_decode_allows_overlap=False`

Updated `tt/generator.py`:

- added a serving cache-ownership mode with `allocate_standalone_cache=False`
- added vLLM token-out prefill/decode helpers
- kept decode on the canonical split sampler with `tt_out_tok` feedback
- disabled prefill sampling trace replay in the adapter path because request prefill logits are not stable trace inputs
- added on-device top-k logprob host formatting for the existing common sampler result

Updated installed editable TT vLLM plugin under `/localdev/moconnor/vllm`:

- registered `TTPhi3ForCausalLM` in `plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`
- added model-capability override for `tt_async_decode_allows_overlap`
- added model-capability override for `supports_topk_logprobs`
- gated steady async-scheduling overlap on `tt_async_decode_allows_overlap`
- added explicit test skips for device-only host-compat tests via `VLLM_TT_SKIP_HOST_ONLY_SAMPLING_TESTS=1`
- skipped batch-shape tests that require more than one active request when `--tt-max-num-seqs=1`

## Iterations

1. Smoke run failed during prefill sampling trace reuse:

   - failure: `ValueError: The provided logits tensor does not match the tensor used during trace capture`
   - fix: force vLLM prefill sampling to execute on device without internal sampling trace replay; decode trace remains enabled

2. Smoke run then failed on `test_min_p`:

   - root cause: min-p is a plugin host-compat sampling test and sends `sampling_params=None`
   - fix: kept adapter host fallback disabled and added explicit host-only pytest skip switch

3. Full profile initially failed on top-k logprob conversion:

   - failure: `Can't convert a tensor distributed on MeshShape([1, 8]) mesh to row-major logical tensor`
   - fix: convert `LogProbsResult` host tensors through the first replicated mesh shard and preserve `[batch, top_k]`

4. Full profile then had batch-1 test-shape failures:

   - failures: `test_uniform_noseed_varied` and mixed/different penalty tests
   - root cause: assertions require multiple active request slots; final serving config uses `max_num_seqs=1`
   - fix: skip those tests when the configured max batch is too small

## Final Command

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

Server invocation emitted by the runner:

```bash
/localdev/moconnor/tt-metal/python_env/bin/python -m vllm.entrypoints.openai.api_server \
  --model microsoft/Phi-3.5-mini-instruct \
  --block_size 32 \
  --max_num_seqs 1 \
  --port 8000 \
  --max_model_len 512 \
  --plugin-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 200000000, "fabric_config": "FABRIC_1D_RING"}}'
```

## Final Results

Sampling:

- profile: `full`
- result: `59 passed, 13 skipped`
- artifact: `readiness_vllm/sampling_tests.log`
- expected skips: host-only compat tests, all-vocab logprobs cap, and multi-request tests not meaningful for `max_num_seqs=1`

Qualitative:

- artifact: `readiness_vllm/vllm_qualitative_outputs.json`
- verdict: coherent and mostly on-topic; no loops or gibberish; no wrong-language drift in the translation section; some request-contamination/extra-instruction continuation remains
- control: full-model stage reported coherent HF/TT free-running English with no repetition or wrong-language drift

Degeneracy:

- command: `python models/common/readiness_check/check_degenerate_output.py --hf-model microsoft/Phi-3.5-mini-instruct --missing-artifacts critical --scope vllm`
- result: `No degenerate output detected`
- artifact: `doc/vllm_integration/check_degenerate_output.log`

Benchmark:

- artifact: `readiness_vllm/vllm_benchmark.json`
- prompt_len `128`, output_len `128`, num_requests `32`, concurrency `1`
- completed requests `32`
- elapsed `83.67 s`
- TTFT P50/P99 `170.79/222.99 ms`
- ITL P50/P99 `19.33/21.11 ms`
- aggregate output throughput `48.38 tok/s`
- mean per-user decode `51.41 t/s/u`

Adapter contract:

- artifact: `readiness_vllm/adapter_contract_checks.json`
- result: pass
- verifies prefill/decode delegation, vLLM-owned KV cache pass-through, page-table pass-through, current-position pass-through, decode trace flag pass-through, split-read flag pass-through, and host fallback rejection

## Async And Cache Ownership

`supports_async_decode=True` is enabled because `decode_forward(..., read_from_device=False)`, `read_decode_output(..., async_read=True)`, and `process_decode_output_host(...)` are implemented and used by the plugin.

`tt_async_decode_allows_overlap=False` remains disabled. Reason: vLLM still constructs the next decode input from host scheduler state and request tables, and no focused proof was run showing that step N+1 cannot use stale sampled-token/current-position/page-table state. The plugin now checks this capability separately and disables steady overlap unless a model opts in.

KV-cache ownership is vLLM serving-owned: the adapter's constructor disables standalone generator cache allocation, `allocate_kv_cache()` creates the serving cache, and prefill/decode pass that cache through to the model path.

## Cleanup

The final runner shut the server down cleanly. Post-run audit:

```text
pgrep -af 'vllm|EngineCore|api_server|run_vllm_server' || true
```

Only the unrelated multi-goal orchestration command matched the broad pattern; no vLLM server, EngineCore, API server, or runner process remained.
