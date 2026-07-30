# vLLM Integration

Batch-1 serving result: prefill TTFT P50/P99 is 52.56/62.82 ms; decode is
116.01 t/s/u per user mean, with ITL P50/P99 7.52/10.06 ms, for prompt
128/output 128, 8 requests, concurrency 1.

Sampling status: `--sampling-profile full` completed with 19 selected tests
passing and 42 adapter-inapplicable tests deselected for batch-1,
no-host-fallback serving. Qualitative verdict: greedy outputs are mostly
coherent and on-topic; sampled outputs are not coherent, although the
degenerate-output checker found no stale-token/position mechanical signature.

Model: `meta-llama/Llama-3.2-1B-Instruct`

Mesh: T3K `1x8`, `FABRIC_1D_RING`

Status: serves through the shared vLLM TT path using the autoport adapter
`models.autoports.meta_llama_llama_3_2_1b_instruct.tt.generator_vllm:Llama32ForCausalLM`.

## Serving Configuration

| Item | Value |
| --- | --- |
| `TT_LLAMA_TEXT_VER` | `autoport_llama32_1b` |
| TT config | `{"trace_region_size": 100000000, "fabric_config": "FABRIC_1D_RING", "sample_on_device_mode": "all"}` |
| Max model length | 4096 |
| Max num seqs | 1 |
| Block size | 64 |
| Sampling profile | `full` with adapter-specific no-host-fallback selection |
| Workload | prompt 128, output 128, 8 requests, concurrency 1 |

Final successful invocation:

```bash
TT_LLAMA_TEXT_VER=autoport_llama32_1b \
PYTHONPATH=/localdev/moconnor/tt-metal:/localdev/moconnor/vllm:/localdev/moconnor/vllm/plugins/vllm-tt-plugin/src:${PYTHONPATH:-} \
LD_LIBRARY_PATH=/localdev/moconnor/tt-metal/build_Release/lib:${LD_LIBRARY_PATH:-} \
python models/common/readiness_check/run_vllm_server.py \
  --model-dir models/autoports/meta_llama_llama_3_2_1b_instruct \
  --hf-model meta-llama/Llama-3.2-1B-Instruct \
  --stages serve,sampling,qualitative,benchmark \
  --mesh-device T3K \
  --port 8016 \
  --max-num-seqs 1 \
  --max-model-len 4096 \
  --block-size 64 \
  --sampling-profile full \
  --server-timeout 1200 \
  --benchmark-prompt-len 128 \
  --benchmark-output-len 128 \
  --benchmark-num-requests 8 \
  --benchmark-concurrency 1 \
  --tt-config '{"trace_region_size": 100000000, "fabric_config": "FABRIC_1D_RING", "sample_on_device_mode": "all"}' \
  --additional-server-args=--disable-log-requests
```

## Metrics

| Metric | Value |
| --- | ---: |
| Completed requests | 8 |
| Elapsed | 8.1319 s |
| Total output tokens | 901 |
| TTFT P50 | 52.5598 ms |
| TTFT P99 | 62.8154 ms |
| TTFT mean | 54.5469 ms |
| ITL P50 | 7.5216 ms |
| ITL P99 | 10.0567 ms |
| ITL mean | 7.6129 ms |
| Aggregate output throughput | 110.7980 tok/s |
| Request throughput | 0.9838 req/s |
| Mean per-user decode throughput | 116.0061 t/s/u |

Teacher-forcing comparison is only a lower-bound sanity check. The selected
datatype-sweep full-model teacher-forcing result is TTFT 183.03 ms and
trace-verified decode 92.68 t/s/u. The vLLM measurement uses a different
serving workload, so this is not a speedup claim; it confirms the shared vLLM
path is not below the selected full-model lower-bound evidence.

## Precision Policy

The vLLM path consumes
`../datatype_sweep/selected_precision_config.json`, config
`cfg08_bfp8_weights_bfp8_kv_bf16_ccl`.

| Group | Dtype / setting |
| --- | --- |
| Token embeddings | `bfloat16` |
| Attention QKV and WO weights | `bfloat8_b`, layers 0-15 |
| MLP gate/up and down weights | `bfloat8_b`, layers 0-15 |
| RMSNorm and final norm | `bfloat16` |
| LM head | `bfloat8_b` |
| Activation, residual, mul | `bfloat16` |
| KV cache | `bfloat8_b`, paged local one-KV-head cache per chip |
| CCL all-gather/reduce-scatter | `bfloat16`, persistent buffers enabled |
| Layer exceptions | none |

Compute fidelity is HiFi2 fp16 accumulate for decode matmuls, prefill QKV/WO,
MLP, RMSNorm, and LM head; prefill SDPA uses HiFi4 fp32 destination accumulate.
The adapter applies the selected CCL environment overrides:
`MD_MULTICHIP_ALL_GATHER_DTYPE=bfloat16`,
`MD_MULTICHIP_REDUCE_SCATTER_DTYPE=bfloat16`, and
`MD_MULTICHIP_USE_PERSISTENT_CCL_BUFFERS=1`.

## Adapter Contract

`generator_vllm.py` subclasses the full-model generator and delegates to its
low-level methods. Prefill calls `prefill_token_out_host()` and decode calls
`decode_token_out_device_for_vllm()`. The adapter uses the full-model canonical
token-out sampling path: split greedy top-1 for greedy requests and
`SamplingGenerator` for sampled requests. It does not implement a separate
sampler, host argmax path, full-logits readback path, generic top-k greedy
fallback, or Python token readback/writeback feedback loop.

KV-cache ownership is vLLM-owned. The adapter's `allocate_kv_cache()` builds the
serving cache in the selected KV dtype, binds it through `bind_external_kv_cache()`,
and all prefill/decode calls require the same cache object. The model is
constructed with `use_vllm_paged_kv_cache=True`, so hidden standalone-cache
assumptions are disabled for serving.

Capabilities:

| Capability | Value | Evidence |
| --- | --- | --- |
| `supports_async_decode` | `True` | adapter capability and plugin registration |
| Decode trace | enabled | adapter contract test |
| `tt_async_decode_allows_overlap` | `False` | no async-scheduling/on-device-sampling overlap proof yet |
| Host sampling fallback | disabled | adapter and platform reject host/compat fallback |
| Device logprobs | disabled | vLLM logprob API requests rejected for this model |

The async-overlap flag is separate from async-decode support. It remains
`False`; the plugin gates the steady async overlap fast path on that flag, so
step N+1 cannot consume stale sampled-token/current-position state through the
overlap path until a dedicated proof exists.

## Readiness Evidence

Sampling tests:

- Artifact: `../../readiness_vllm/sampling_tests.log`
- Result: 19 passed, 42 deselected, 16 warnings.
- Selection artifact: `../../readiness_vllm/sampling_selection.json`
- Deselection reason: host-only, compatibility/logprob API, and batch-variation
  tests are not applicable to this batch-1, `sample_on_device_mode="all"`,
  no-host-fallback adapter.

Adapter stale-state contract:

- Command: `pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_generator_vllm.py::test_vllm_adapter_trace_input_refresh_contract`
- Artifact: `../../readiness_vllm/adapter_trace_input_refresh_contract.json`
- Result: passed.
- Steady unchanged page table: trace replay and device token feedback occurred;
  token/current-position/RoPE/page-table refreshes, syncs, readbacks, host
  argmax decode steps, and full-logits decode readbacks were all zero.
- Changed page table: only page table refresh changed.
- Reset batch: token/current-position/RoPE/page-table inputs refreshed.

Degenerate-output check:

- Command: `python models/common/readiness_check/check_degenerate_output.py --model-dir models/autoports/meta_llama_llama_3_2_1b_instruct --hf-model meta-llama/Llama-3.2-1B-Instruct --missing-artifacts critical --scope vllm --json models/autoports/meta_llama_llama_3_2_1b_instruct/readiness_vllm/degenerate_output_report.json`
- Artifact: `../../readiness_vllm/degenerate_output_report.json`
- Result: exit code 0, no degenerate output detected.
- Highest sampled adjacent duplication was 0.0144 and highest sampled trigram
  loop fraction was 0.0238, both below checker thresholds.

Process cleanup:

- Audit command: `pgrep -af 'run_vllm_server|pytest .*vllm-tt-plugin|vllm\.entrypoints|EngineCore|api_server' || true`
- Result after the final successful run: no leftover vLLM, pytest, API server,
  or EngineCore process beyond the audit command itself.
- Device availability check after cleanup: `ttnn.GetNumAvailableDevices()` returned 8.

## Qualitative Verdict

`../../readiness_vllm/vllm_qualitative_outputs.json` was read manually.

Greedy completions are mostly coherent and on-topic. The supervised-learning,
story, thermodynamics, and Fibonacci prompts are understandable; the haiku is
on-topic but repetitive; the translation answer is related but does not provide
the direct French translation requested.

Sampled completions with `temperature=0.7`, `top_p=0.9` are not acceptable:
all six sampled outputs drift into gibberish, code/path fragments,
multilingual/token noise, and request contamination. No stable wrong-language
mode or tight repetition loop dominates, but the sampled text is not coherent.
The degenerate-output checker found no stale-token/position mechanical
signature, so this is recorded as a sampled-quality limitation of the current
shared on-device sampling path, not bypassed by a host fallback.

## Artifacts

Artifacts live under `../../readiness_vllm/`:

- `adapter_trace_input_refresh_contract.json`
- `degenerate_output_report.json`
- `sampling_selection.json`
- `sampling_tests.log`
- `server.log`
- `vllm_benchmark.json`
- `vllm_qualitative_outputs.json`

## Limitations

- vLLM serving is batch-1 only for this adapter; startup rejects
  `--max_num_seqs` values other than 1.
- Host/compat sampling fallback and vLLM logprob API support are disabled.
- Sampled output quality is poor under the qualitative `temperature=0.7`,
  `top_p=0.9` profile.
- `tt_async_decode_allows_overlap=False` until async-scheduling overlap is
  proven with on-device sampling.
- Prefix caching is disabled.
