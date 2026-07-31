# vLLM Integration Work Log

Date: 2026-06-15

Model: `meta-llama/Llama-3.2-1B-Instruct`

Mesh: T3K `1x8`, `FABRIC_1D_RING`

## Code Changes

- Added `tt/generator_vllm.py` with `Llama32ForCausalLM`, the vLLM adapter for
  the datatype-selected full-model generator.
- Added vLLM-owned cache binding through `allocate_kv_cache()` and
  `bind_external_kv_cache()`, using the selected `bfloat8_b` KV-cache dtype.
- Added `use_vllm_paged_kv_cache` plumbing through the full model and optimized
  decoder so serving does not assume a hidden standalone cache.
- Added vLLM prefill/decode token-out helpers to `tt/generator.py`, delegating
  sampling to the existing split greedy path or shared `SamplingGenerator`.
- Added selected precision CCL environment propagation from the config file.
- Registered `TT_LLAMA_TEXT_VER=autoport_llama32_1b` in the vLLM TT plugin.
- Added plugin capability propagation for `tt_async_decode_allows_overlap`,
  host-fallback support, and device-logprob support.
- Gated async decode overlap on `tt_async_decode_allows_overlap`; async decode is
  supported, but overlap remains disabled for this adapter.
- Added request validation so host/compat sampling fallback and logprob API
  requests are rejected instead of silently falling back.
- Added `test_generator_vllm.py::test_vllm_adapter_trace_input_refresh_contract`.
- Updated `run_vllm_server.py` to emit sampling selection artifacts and to run
  the no-host-fallback, batch-1 applicable test subset for this adapter.

## Syntax And Registration

Syntax check:

```bash
python -m py_compile \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tt/optimized_decoder.py \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tt/model.py \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tt/generator.py \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tt/generator_vllm.py \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_generator_vllm.py \
  models/common/readiness_check/run_vllm_server.py
```

Result: passed.

vLLM plugin files were also compiled:

```bash
python -m py_compile \
  plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py \
  plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py \
  plugins/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py
```

Result: passed.

Registration smoke confirmed `TT_LLAMA_TEXT_VER=autoport_llama32_1b` resolves to
`models.autoports.meta_llama_llama_3_2_1b_instruct.tt.generator_vllm:Llama32ForCausalLM`
and exposes the expected capabilities.

## Adapter Contract Test

Command:

```bash
pytest --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_generator_vllm.py::test_vllm_adapter_trace_input_refresh_contract
```

Result: passed. Artifact:
`models/autoports/meta_llama_llama_3_2_1b_instruct/readiness_vllm/adapter_trace_input_refresh_contract.json`.

Recorded proof:

- `supports_async_decode=true`
- `tt_async_decode_allows_overlap=false`
- `decode_trace_enabled=true`
- KV cache owner: `vLLM adapter allocate_kv_cache`
- Unchanged page table replay: zero token/current-position/RoPE/page refreshes,
  zero syncs, zero readbacks, zero host argmax steps, zero full-logits readbacks.
- Changed page table replay: one page-table refresh only.
- Reset batch replay: token/current-position/RoPE/page-table refreshes occur.

## Server Attempts

Smoke serving with batch 1 passed after fixing `top_k=1` greedy normalization
and filtering host-only smoke tests for the no-host-fallback adapter.

A full `--max-num-seqs 8` attempt was rejected as incompatible with the current
adapter: prefill and vLLM serving are batch-1 only. The adapter now validates
this at startup and instructs callers to use `--max_num_seqs 1`.

An intermediate logprob-path run hung and was killed. The following startup hit
a TT ETH heartbeat error, so local boards were reset with `tt-smi -r all`.
Subsequent `ttnn.GetNumAvailableDevices()` returned 8.

## Final Server Run

Command:

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

Result: succeeded.

Sampling result: 19 passed, 42 deselected, 16 warnings. Deselections are
recorded in `readiness_vllm/sampling_selection.json` and cover host-only,
compat/logprob API, and batch-variation tests that are outside this adapter's
contract.

Benchmark result from `readiness_vllm/vllm_benchmark.json`:

| Metric | Value |
| --- | ---: |
| Completed requests | 8 |
| Elapsed | 8.131916493177414 s |
| Total output tokens | 901 |
| TTFT P50 | 52.559833973646164 ms |
| TTFT P99 | 62.81541660428047 ms |
| ITL P50 | 7.521618157625198 ms |
| ITL P99 | 10.056659579277039 ms |
| Aggregate output throughput | 110.79798971815916 tok/s |
| Mean per-user decode throughput | 116.0061228042662 t/s/u |

Selected full-model teacher-forcing lower bound from datatype sweep:
TTFT 183.02760645747185 ms and decode 92.67547693697753 t/s/u. The vLLM
measurement is a serving workload and is compared only against that lower-bound
evidence.

## Qualitative Review

Source: `readiness_vllm/vllm_qualitative_outputs.json`.

Greedy completions were read and judged mostly coherent and prompt-related.
Issues: the haiku repeats after the first stanza, the thermodynamics answer ends
with an odd boxed value, and the translation prompt does not produce the direct
French translation.

Sampled completions were read and judged not coherent. All six sampled outputs
contain gibberish, code/path fragments, multilingual/token drift, or request
contamination. No sampled output should be treated as production-quality text.
The adapter keeps host fallback disabled; this limitation remains visible in the
artifact instead of being hidden by a host sampling path.

Degenerate-output check:

```bash
python models/common/readiness_check/check_degenerate_output.py \
  --model-dir models/autoports/meta_llama_llama_3_2_1b_instruct \
  --hf-model meta-llama/Llama-3.2-1B-Instruct \
  --missing-artifacts critical \
  --scope vllm \
  --json models/autoports/meta_llama_llama_3_2_1b_instruct/readiness_vllm/degenerate_output_report.json
```

Result: passed with exit code 0. No degenerate output was detected. The highest
sampled adjacent duplication was 0.0144 and the highest sampled trigram loop
fraction was 0.0238, below the critical/advisory thresholds. This does not make
the sampled text good; it rules out the checker-visible stale-feedback
degeneracy pattern.

## Runtime Cleanup

Command:

```bash
pgrep -af 'run_vllm_server|pytest .*vllm-tt-plugin|vllm\.entrypoints|EngineCore|api_server' || true
```

Result after the final successful run: only the audit command matched; no
leftover vLLM, pytest, API server, or EngineCore process was holding devices.

Device availability check:

```bash
LD_LIBRARY_PATH=/localdev/moconnor/tt-metal/build_Release/lib:${LD_LIBRARY_PATH:-} \
  /localdev/moconnor/tt-metal/python_env/bin/python - <<'PY'
import ttnn
print(ttnn.GetNumAvailableDevices())
PY
```

Result: 8.

## Final Artifacts

- `models/autoports/meta_llama_llama_3_2_1b_instruct/readiness_vllm/adapter_trace_input_refresh_contract.json`
- `models/autoports/meta_llama_llama_3_2_1b_instruct/readiness_vllm/degenerate_output_report.json`
- `models/autoports/meta_llama_llama_3_2_1b_instruct/readiness_vllm/sampling_selection.json`
- `models/autoports/meta_llama_llama_3_2_1b_instruct/readiness_vllm/sampling_tests.log`
- `models/autoports/meta_llama_llama_3_2_1b_instruct/readiness_vllm/server.log`
- `models/autoports/meta_llama_llama_3_2_1b_instruct/readiness_vllm/vllm_benchmark.json`
- `models/autoports/meta_llama_llama_3_2_1b_instruct/readiness_vllm/vllm_qualitative_outputs.json`

## Open Limitations

- Batch-1 serving only.
- Host/compat sampling fallback disabled.
- vLLM logprob API disabled for this adapter.
- Sampled qualitative output is poor under `temperature=0.7`, `top_p=0.9`.
- `tt_async_decode_allows_overlap=false` because overlap safety is not proven.
