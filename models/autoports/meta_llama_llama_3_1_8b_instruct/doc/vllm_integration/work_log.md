# vLLM Integration Work Log

Date: 2026-06-15

Model: `meta-llama/Llama-3.1-8B-Instruct`

Autoport:
`models/autoports/meta_llama_llama_3_1_8b_instruct`

## Code Changes

- Added `tt/generator_vllm.py` with
  `Llama31_8B_InstructForCausalLM`.
- Registered the adapter in
  `/localdev/moconnor/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`
  via `TT_LLAMA_TEXT_VER=autoport_llama31_8b_instruct` and
  `TT_LLAMA_TEXT_VER=meta_llama_llama_3_1_8b_instruct`.
- Kept vLLM serving strict on-device sampling. The adapter rejects missing
  sampling params and non-token host output processing instead of falling back
  to host argmax/full-logits sampling.
- Added platform validation and pytest awareness for `sample_on_device_mode=all`
  so host-compat sampling requests are rejected/skipped before they can reach a
  strict device-only adapter.
- Gated TT scheduler overlap on
  `model_capabilities["tt_async_decode_allows_overlap"]`, separate from
  `supports_async_decode`.
- Fixed request-boundary decode buffer lifetime by releasing decode persistent
  CCL/gather buffers before the next prefill.
- Fixed vLLM prefill cleanup so transient sampled-token tensors are deallocated
  but sampler-owned persistent logprob output tensors are not.
- Updated the shared sampling trace reset to clear Python references to trace
  input/output/kwargs before releasing TT trace IDs.
- Fixed TT sampling temporary tensor cleanup in the top-k/top-p path.
- Hardened readiness prompt token handling for tokenizers returning
  `BatchEncoding` or dict-like `input_ids`.

## Static Validation

```bash
python_env/bin/python -m py_compile \
  models/common/readiness_check/run_vllm_server.py \
  models/common/readiness_check/generate.py \
  models/common/sampling/generator.py \
  models/common/sampling/tt_sampling.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/generator.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/generator_vllm.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/model.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/multichip_decoder.py
```

Result: pass.

```bash
python_env/bin/python -m py_compile \
  /localdev/moconnor/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py \
  /localdev/moconnor/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py \
  /localdev/moconnor/vllm/plugins/vllm-tt-plugin/tests/tt/conftest.py \
  /localdev/moconnor/vllm/plugins/vllm-tt-plugin/tests/tt/test_host_only_params.py \
  /localdev/moconnor/vllm/plugins/vllm-tt-plugin/tests/tt/test_logprobs.py \
  /localdev/moconnor/vllm/plugins/vllm-tt-plugin/tests/tt/test_seeding_and_variety.py \
  /localdev/moconnor/vllm/plugins/vllm-tt-plugin/tests/tt/test_tt_penalties.py
```

Result: pass.

`git diff --check` passed for the touched tt-metal and vLLM plugin files.

## Precision Policy

Serving uses
`models/autoports/meta_llama_llama_3_1_8b_instruct/doc/datatype_sweep/selected_precision_config.json`.

Selected policy:

- Attention and MLP weights: `bfloat4_b`
- LM head: `bfloat8_b`
- Embedding and norms: `bfloat16`
- Activation, residual, CCL, KV cache, MLP multiply: `bfloat8_b`
- Attention and LM head fidelity: `HiFi2`
- MLP fidelity: `LoFi`
- Norm fidelity: `HiFi4`
- Layer exceptions: none

## Serving Runs

Successful smoke, qualitative, and initial benchmark run:

```bash
env TT_LLAMA_TEXT_VER=autoport_llama31_8b_instruct timeout 7200s \
  python_env/bin/python -u -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --mesh-device T3K \
  --max-num-seqs 1 \
  --max-model-len 1024 \
  --sampling-profile smoke \
  --tt-config '{"trace_region_size":216580672,"fabric_config":"FABRIC_1D_RING","trace_mode":"decode_only"}' \
  --port 8001
```

Result: pass. Artifacts archived at
`artifacts/success_smoke_sampling_qual_bench_20260615/`.

Successful final full sampling run:

```bash
env TT_LLAMA_TEXT_VER=autoport_llama31_8b_instruct timeout 7200s \
  python_env/bin/python -u -m models.common.readiness_check.run_vllm_server \
  --stages serve,sampling \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --mesh-device T3K \
  --max-num-seqs 1 \
  --max-model-len 1024 \
  --sampling-profile full \
  --tt-config '{"trace_region_size":216580672,"fabric_config":"FABRIC_1D_RING","trace_mode":"decode_only"}' \
  --port 8001
```

Result: pass.

```text
41 passed, 22 skipped, 9 xfailed, 16 warnings in 119.69s
```

Artifacts:

- `readiness_vllm/sampling_tests.log`
- `artifacts/success_full_sampling_20260615/server.log`
- `artifacts/success_full_sampling_20260615/sampling_tests.log`

Successful batch-1 benchmark run:

```bash
env TT_LLAMA_TEXT_VER=autoport_llama31_8b_instruct timeout 7200s \
  python_env/bin/python -u -m models.common.readiness_check.run_vllm_server \
  --stages serve,benchmark \
  --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --mesh-device T3K \
  --max-num-seqs 1 \
  --max-model-len 1024 \
  --tt-config '{"trace_region_size":216580672,"fabric_config":"FABRIC_1D_RING","trace_mode":"decode_only"}' \
  --benchmark-prompt-len 128 \
  --benchmark-output-len 128 \
  --benchmark-num-requests 32 \
  --benchmark-concurrency 1 \
  --port 8001
```

Result: pass. Metrics from `readiness_vllm/vllm_benchmark.json`:

- TTFT P50/P99: `81.10 ms` / `1047.01 ms`
- ITL P50/P99: `15.77 ms` / `25.09 ms`
- Aggregate output throughput: `51.84 tok/s`
- Mean per-user decode: `53.90 t/s/u`
- Completed requests/tokens: `32` / `4090`
- Elapsed: `78.90 s`

Artifacts archived at `artifacts/success_batch1_benchmark_20260615/`.

## Sampling Exceptions

The final full sampling profile intentionally skips request features that
require host compatibility sampling or full/top-k logprob readback under
`sample_on_device_mode=all`: `min_p`, bad words, logit bias, allowed token IDs,
minimum tokens, top-k logprobs, and all-vocab logprobs.

The final profile xfails no-seed variety and top-k variety checks. The vLLM
integration skill classifies these reproducibility/variety-only checks as out
of scope when correctness failures are absent. Seeded reproducibility, request
isolation, sampled-token logprobs, top1 greedy, and batch-1 penalty smoke checks
passed through the adapter.

## Qualitative Review

`readiness_vllm/vllm_qualitative_outputs.json` was read manually.

Verdict: coherent and generally on topic. Haiku, learning explanation,
translation, thermodynamics, and Fibonacci outputs match the prompts. The story
continuation stays on topic. Minor issues: duplicated leading tokens (`a a`,
`The The`) and one sampled haiku answer drifts into meta-commentary. No
gibberish, no wrong-language drift beyond the requested French translation, and
no cross-request contamination were observed.

Degeneracy check:

```bash
python_env/bin/python models/common/readiness_check/check_degenerate_output.py \
  --hf-model meta-llama/Llama-3.1-8B-Instruct \
  --missing-artifacts critical \
  --scope vllm
```

Result saved to `readiness_vllm/check_degenerate_output.log`:

```text
No degenerate output detected.
```

## Trace And Cache Ownership

The vLLM adapter uses the full-model canonical split-sampling path:
`_decode_trace_sample(..., readback=False)` runs traced decode and calls
`self.sampling.sample(..., tt_out_tok=self._decode_trace.tokens)`, so sampled
tokens feed the persistent decode token input on device.

Full-model trace-feedback artifacts show:

- `all_trace_feedback_assertions_passed=true`
- `unchanged_page_table_not_recopied=true`
- `changed_page_table_copied_once=true`
- `changed_page_table_refresh_marked_changed_only=true`
- `greedy_uses_split_sampling=true`

The vLLM full sampling run exercises that delegated path with decode trace
enabled. Request isolation, seeded reproducibility, sampled-token logprobs, and
top1-greedy all passed through the adapter.

vLLM serving owns attention KV cache allocation. The adapter allocates the cache
from vLLM's requested shape using the selected `kv_cache_dtype`, stores it as
`self._vllm_kv_cache`, and passes the same cache through prefill/decode.
Standalone generator mode keeps the model-owned cache.

## Async Decode

Adapter capability flags:

```python
supports_async_decode = True
supports_sample_on_device = True
supports_prefix_caching = False
tt_async_decode_allows_overlap = False
```

`supports_async_decode=True` means submit/read/host formatting is split and
decode trace is enabled. It does not enable scheduler overlap.

`tt_async_decode_allows_overlap=False` because the adapter still refreshes
token/current-position/page-table tensors from vLLM host scheduler state at
trace/layout changes. No async-scheduling overlap proof was run, so the plugin
keeps overlap disabled.

## Performance Comparison

The datatype-selected full-model teacher-forcing lower bound is `49.49 t/s/u`.
The optimized full-model no-readback token-out path measured `70.55 t/s/u`.

The final vLLM batch-1 serving benchmark measured `53.90 t/s/u`. This is above
the teacher-forcing lower bound and below the direct no-readback token-out path,
which is expected because vLLM adds serving orchestration, OpenAI API handling,
streaming response work, and token/logprob readback. Source audit found no
adapter-side host argmax path, full-logits readback, generic greedy fallback, or
Python readback/writeback token feedback loop in the measured vLLM path.

## Runtime Cleanup

All successful `run_vllm_server` invocations terminated cleanly.

Final process audit:

```bash
pgrep -af "vllm.entrypoints|EngineCore|TTEngineCore|run_vllm_server" || true
```

Result: only the audit command itself matched; no leftover server/EngineCore
process held devices.

Final device audit:

```bash
tt-smi -ls --local
```

Result: all 8 devices listed.

## Limitations

- Current adapter supports `max_num_seqs=1`.
- Host-compatibility sampling features are rejected/skipped in strict
  `sample_on_device_mode=all`.
- Top-k/all-vocab logprobs are unsupported for this model without a device top-k
  logprob path.
- Scheduler overlap is disabled (`tt_async_decode_allows_overlap=False`) until
  focused async overlap tests prove token/current-position/page-table freshness
  for step N+1.
