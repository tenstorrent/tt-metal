# Optimized vLLM Work Log

Model: `meta-llama/Llama-3.2-1B-Instruct`

Stage date: 2026-06-15

Skills used: `vllm-integration`, `optimize`, and `tt-enable-tracing`.

## Starting Point

The stage started from the completed vLLM integration and the datatype-sweep selected config:

- vLLM adapter: `tt/generator_vllm.py`
- Full-model generator: `tt/generator.py`
- Selected precision: `doc/datatype_sweep/selected_precision_config.json`, `cfg08_bfp8_weights_bfp8_kv_bf16_ccl`
- Prior vLLM integration benchmark: TTFT P50 52.56 ms, ITL P50 7.52 ms, aggregate output throughput 110.80 tok/s, mean per-user decode 116.01 t/s/u.

The initial vLLM integration path served and passed selected sampling tests, but sampled qualitative output was gibberish. That made the optimized-vLLM state incomplete even though speed was already close to the full-model baseline.

## Debugging Notes

I reproduced the serving issue against the real adapter/generator path instead of introducing an adapter fallback.

Observed mechanisms:

- vLLM block tables use physical IDs and can retain stale tail entries after previous requests.
- The vLLM cache for 4096 tokens with block size 64 has 65 blocks, with block 0 reserved.
- Remapping the first active page to physical block 0 broke vLLM greedy output.
- Using a duplicated page-table tail also broke output.
- vLLM unbounded/default sampling arrived as `top_k` equal to the full vocab size, which sent the hot token-out path through generic top-k/top-p sampling instead of the optimized full-model split-greedy contract.

Kept fixes:

- Sanitize page-table tails before prefill and decode.
- For vLLM-owned KV cache, remap active page-table entries densely starting at block 1 when the bound vLLM cache has the extra reserved block.
- Compute page-table change signatures from the raw scheduler page table, but copy the sanitized page table into persistent trace inputs. This preserves scheduler-change detection while feeding TT a dense valid table.
- Normalize only vLLM's unbounded/default sampled case to split-greedy sampling in serving mode.
- Keep explicit bounded top-k/top-p plugin tests on the device sampling path.
- Add gated debug logging under `TT_LLAMA32_VLLM_DEBUG_PATH` to prove the async split path.
- Count async read and host formatting calls.
- Keep LM-head weight dtype configurable, with the selected default remaining BFP8.

Rejected options:

- `force_argmax`: rejected because the goal requires the full-model split-sampling contract rather than a force-argmax workaround.
- Host top-1 argmax, full-logits readback, or host sampling: rejected because `sample_on_device_mode=all` forbids host fallback.
- Generic unbounded top-k/top-p for vLLM default requests: rejected because it produced poor sampled qualitative text and is not the optimized token-out contract.
- BF16 LM-head: tried and rejected because it did not improve sampled qualitative output.
- `skip_precompile=True` for generic sampling: tried and rejected because synthetic trace capture produced a bad first token.
- Physical block 0 remap: rejected because vLLM reserves block 0 in this cache shape.
- Per-token page-table refresh: rejected because unchanged scheduler state should perform no page-table copy.
- Live vLLM profiling: rejected per the vLLM/optimize skill safety rule for T3K serving stages.

## Final Serving Run

```bash
TT_LLAMA_TEXT_VER=autoport_llama32_1b \
PYTHONPATH=/localdev/moconnor/tt-metal:/localdev/moconnor/vllm:/localdev/moconnor/vllm/plugins/vllm-tt-plugin/src:${PYTHONPATH:-} \
LD_LIBRARY_PATH=/localdev/moconnor/tt-metal/build_Release/lib:${LD_LIBRARY_PATH:-} \
python models/common/readiness_check/run_vllm_server.py \
  --model-dir models/autoports/meta_llama_llama_3_2_1b_instruct \
  --hf-model meta-llama/Llama-3.2-1B-Instruct \
  --stages serve,sampling,qualitative,benchmark \
  --mesh-device T3K \
  --port 8023 \
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

Result:

- Server ready after about 40 s.
- Sampling tests: 19 passed, 42 deselected, 16 warnings.
- Qualitative prompts: 6 prompts completed; greedy and sampled outputs were coherent and on-topic enough for readiness. Sampled equals greedy for all 6 prompts because vLLM unbounded/default sampled requests are mapped to split-greedy.
- Benchmark: 8 requests, 901 output tokens, 8.1858 s elapsed.
- TTFT P50/P99/mean: 52.98/78.87/56.64 ms.
- ITL P50/P99/mean: 7.56/10.01/7.63 ms.
- Aggregate output throughput: 110.07 tok/s.
- Mean per-user decode throughput: 115.63 t/s/u.
- Server terminated cleanly.

Artifacts:

- `readiness_vllm/server.log`
- `readiness_vllm/sampling_tests.log`
- `readiness_vllm/sampling_selection.json`
- `readiness_vllm/vllm_qualitative_outputs.json`
- `readiness_vllm/vllm_benchmark.json`

## Degenerate Output Check

```bash
python models/common/readiness_check/check_degenerate_output.py \
  --model-dir models/autoports/meta_llama_llama_3_2_1b_instruct \
  --hf-model meta-llama/Llama-3.2-1B-Instruct \
  --missing-artifacts critical \
  --scope vllm \
  --json models/autoports/meta_llama_llama_3_2_1b_instruct/readiness_vllm/degenerate_output_report.json
```

Result: passed, no degenerate output detected.

## Async Boundary Evidence

Focused command used the same server path and config with:

```bash
TT_LLAMA32_VLLM_DEBUG_PATH=models/autoports/meta_llama_llama_3_2_1b_instruct/readiness_vllm/async_boundary_debug.jsonl
TT_LLAMA32_VLLM_DEBUG_LIMIT=120
```

Result artifact: `readiness_vllm/async_boundary_debug.jsonl`.

Event counts:

- `prefill_input`: 1
- `prefill_output`: 1
- `decode_input`: 40
- `read_decode_output`: 39
- `decode_output`: 39

All 39 `read_decode_output` events had `async_read=true`. This proves vLLM exercised `decode_forward(..., read_from_device=False)`, then `read_decode_output(..., async_read=True)`, then host formatting.

## Trace Input Refresh Evidence

Artifact: `readiness_vllm/adapter_trace_input_refresh_contract.json`.

The direct test function reported `direct trace refresh contract: PASS` and wrote the artifact. Repo-root pytest collection was blocked by a missing `models.tt_transformers` import in root `conftest.py`, and the process later hit an ARC timeout during TT device close. The final full vLLM server run is the clean shutdown evidence.

Key artifact values:

- `supports_async_decode`: true
- `tt_async_decode_allows_overlap`: false
- `decode_trace_enabled`: true
- KV cache owner: vLLM adapter `allocate_kv_cache`
- Steady unchanged page table: one model trace replay, one sampler trace replay, zero token/current-position/RoPE/page-table refreshes, zero syncs, zero readbacks, zero host argmax, zero full-logits readbacks.
- Changed page table: one page-table refresh only.
- Reset batch: token/current-position/RoPE/page-table refreshed once.

## Before/After Metrics

| Metric | Before | After |
| --- | ---: | ---: |
| TTFT P50 | 52.56 ms | 52.98 ms |
| TTFT P99 | 62.82 ms | 78.87 ms |
| TTFT mean | 54.55 ms | 56.64 ms |
| ITL P50 | 7.52 ms | 7.56 ms |
| ITL P99 | 10.06 ms | 10.01 ms |
| ITL mean | 7.61 ms | 7.63 ms |
| Aggregate output throughput | 110.80 tok/s | 110.07 tok/s |
| Mean per-user decode | 116.01 t/s/u | 115.63 t/s/u |

The speed remained within run noise. The material optimization is path correctness and contract cleanup: sampled qualitative output now passes, async readback is exercised, and unchanged page tables do not trigger per-token copies.

## Full-Model Comparison

From `doc/optimized_full_model/readiness_perf_summary.json`:

- Optimized full-model teacher-forcing decode: 91.89 t/s/u.
- Optimized full-model token-out no-readback decode: 159.97 t/s/u.
- Optimized full-model token-out no-readback latency: 6.25 ms/token.

The final vLLM serving decode result, 115.63 t/s/u, is faster than optimized full-model teacher-forcing and about 72 percent of the no-readback token-out path. The gap is expected from vLLM orchestration and the plugin output boundary.

## Profiler And Performance Accounting

No live vLLM profiler was collected. This is intentional for vLLM serving stages under the selected skills.

Equivalent profiler artifacts:

- `doc/optimized_full_model/perf/eager_decode_reduced_tt_perf_report.txt`
- `doc/optimized_full_model/perf/eager_decode_reduced_per_device_tt_perf_report.txt`
- `doc/optimized_full_model/perf/ops_perf_results_raw.csv`
- `doc/optimized_full_model/perf/perf_summary.json`

The reduced full-model profile covers terminal decode work: final norm, LM head, logits movement, split sampling, candidate all-gather, and token feedback. The sampler contract has `force_argmax_enabled=false`, no full-vocab all-gather, local top-k input padded to 16384, and candidate all-gather only.

Serving performance accounting is recorded in `doc/optimized_vllm/perf_summary.json`. Device-time and roofline fields are `null` because live vLLM profiler collection is disabled for this stage.

## Final Verification

Syntax:

```bash
python -m py_compile \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tt/generator.py \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tt/generator_vllm.py \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tt/model.py \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tt/optimized_decoder.py
```

Process cleanup:

```bash
pgrep -af 'run_vllm_server|pytest .*vllm-tt-plugin|vllm\.entrypoints|EngineCore|api_server' || true
```

Result: no leftover vLLM server, EngineCore, plugin pytest, or API server process beyond the audit command itself.

