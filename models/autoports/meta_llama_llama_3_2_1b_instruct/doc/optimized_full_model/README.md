# Optimized Full Model

Optimized-full-model state for `meta-llama/Llama-3.2-1B-Instruct` on a T3K
`1x8` mesh with `FABRIC_1D_RING`. This stage stops before vLLM integration.

## Top-Line Results

| Path | Source | Prompt | Decode steps | TTFT ms | Decode t/s/u | Decode ms/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full-model baseline teacher forcing | `../full_model/README.md` | 184 | 100 | 252.12 | 52.55 | 19.03 |
| Optimized teacher forcing | refreshed AIME24 run | 184 | 100 | 277.31 | 91.89 | 10.88 |
| Full-model baseline token-out | `../full_model/README.md` | 60 | 128 | 173.64 | 56.44 | 17.72 |
| Optimized traced token-out, no readback | `token_out_no_readback_perf.json` | 60 | 128 | 180.00 | 159.97 | 6.25 |

Teacher forcing remains readback-inclusive and resets decode inputs from host for
the AIME24 comparison. The optimized token-out row is the serving-style path:
one captured full-model trace plus split greedy sampling trace, persistent
`tt_out_tok` feedback, device-side position/RoPE advance, changed-only page
tables, nonblocking trace replay, and no per-token host sync/readback.

## Correctness

| Check | Result |
| --- | --- |
| AIME24 chat-template prefill | top1 88/100, top5 100/100, top100 100/100 |
| AIME24 teacher forcing | top1 86/100, top5 100/100, top100 100/100 |
| Autoregressive 128 tokens | HF produced 128, TT produced 128, artifacts saved under `artifacts/autoregressive_default_128/` |
| Degenerate-output audit | Passed: adjacent duplication 0.0, trigram loop fraction 0.0841 for optimized TT completion |
| Top-k/top-p smoke | `top_k=16`, `top_p=0.9`, `temperature=0.8`, 16 tokens, one common sampler trace, force-argmax disabled |
| Runtime fallback audit | Passed guarded no-readback loop: no `from_torch`, `to_torch`, `copy_host_to_device_tensor`, sync, or readback inside the measured loop |
| Watcher | Passed with scoped `TT_METAL_WATCHER_DISABLE_ETH=1`; zero critical watcher issues |

## Implementation

The full model keeps the optimized multichip decoder policy unchanged:

- BFP8 attention weights and KV cache;
- BFP4 MLP weights;
- BF16 activations, residual stream, final norm, and LM-head output;
- BFP8 residual CCL payloads with persistent CCL buffers;
- W2 16-core decode target;
- replicated full-hidden inter-layer residual boundary.

Full-model-only changes in this stage:

- `Llama32FullModel` accepts a `num_layers` profiling override while defaulting
  to all 16 layers.
- LM-head logits remain vocab-sharded over 8 devices and split into 8192 and
  7840 local columns.
- Greedy split sampling pads the local shard from 16032 to 16384 columns before
  `TopKDeviceOperation`, using `-max_float` for padded values.
- Greedy sampling is semantically greedy: local top-32 per shard, gather only
  candidate values/ids, global argmax over candidates, then write the selected
  token to `tt_out_tok`.
- Top-k/top-p requests continue to use `SamplingGenerator` with internal trace
  and the same output-token feedback contract.
- No force-argmax path and no full-vocab all-gather are used for the measured
  greedy token-out path.

## Performance Accounting

Decoder-stack lower bound from the optimized multichip decoder layer latencies:

| Component | Per layer ms | 16-layer ms | t/s/u |
| --- | ---: | ---: | ---: |
| Isolated layer traced host wall | 0.648592 | 10.3775 | 96.36 |
| Isolated layer device window | 0.613593 | 9.8175 | 101.86 |

The measured optimized token-out path is 6.251 ms/token, so there is no slower
than lower-bound-plus-terminal gap to close. The layer-derived number is from an
isolated one-layer trace replay window; the single full-stack captured trace
amortizes per-layer replay/window overhead. This is why the full-stack measured
number is faster than multiplying isolated-layer signpost latency by 16.

Reduced full-path `tt-perf-report` evidence is in `perf/`. The profiler run uses
one real decoder layer plus embeddings, final norm, LM head, device
position/RoPE advance, and split greedy sampling. The full traced 16-layer
no-readback benchmark supplies the top-line token-out rate.

Reduced eager decode summary:

| Bucket | Device time |
| --- | ---: |
| Matmuls, including LM head | 730.78 us, 47.60% |
| TopKDeviceOperation | 154.17 us, 10.04% |
| Sampler candidate all-gather | 63.13 us |
| Sampler argmax/gather/pack | 166.02 us |
| Final norm | 7.89 us |
| LM-head matmuls | 619.60 us |

Sampler work is visible but not dominant; LM-head matmuls are the largest
terminal component. The measured path avoids full-vocab all-gather and
force-argmax.

## Artifacts

- `token_out_no_readback_perf.json`
- `runtime_fallback_audit.json`
- `readiness_perf_summary.json`
- `perf_trace_contract.json`
- `perf_trace_contract_eager.json`
- `perf/ops_perf_results_raw.csv`
- `perf/profile_log_device_raw.csv`
- `perf/tracy_profile_log_host.tracy`
- `perf/prefill_reduced_*`
- `perf/eager_decode_reduced_*`
- `perf/perf_summary.json`
- `perf/perf_provenance.json`
- `artifacts/autoregressive_default_128/`
- `artifacts/trace_evidence/topk_topp_trace_smoke.json`
- `watcher/watcher_clean_eth_disabled_summary.json`

## Commands

Key commands are recorded in `work_log.md`. The final no-readback token-out
benchmark command was:

```bash
MD_OPT_FULL_MODEL_ARTIFACT_DIR=models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_full_model \
MD_OPT_FULL_MODEL_PROMPT_LEN=60 \
MD_OPT_FULL_MODEL_DECODE_STEPS=128 \
pytest --timeout=900 --confcutdir=models/autoports/meta_llama_llama_3_2_1b_instruct/tests -q -s \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tests/test_optimized_full_model.py::test_optimized_full_model_token_out_no_readback_benchmark
```

## Limitations

- High-level `generate()` still reads each token back so it can return Python
  token ids; the optimized no-readback benchmark covers the serving-style
  traced token-out loop.
- Prefill still reads logits on host for readiness and first-token selection.
- A traced reduced profiling run hung before emitting CSV; the retained
  `tt-perf-report` tables use a reduced eager full-path profile, while the
  traced full-model performance number comes from the non-profiler no-readback
  benchmark.
- Watcher evidence uses `TT_METAL_WATCHER_DISABLE_ETH=1`, matching the scoped
  workaround recorded in the optimized multichip decoder stage for ETH watcher
  firmware overflow.
