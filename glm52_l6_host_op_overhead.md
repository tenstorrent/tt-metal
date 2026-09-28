# GLM-5.2 layer 6 host op overhead

Measured a complete untraced layer 6 forward for one 5,120-token prefill chunk at a 51,200-token KV depth on an 8×4 Blackhole mesh. Layer 6 runs sparse indexer, sparse attention, and MoE. The input hidden state and KV/indexer prefix came from the GLM-5.2 golden trace; KV and indexer caches were TP-sharded. The run used `TT_METAL_SHM_TRACKING_DISABLED=1`, `LOGURU_LEVEL=ERROR`, and `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`.

The figures below come from the fastest of 10 measured chunks after 10 warmups. Each row sums **exclusive host time inside TTNN API calls** for that chunk. These are host call durations, not device kernel durations or necessarily latency recoverable by tracing. The 114 calls include 28 `ttnn.deallocate` calls and are not a count of device programs.

| TTNN op | Calls | Host sum (ms) | Avg/call (µs) |
|---|---:|---:|---:|
| `ttnn.experimental.reduce_scatter_minimal_async` | 5 | 1.590 | 317.9 |
| `ttnn.experimental.high_bw_all_gather` | 7 | 1.358 | 194.0 |
| `ttnn.experimental.all_reduce_async` | 1 | 0.610 | 609.8 |
| `ttnn.linear` | 9 | 0.586 | 65.1 |
| `ttnn.experimental.all_to_all_async_generic` | 2 | 0.566 | 282.9 |
| `ttnn.experimental.deepseek_prefill.rotary_embedding_indexed` | 4 | 0.541 | 135.3 |
| `ttnn.experimental.deepseek_prefill.combine` | 1 | 0.487 | 487.2 |
| `ttnn.experimental.deepseek_prefill.dispatch` | 1 | 0.454 | 454.3 |
| `ttnn.reduce_scatter` | 1 | 0.425 | 424.9 |
| `ttnn.matmul` | 6 | 0.392 | 65.4 |
| `ttnn.add` | 3 | 0.378 | 126.1 |
| `ttnn.experimental.deepseek_prefill.offset_cumsum` | 1 | 0.372 | 372.0 |
| `ttnn.experimental.ring_indexer_score_dsa` | 1 | 0.344 | 343.8 |
| `ttnn.experimental.deepseek_prefill.update_padded_kv_cache` | 2 | 0.344 | 171.9 |
| `ttnn.experimental.all_gather_async` | 1 | 0.300 | 300.3 |
| `ttnn.to_layout` | 5 | 0.224 | 44.9 |
| `ttnn.mesh_partition` | 1 | 0.209 | 208.6 |
| `ttnn.deallocate` | 28 | 0.173 | 6.2 |
| `ttnn.multiply_` | 1 | 0.153 | 153.3 |
| `ttnn.unsqueeze` | 5 | 0.147 | 29.5 |
| `ttnn.rms_norm_post_all_gather` | 2 | 0.145 | 72.5 |
| `ttnn.rms_norm_pre_all_gather` | 2 | 0.134 | 67.1 |
| `ttnn.experimental.deepseek_prefill.unified_routed_expert_moe` | 1 | 0.113 | 112.7 |
| `ttnn.rms_norm` | 2 | 0.112 | 55.9 |
| `ttnn.experimental.deepseek_prefill.moe_fused_swiglu` | 1 | 0.102 | 101.8 |
| `ttnn.to_memory_config` | 2 | 0.097 | 48.4 |
| `ttnn.experimental.topk_large_indices` | 1 | 0.090 | 90.5 |
| `ttnn.experimental.nlp_create_qkv_heads` | 1 | 0.083 | 82.7 |
| `ttnn.experimental.fast_reduce_nc_split` | 1 | 0.073 | 72.5 |
| `ttnn.experimental.nlp_create_q_heads_split` | 1 | 0.071 | 70.9 |
| `ttnn.experimental.deepseek_prefill.moe_grouped_topk` | 1 | 0.070 | 69.8 |
| `ttnn.experimental.deepseek_prefill.masked_bincount` | 1 | 0.064 | 64.1 |
| `ttnn.concat` | 1 | 0.063 | 63.1 |
| `ttnn.squeeze` | 4 | 0.062 | 15.6 |
| `ttnn.layer_norm` | 1 | 0.061 | 60.6 |
| `ttnn.experimental.deepseek_prefill.post_combine_reduce` | 1 | 0.055 | 55.1 |
| `ttnn.typecast` | 1 | 0.052 | 51.9 |
| `ttnn.transformer.sparse_sdpa` | 1 | 0.051 | 51.3 |
| `ttnn.experimental.nlp_concat_heads` | 1 | 0.047 | 47.3 |
| `ttnn.reshape` | 2 | 0.026 | 13.2 |
| `ttnn.view` | 1 | 0.014 | 13.9 |
| **Total** | **114** | **11.239** | **98.6** |

The same chunk took **14.948 ms** end to end: 11.239 ms inside TTNN calls, 1.379 ms inside `forward()` outside those calls, 2.242 ms waiting at the final device synchronization, and 0.089 ms for input deallocation and other chunk work. The median of the 10 measured chunks was 14.975 ms.

Measurement test: `models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_glm52_prefill_single_moe_layer_perf` with `TT_PREFILL_HOST_OPS=1`, selecting `notrace`.

## Explicit-link lookup optimization

The model passes an explicit link count to `all_reduce_async`, `combine`, and `reduce_scatter`. Their wrappers used `optional::value_or(get_num_links(...))`, which evaluates link discovery even when the optional is set. The wrappers now call `get_num_links` only when no link count is supplied, following the dispatch fix from [PR 57133](https://github.com/tenstorrent/tt-metal/pull/57133).

Same GLM-5.2 L6 setup and CI-like environment as above: 10 warmups, 10 measured untraced chunks, selecting the fastest measured chunk in each run. Host figures are from that chunk.

| Op | Before (µs/call) | After (µs/call) | Change (µs) |
|---|---:|---:|---:|
| `ttnn.experimental.all_reduce_async` | 609.8 | 556.4 | -53.5 |
| `ttnn.experimental.deepseek_prefill.combine` | 487.2 | 449.8 | -37.4 |
| `ttnn.reduce_scatter` | 424.9 | 387.4 | -37.5 |
| `ttnn.experimental.deepseek_prefill.dispatch` | 454.3 | 458.8 | +4.5 |
| `ttnn.experimental.deepseek_prefill.offset_cumsum` | 372.0 | 395.3 | +23.3 |
| All TTNN host calls | 11.239 ms | 10.965 ms | -0.274 ms |
| Full layer, fastest chunk | 14.948 ms | 14.787 ms | -0.161 ms |
| Full layer, median chunk | 14.975 ms | 14.865 ms | -0.110 ms |

The unchanged calls show run-to-run variation; their differences are not attributed to this patch. An experiment that skipped `to_layout(ROW_MAJOR)` after offset-cumsum's internal all-gather did not improve offset-cumsum host time (395.3 → 392.6 µs) or full-layer latency, so it was reverted. Dispatch's explicit-link lookup was already optimized in PR 57133.

### Follow-up: persistent reduce-scatter output

The reduce-scatter device operation used `optional_output_tensor.value_or(create_device_tensor(...))`. Because `value_or` evaluates its fallback, passing a persistent output still allocated a temporary tensor. The C++ path now creates an output only when none was supplied.

I also tried caching the GLM-5.2 MoE reduce-scatter output in the model. The same 10-warmup/10-measured L6 test passed, but the call was 390.8 µs versus 387.4 µs without reuse, and median layer time was 15.172 ms versus 14.865 ms. Both changes were reverted after confirming that the intended next work is runtime-argument overhead. These runs do not show a host benefit from persistent output for this call.

### Follow-up: all-gather runtime-argument lookups

The gate's composite all-reduce uses the default async all-gather. An experiment hoisted runtime-argument table lookups and buffer/semaphore address reads out of its worker loop without changing kernel arguments. On the same 10-warmup/10-measured L6 test, the outer all-reduce call was 572.8 µs versus 556.4 µs before the experiment; layer medians were 14.897 ms versus 14.865 ms. The change was reverted. This suggests those lookups are not the dominant cost of this call.
