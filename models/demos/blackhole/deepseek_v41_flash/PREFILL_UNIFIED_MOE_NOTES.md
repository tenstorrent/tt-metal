# Prefill routed experts through ttnn.experimental.deepseek_prefill (DSV41_PREFILL_MOE=unified, default OFF)

Files: `tt/prefill_unified_moe.py` (pipeline, weights, mapping), `tt/prefill_layer.py` (`_moe_unified`, used by the column-split layer only), `tt/dsv41_model.py`
(`UNI_MOE`, `UNI_LAYERS`), `tests/test_unified_moe.py` (standalone single-layer test), `tools/build_unified_cache.py`, `tools/devrun_pf.sh`.

## Mapping (mesh 4 rows x 8 columns, x replicated over the columns, every row its own users)
* Dispatch group axis = mesh ROWS (cluster_axis 0, `dispatch_group_size` 4, `seq_len_per_chip` = tokens of the row in the chunk = U*C).
* The 8 COLUMNS are 8 independent dispatch groups of 48 experts each (12 per chip): expert e -> group (column) e // 48, chip (row) (e % 48) // 12, local slot e % 12
  (`ExpertMapping`, column-major, models/demos/deepseek_v3_d_p). This is the layout the research report suggested; the standalone test checks it against the moe_compute path
  and a torch reference.
* Every column routes the same tokens; dispatch / combine only move tokens between the 4 rows of a column, `post_combine_reduce` keeps the experts of the column's own group,
  the 8 partial sums are added by a reduce-scatter over the columns (see "Findings" for why it is over the hidden dim + all_to_all, not over tokens).

## Layer flow (column-split layer, per chunk and layer; N = U*C tokens per row, n = N/8 own tokens per column)
1. own normed hidden rows hh_own [1,1,n,D] (own 32-token chunks of all groups, concatenated)
2. router on the own rows only (per 32-row slice, as before), routing (expert ids + weights) packed in fp32 row-major pages of 384 values and all-gathered over the columns
3. all_gather of hh_own over the columns -> [1,1,N,D] in row order [column][own chunk][32] (dispatch does not care about the order)
4. masked_bincount + offset_cumsum -> dispatch (RM bf16) -> ONE `unified_routed_expert_moe` (ClampedSiluGlu, limit 10, 12 local experts, bf8 weights) -> combine -> post_combine_reduce
5. ttnn.reduce_scatter over the hidden dim (columns) + all_to_all_async_generic hidden-shard -> own tokens -> [1,1,n,D], sliced into the own 32-token chunks for the mHC expand.
Shared expert, mHC, attention, Engram are untouched.

## Numerics
The op is fixed: LoFi, bf8 activations in, bf8 intermediates and bf8 output (compute_kernel_config fidelity / fp32 acc are ignored by the op, measured: identical results and time).
It applies the swiglu clamp (|up| <= 10, gate <= 10) that the moe_compute path does not (moe_compute has no clamp).

## Findings / pitfalls (all reproduced in tests/test_unified_moe.py with DSV41_UM_DEBUG=1)
* reduce_scatter_minimal_async over the TOKEN dim of the big partial-sum tensor (also in 256- and 512-row pieces) corrupts some 32-row blocks (nondeterministic positions).
  The generic `ttnn.reduce_scatter(dim=hidden)` + `all_to_all_async_generic` is exact at N = 256 / 512 / 2048 / 4096 rows (N = 8192 only ran in the 6-layer traced test, not compared).
* all_gather of tensors that are <= 1 tile wide (routing [n, 12], tile or row-major pages of 12-48 B) drops blocks of rows; gathering 384-value row-major pages is exact.
* DRAM: the unified weights are a second copy of the bf8 expert weights (497 MB / chip / layer, 19.9 GB / chip for 40 layers); the moe_compute decode weights are the same size, so
  decode + unified prefill for all 40 layers does NOT fit a 32 GB chip next to KV and the rest. DSV41_UNI_NODECODE=1 skips the moe_compute weights (prefill-only runs);
  DSV41_UNI_LAYERS=2-9 limits the unified path to a layer subset (decode intact).

## Measurements (BH Galaxy 4x8, bf8 experts, links=2, own-row routing)
Single layer, standalone test (layer 3, random normal hidden, tokens per mesh row N, MoE section of one layer = router + gathers + experts + reduce):
| N per row | moe_compute path (T=256 per call) | unified | PCC vs torch clamp (first 16 tokens) |
| 256 | 2.47 ms | 2.42 ms | 0.99976 (baseline 0.99998) |
| 512 | 4.88 ms | 2.71 ms | 0.99976 |
| 2048 | 23.1 ms | 5.8 ms (links=1) | 0.99976 |
| 4096 | 46.3 ms | 7.7 ms (links=1: 10.0) | 0.99976, 0 bad 32-row blocks vs the baseline output |
| 8192 | -- | 19.2 ms (links=1 only measured) | |
Stage times at N=4096, links=2: bincount+cumsum 0.35, dispatch 1.5, experts 1.96, combine 1.07, post_combine_reduce 0.86, router (own rows) 0.7, gathers + reduce-scatter + all_to_all ~2 ms.

Whole model, 40 layers, prefill only (DSV41_PREFILL_MOE=unified DSV41_UNI_NODECODE=1 DSV41_PREFILL_ONLY=1), TTFT of the demo (second call) / total tok/s:
| scenario | baseline (grid) | unified |
| isl4k_b16 (ISL 3720) | 20.17 s, 2951 tok/s | 11.99 s, 4965 tok/s |
| isl8k_b4 (ISL 7443) | 10.99 s, 2708 tok/s | 6.55 s, 4547 tok/s |
| isl32k_b8 (ISL 30059) | 77.2 s, 3114 tok/s | 44.8 s, 5364 tok/s |
The replay loop alone: 18.77 -> 10.61 s, 10.88 -> 5.49 s, 74.2 -> 43.4 s.
Chunk size (6 layers, U=4): replay per row token is flat from 512 to 2048 tokens per user (0.113-0.115 ms for unified, 0.205-0.210 ms for the moe_compute path): a bigger chunk does not help once the experts see >= 256 rows each.
