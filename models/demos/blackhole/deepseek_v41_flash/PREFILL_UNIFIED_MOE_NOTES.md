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
  The generic `ttnn.reduce_scatter(dim=hidden)` + `all_to_all_async_generic` is exact at N = 2048 / 4096 / 8192 rows.
* all_gather of tensors that are <= 1 tile wide (routing [n, 12], tile or row-major pages of 12-48 B) drops blocks of rows; gathering 384-value row-major pages is exact.
* DRAM: the unified weights are a second copy of the bf8 expert weights (497 MB / chip / layer, 19.9 GB / chip for 40 layers); the moe_compute decode weights are the same size, so
  decode + unified prefill for all 40 layers does NOT fit a 32 GB chip next to KV and the rest. DSV41_UNI_NODECODE=1 skips the moe_compute weights (prefill-only runs);
  DSV41_UNI_LAYERS=2-9 limits the unified path to a layer subset (decode intact).
