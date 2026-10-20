# Orchestration v2: follow the DeepSeek-V3 Galaxy decode reference (2026-10-20)

Reference: `models/demos/deepseek_v3` (2D decoder block, `tt/moe.py`, `tt/experts.py`, `tt/mla/mla1d.py`,
`tt/ccl.py`, `tt/model/row_batched_model.py`) and the single-Galaxy engine `models/demos/deepseek_v3_d_p`
(prefill modules for KDA / mHC / MoE / DSA-MLA, GLM-5.3 adapter). User direction: follow this reference.

## What the reference does per MoE layer (decode)
1. Residual hidden sharded over the TP columns (`hidden / mesh_shape[1]` per chip); `DistributedRMSNorm`
   (statistics exchanged over the TP group), reshards before each sublayer.
2. MLA1D: heads TP over the columns, batch per row (DP rows): identical to our attention layout.
3. MoE: all-gather over the TP columns (full hidden per token) -> gate (top-k) -> row-major tokens ->
   `ttnn.all_to_all_dispatch` over ALL devices (expert mapping one-hot [E, D]) -> every device runs its
   `n_routed_experts / num_devices` FULL experts (w1/w3 bfp4, w2 bfp8; batched `ttnn.linear` over the
   local experts on the dispatched rows) -> `ttnn.all_to_all_combine` -> sum over the K expert rows
   (`ttnn.sum(dim=0)` or `deepseek_moe_fast_reduce_nc` on a 1D ring) -> reduce-scatter over the TP
   columns -> residual add. Pure expert parallelism; no intermediate split, no row gather/reduce-scatter.
4. Row-batched model: rows are independent user batches; LM head 1D; MTP2D.

## Mapping to GLM-5.3-Flash on one 8x4 BH Galaxy
| piece | reference | ours today | change |
|---|---|---|---|
| attention (DSA, KDA) | MLA1D: TP heads over cols, DP rows | same (TP4 x DP8), fused blocks as custom ops | keep; fuse the TP all-reduce into o_proj (`matmul_reduce_scatter_async`) later |
| routed experts | EP over all devices + dispatch/combine, full-width experts | hybrid EP4 x expert-TP8 (72 experts x 1/8 slice per chip), 3 collectives, narrow kernels at 38-54 % of peak | **EP32: 9 full experts per chip, `all_to_all_dispatch` / `all_to_all_combine`, the streaming expert op at full width (reads only active experts), sum over K, TP reduce-scatter** |
| expert dtypes | gate/up bfp4, down bfp8 | bfp8 / bfp8 (bfp4 run passed the 45-layer token check) | adopt bfp4 gate/up + bfp8 down under the real-prompt eval gate |
| residual + norms | TP-sharded residual, distributed norm | replicated 4-stream fp32 residual, all-reduce after each sublayer | phase 2c (after the mHC custom ops): shard the packed streams over the columns, distributed mHC statistics; not before the EP32 MoE lands |
| CCL | `deepseek_v3/tt/ccl.py` (async ops, persistent semaphores) | GLMMesh async collectives (TT_CCL) | equivalent; reuse their ring/linear choices per axis |
| prefill | d_p engine modules (KDA, mHC, MoE dispatch, sparse MLA) | SP v1 on the decode layout | Phase 3: the d_p prefill path with the state hand-off (design section 5) |

## Why EP32 (numbers)
- Bytes per chip per step are placement-invariant (balanced): 227 MB at 256 users, ~135 MB active at 32
  users. The 1/8-wide hybrid slices run at 38-54 % of DRAM peak (sweep); full-width experts at 66-74 %, so
  the design's 0.67 ms/layer at 256 users is reachable only at full width.
- Collectives per MoE layer: 3 (gather rows, reduce-scatter rows, all-reduce cols) -> 2 all-to-alls +
  the TP reduce-scatter; measured cost of the all-to-all pair at our shapes: `tests/test_a2a_perf.py`
  (rows 6-7 run queued; adoption criterion: dispatch + combine <= ~150 us at 256 tokens).
- Load imbalance: dispatch counts per device vary with the routing; the streaming kernel's per-expert
  loop absorbs it (no padding to the max); measure the max/mean over real routing.

## Work split
- OPT_EXPERTS: the EP32 MoE path (`tt/moe.py` `TtMoE(ep=32)`): all_gather_cols (no-op on the replicated
  residual today) -> exact router -> RM dispatch -> stream op (pairs_per_bank = 8 at I = 2048) over the
  dispatched rows -> combine -> sum over K -> reduce_scatter_cols; validated on the (2,4) block (36 experts
  per chip) vs the hybrid path, per-layer time at 4 / 32 tokens per device; then the 8x4.
- Coordinator: `tests/test_a2a_perf.py`, the e2e/demo policy switch (`GLM53_MOE_LAYOUT=ep|hybrid`), the
  bfp4 eval gate, phase 2c design (sharded residual) after the mHC custom ops.
