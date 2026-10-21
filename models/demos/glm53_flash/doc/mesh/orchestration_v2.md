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

## EP32 bring-up on the (2,4) block (2026-10-21, branch arg/glm53_ep32)

Block 11,10,15,14,7,6,3,2 (grid rows 2-3), real layer-4 weights and routing, FABRIC_1D (the EP layout's default;
`GLM53_MOE_LAYOUT=ep` selects the layout for every mesh entry point, `GLM53_FABRIC` overrides the fabric).

### What runs
`tt/moe.py::TtEPExperts` on a mesh = three stages per MoE layer:
1. `dispatch_rows`: the bf16 routing weights ride as a K-wide tail of every token row (`[x | w]`, one row-major
   concat), the uint32 ids from `ttnn.topk` typecast to uint16 rows, ONE `ttnn.all_to_all_dispatch` along the rows
   (6 device ops + the dispatch; no second dispatch of the weights, no dense routing matrix).
2. `run_experts`: `custom_ops/expert_stream` `ExpertDispatch(ep=True, ep_rm=True)` (2 generic ops) on the dispatched
   rows: the kernels read the ids straight from the dispatch metadata (exact uint16; the bf16 index tiles of the
   single-chip EP path cannot hold ids 257..287) and the weights from the row tails; short batches are tile-padded
   with the valid row count as a runtime arg.
3. `combine_rows`: `all_to_all_combine(local_reduce=True)` into a zeroed `[K, T, 1, H]` slot tensor (the op writes
   the slot of the device's first local expert in the token's list and leaves the rest untouched), tilize, sum over
   K (4 device ops + the combine). The result is the column's partial; `TtMoE` adds the shared expert and all-reduces
   over the columns as before.

### Correctness (tests/test_mesh_moe.py, GLM53_MOE_LAYOUT=ep, bar 0.995)
| users / row | EP worst PCC vs HF | hybrid (stream) worst PCC vs HF | EP build + upload | EP first forward (compiles) | warm eager forward |
|---|---|---|---|---|---|
| 2 | 0.999890 | 0.999939 | 117 s | 33.8 s | 0.01 s |
| 32 | 0.999893 | 0.999922 | 99 s | 4.2 s | 0.01 s |

Replays are bit-identical (iteration 0 == iteration 1 on every chip). Logs: `_logs/agents/ep32/mesh_moe_ep_1d.log`
(EP, 2026-10-21), `_logs/mesh/test_mesh_moe_stream2_rows67.log` (hybrid stream, rows 6-7, 2026-10-20).

### Decode step cost per MoE layer (tests/test_a2a_perf.py::test_moe_decode_components, traced replay, us)
Block 11,10,15,14,7,6,3,2, FABRIC_1D, num_links 1, real layer-4 routing (`moe_components_1d_r2.log`, `_r3.log`);
"ops" = device ops besides the collective / the two expert kernels.

| stage | 4 users/row | 32 users/row |
|---|---|---|
| EP router (`ep_rows`: no dense scatter) | 150 | 137 |
| EP dispatch stage: 6 ops + `all_to_all_dispatch` | 33 | 68 |
| EP expert compute (2 kernels, full-width experts) | 1363 | 3019 |
| EP combine stage: `all_to_all_combine` + zero clone + 2 layouts + K-sum | 208 (ttnn.sum) / 201 (fast_reduce_nc) | 575 (ttnn.sum) / 575 (fast_reduce_nc) |
| **EP routed total (stages)** | 1747 | 3798 |
| hybrid router (dense matrix) | 178 | 168 |
| hybrid `all_gather_rows` x2 | 64 | 37 |
| hybrid local expert slices (stream / dispatch ops) | 1164 | 2791 |
| hybrid `reduce_scatter_rows` | 72 | 32 |
| **hybrid routed total (stages)** | 1478 | 3028 |
| shared expert (TP, both layouts) | 43 | 41 |
| `all_reduce_cols` (both layouts) | 49 | 49 |
| EP `TtMoE.forward` (whole layer, traced) | 1760 | 3826 |
| hybrid `TtMoE.forward` (whole layer, traced) | 1487 | 3091 |

Accuracy in the same run (worst PCC vs HF over the 8 chips): 32 users/row EP 0.999893, hybrid 0.999891, EP vs
hybrid 0.999987 (4 users/row: EP 0.999890, hybrid 0.999886, EP vs hybrid 0.999982). Dispatched rows per chip: [3, 4, 7, 7, 5, 5, 6, 5]
of 8 gathered at 4 users/row, [41, 39, 44, 44, 45, 46, 41, 34] of 64 at 32 (max/mean 1.09). The K-sum as one `fast_reduce_nc` kernel equals `ttnn.sum` (PCC 0.999999); it
reports the tile-padded row count, so a metadata view restores T rows (`GLM53_EP_KSUM=fast`).

Reading: on a (2,4) block every chip holds 36 experts and at 32 users/row nearly all of them are active, so both
layouts stream ~960 MB per chip per layer and sit at the DRAM floor (~2.7 ms at 358 GB/s; the EP pass runs at
~318 GB/s): the EP expert pass (3.02 ms) is not faster than the hybrid slices (2.79 ms) here, and the EP exchange
costs 643 us against the hybrid's 69 us, almost all of it in the combine stage. At 4 users/row (<= 7 active experts
per chip) the EP kernels fall to ~38 % of peak (1.36 ms), the same fixed low-batch cost the single-chip EP test shows
(36 experts, 11 active: 1221 us; hybrid stream 1219 us) — a kernel property (OPT_EXPERTS), not a mesh-path cost. The block cannot show EP32's premise (full-width kernels at 66-74 % of
peak vs 38-54 % on 1/8 slices with 9 experts per chip); the 8x4 run decides. The combine stage is the item to fix
first.

Op-by-op (`tests/test_a2a_perf.py::test_ep_combine_breakdown`, traced, us):

| op | 4 users/row | 32 users/row |
|---|---|---|
| x / w / idx to_layout, concat, typecast (6 ops) | 6-11 each | 7-13 each |
| `all_to_all_dispatch` (links 1 / 2) | 18 / 18 | 36 / 27 |
| expert kernels | 1364 | 3046 |
| out to ROW_MAJOR, zero-slot clone | 12, 12 | 11, 13 |
| `all_to_all_combine` (links 1 / 2) | 162 / 141 | 494 / 310 |
| slots tilize | 36 | 80 |
| K-sum `ttnn.sum` / `fast_reduce_nc` | 29 / 16 | 17 / 17 |

The combine op itself is the cost. Its program runs on `min(num_links, ..)` worker cores (one or two per chip); the
reader issues one NOC read + barrier per gathered token's metadata page and per contributed row and the writer
flushes per local write, so the op is serialized per gathered token (~100 us fixed plus a few us per token and row)
while the dispatch, which moves the same rows, takes 18 us. On the 8x4 at 32 users/row every chip scans 256
gathered tokens: expect this op to dominate the EP exchange there. Options, in order: num_links = the control
plane's maximum (2 on the block: −13 % at 4 users/row, −37 % at 32), a model-local combine under custom_ops (the dispatch's write pattern, many cores),
or an op-level fix (batched reads, more worker cores). The exchange otherwise meets the adoption criterion
(dispatch + its 6 ops = 33-68 us). With the op's own link choice (2 on the block; now the model default,
`GLM53_A2A_LINKS=0`; `a2a_links2.log`): golden test exact at both sizes, 32 users/row EP 0.999893 vs HF as with one
link, combine stage 390 us, EP routed stages 3615 us (forward 3662) against the hybrid's 3052 (3087).

### Fabric
- FABRIC_1D: validated (op-level golden `test_dispatch_combine_golden` byte-exact / PCC 1.000000 at Td = 8 and 64,
  the layer test above). It is also what the DeepSeek-V3 Galaxy model runs under (`get_fabric_config`) and the only
  fabric the Blackhole CI exercises these ops under (FABRIC_1D + Linear + num_links=1).
- FABRIC_2D with the fabric-derived Mesh topology: NOT run. The one attempt on rows 6-7 hung after mesh creation and
  the SIGKILL left the block's ethernet cores busy for every later run (board reset). On the 8x4 the whole model must
  therefore run under FABRIC_1D when the EP layout is on (the async gathers / reduce-scatters and the generic
  all-reduce take the Linear topology there; test_mesh_ccl.py covers them under the block's fabric).
