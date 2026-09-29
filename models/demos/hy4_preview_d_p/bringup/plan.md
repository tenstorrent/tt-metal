# Hy4 Preview prefill (layers 0-5): sharding on 4x Blackhole p150b (mesh 2x2, FABRIC_2D)

The gate (PL.1, `python -m models.demos.common.bringup.plan.check_plan`, output in `results/plan_memory.json`) computes
the numbers from `plan.yaml` and the real checkpoint shapes. This page gives the reasoning. `components.yaml` maps each
step to its ops.

Target: 56320 tokens in 5120-token chunks (the ladder goes up to 8192-token chunks), 1 user, bf16 activations, fp32 iHC.
Each chip has 32 GB of DRAM; the budget is 27.2 GiB (15% headroom). Text decoder only: the MTP layer is not loaded.

## Scope: layers 0-5, not 78

The spec runs layers 0-5. They cover all three block types: `dense_full` (layer 0), `moe_full` (layers 1 and 5, own
indexer) and `moe_shared` (layers 2-4, reuse layer 1's top-k). The full model does not fit this box: 77 MoE layers
of bfp8 routed experts come to 77 x 2.39 = 184 GiB per chip. Layers 6-77 and `model.mtp_layers.*` are `skip` in
`plan.yaml`. A subset result is never reported as a full-model result.

## Scheme

Mesh coordinates are (row r, col c). The layout is **SP=2 over rows (axis 0) x TP=2 over columns (axis 1)**. This is
the (2, 2) layout that `deepseek_v3_d_p`'s ttMLA and TtIndexer were built for. It is also what `tests/sparse_mla` runs
on the QuietBox (`sparse_mla_mesh.py`: SP2xTP2, FABRIC_2D).

- **Activations.** Chip (r, c) holds chunk rows `[r*S/2, (r+1)*S/2)` and hidden columns `[3072c, 3072(c+1))` of every
  stream. The engine's uint32 input is already `[sp=2, 1, chunk/2]` over axis 0, so the ids need no gather.
- **Residual.** The 4 iHC streams stay resident in fp32 as `[S/2, 4 x 3072]` per chip: each stream is split by
  column and packed along the last dim, like `tt_mhc`. iHC mixes reduce over all 24576 values. Each chip computes its
  partial mixes and partial sum of squares, then does one `[S/2, 32]` fp32 all_reduce over axis 1.
- **Attention: TP=2 by head, SP=2 by sequence.** Each chip has 32 of the 64 heads, which is exactly what
  `sparse_sdpa` needs (per-chip heads a multiple of 32). The weights are split over the 2 columns and replicated over
  the 2 rows. The MLA latent cache (one 576-wide row per token) and the index-key cache are striped block-cyclic over
  all 4 chips (ttMLA's KV dedup). Each layer gathers the prefix into one replicated scratch for `sparse_sdpa`. The
  indexer runs a ring over axis 0.
- **Routed experts: EP=4, with the DeepSeek 2D dispatch** proven on 2x2 by `mimo_v2_6_d_p_2x2`. Dispatch runs along
  axis 0: each column is one dispatch group of 128 experts, and chip (r, c) holds experts `128c + 64r .. +63`. Each
  chip dispatches its row's S/2 tokens within its column. After combine and the weighted sum, a reduce_scatter over
  axis 1 adds the other group and returns the residual's column split.
- **Dense MLP (layer 0) and shared expert: TP=2 over columns.** gate/up are column-parallel and down is row-parallel.
  The input is the row's full hidden (ffn_norm all_gathers once over axis 1), and a reduce_scatter over axis 1 closes
  each block.
- **Weight dtypes.** Attention, the indexer, the dense MLP, the shared expert, the embedding and the LM head are bf16
  as stored. The router and iHC are fp32. The routed experts are bf16 in the checkpoint and **bfp8** on the device
  (never bfp4: the checkpoint is not 4-bit). Every matmul runs at HiFi4 with fp32 accumulation, and so does
  `sparse_sdpa`.
- **Code.** The Hy4 modules go in `models/demos/hy4_preview_d_p/tt`. They copy or wrap the `deepseek_v3_d_p` and
  `mimo_v2_6_d_p_2x2` code, which stays read-only. Op changes go through the forks in `ttnn/ttnn/bringup`
  (dispatch / combine / offset_cumsum, `unified_routed_expert_moe` with ClampedSiluGlu, `rms_norm`).

## dense_full layer (layer 0)

MLA with 64 heads, q_lora 2048, kv_lora 512, qk 192 + 64, v 256, scale 1/16, a learnable sink per head and a sigmoid
output gate. It has its own indexer (32 x 128, top-2048) and a dense SwiGLU 18432. Chunk = 5120 in the collective sizes
(2560 rows per chip).

| Component | Placement | Each chip holds | Collective after | Per chip |
|---|---|---|---|---|
| attn_hc (hc_attn_layer fn [8, 24576], base, scale) | fn split by column | fn [8, 12288] fp32 | **all_reduce axis 1**, [S/2, 32] fp32 (0.3 MB) | 0.4 MB |
| attn_hc_pre (sum_j pre_j x stream_j) | local | 4 x [S/2, 3072] | none | 0 |
| attn_norm (input_layernorm, distributed RMSNorm) | weight split by column | [3072] | all_gather of [S/2, 32] stats, axis 1 | 6 KB |
| q_a_proj [6144 -> 2048] + q_a_layernorm (eps 1e-6) | K-split over columns | [3072, 2048] bf16 | **reduce_scatter + all_gather axis 1** [S/2, 2048] (10 MB) | 13 MB |
| indexer: wq_b [2048 -> 32 x 128] | replicated | all 32 index heads | none | 17 MB |
| indexer: wk [6144 -> 128] + k_norm LayerNorm (eps 1e-5), weights_proj [6144 -> 32] | K-split over columns | [3072, 128], [3072, 32] | reduce over axis 1 ([S/2, 128], [S/2, 32]) | 1 MB |
| indexer: RoPE (last 64 dims, interleaved), key cache, ring score, top-2048 | cache block-cyclic over 4 chips | index keys: a quarter of [56320, 128] bf16 | ring gather of the keys over axis 0 (up to 14 MB at 56k); TP-inner gather over axis 1 | 3.6 MB cache |
| attention: kv_a_proj_with_mqa [6144 -> 576] + kv_a_layernorm, RoPE on k_rope | K-split over columns | [3072, 576] bf16 | reduce over axis 1 [S/2, 576] (3 MB) | 3.5 MB |
| attention: MLA latent cache [56320, 576] bf16 | block-cyclic over 4 chips | a quarter of the rows | **full-mesh gather of the prefix** (cluster_axis None, up to 65 MB at 56k) into one shared scratch | 16 MB |
| attention: q_b_proj [2048 -> 64 x 256], kv_b_proj (-> wkv_b1 192->512, wkv_b2 512->256) | split by head | 32 heads | none | 48 MB |
| attention: sparse_sdpa (absorbed 576 / 512, top-2048, sink x 16, scale 1/16) | local | 32 heads over the gathered prefix | none | 128 B sinks |
| attention: linear_gate [6144 -> 64 x 256], sigmoid | split by head | [6144, 8192] bf16 | **all_gather attn_norm axis 1** [S/2, 6144] (31 MB) before it | 101 MB |
| attention: o_proj [16384 -> 6144] | row-parallel | [8192, 6144] bf16 | **reduce_scatter axis 1** -> [S/2, 3072] (31 MB) | 101 MB |
| attn_residual (stream_j += post_j x attn_out) | local | fp32 | none | 0 |
| ffn_hc, ffn_hc_pre | as attn_hc | fn [8, 12288] fp32 | **all_reduce axis 1**, [S/2, 32] fp32 | 0.4 MB |
| ffn_norm (post_attention_layernorm) | replicated | [6144] | **all_gather ffn_x axis 1** [S/2, 6144] (31 MB) before it | 12 KB |
| mlp: SwiGLU 18432 (gate, up, down), fp32 intermediates | gate/up column, down row | intermediate 9216c .. 9216c + 9215 | **reduce_scatter axis 1** -> [S/2, 3072] (31 MB) | 340 MB |
| ffn_residual | local | fp32 | none | 0 |
| **Layer total** | | | **3 tiny all_reduces, 2 all_gathers, 3 reduce_scatters, 1 q_a gather, KV prefix gather, index ring** | **about 0.60 GiB** |

## moe_full layer (layers 1 and 5)

Attention and the indexer are as in layer 0. The MoE has 256 routed experts, top-8, SwiGLU 2048 clamped at 10, and 1
shared expert. The router is sigmoid + correction bias with normalised top-8, scaled by 2.827.

| Component | Placement | Each chip holds | Collective after | Per chip |
|---|---|---|---|---|
| attn_hc .. attn_residual, indexer | as layer 0 | as layer 0 | as layer 0 | 0.30 GB |
| ffn_hc, ffn_hc_pre, ffn_norm | as layer 0 | as layer 0 | all_reduce [S/2, 32]; all_gather ffn_x axis 1 | 0.4 MB |
| router [6144 -> 256] + correction bias, fp32 (linear, sigmoid, add, topk, gather, renorm, x 2.827) | replicated | full [256, 6144] fp32 | none (both chips of a row compute the same routing) | 6.3 MB |
| routed experts: masked_bincount, offset_cumsum, dispatch, unified_routed_expert_moe (ClampedSiluGlu, high_precision), combine, post_combine_reduce | expert-parallel, dispatch axis 0, group = column | experts 128c + 64r .. +63, bfp8 | offset_cumsum histograms over axis 0; dispatch and combine over axis 0 (about 2 x S/2 x 6144 bf16 = 63 MB each way); **reduce_scatter axis 1** -> [S/2, 3072] | 2.39 GiB |
| shared expert: SwiGLU 2048, no clamp | gate/up column, down row | intermediate 1024c .. +1023 | **reduce_scatter axis 1** (fusable with the experts' one) | 38 MB |
| moe_combine (experts + shared) | local | | none | 0 |
| ffn_residual | local | fp32 | none | 0 |
| **Layer total** | | | **as layer 0, plus dispatch/combine over axis 0 and a second reduce_scatter (fusable)** | **about 2.71 GiB** |

## moe_shared layer (layers 2-4)

This is moe_full without the indexer. `topk_shared` hands on layer 1's device top-k tensor for the same chunk, with
no op. The layer has no indexer weights and no index-key cache, but it keeps its own MLA latent cache.

| Component | Placement | Each chip holds | Collective after | Per chip |
|---|---|---|---|---|
| attention (no indexer), iHC, norms | as moe_full | as moe_full | as moe_full, no index ring | 0.28 GB |
| topk_shared | local | layer 1's uint32 [S/2, 2048] indices (kept alive until the next full layer) | none | 0 |
| router, routed experts, shared expert, moe_combine, ffn_residual | as moe_full | as moe_full | as moe_full | 2.61 GB |
| **Layer total** | | | **as moe_full minus the indexer's** | **about 2.69 GiB** |

## Model level

| Component | Placement | Each chip holds | Collective | Per chip |
|---|---|---|---|---|
| embed_tokens [120832, 6144] bf16 | hidden split over columns | [120832, 3072] | none (ids already split by row); ttnn.repeat to 4 streams | 0.69 GiB |
| hc_head (fn [4, 24576] fp32) + final norm | fn split by column; norm replicated | fn [4, 12288] | all_reduce [S/2, 32] axis 1; all_gather [S/2, 6144] axis 1 before the norm | 0.2 MB |
| lm_head [120832, 6144] bf16 (untied; HF runs it in fp32) | vocab-sharded over 4 chips | rows 30208d .. +30207 | the last token's hidden goes to all 4 chips; all_gather the logits over both axes | 0.35 GiB |
| MTP layer, layers 6-77 | skipped | none | none | 0 |

## Per-chip total (from the gate, GiB)

| Group | Per chip |
|---|---|
| routed experts, bfp8 (5 layers x 64 experts) | 11.95 |
| activations (planner estimate, one 8192-token chunk) | 5.00 |
| attention weights (bf16, 6 layers) | 1.48 |
| embedding (hidden split over columns) | 0.69 |
| KV gather scratch, indexer ring buffer, RoPE and dispatch tables, CCL buffers | 0.35 |
| LM head, vocab-sharded | 0.35 |
| dense MLP (layer 0) | 0.32 |
| shared expert (5 layers) | 0.18 |
| contract KV copy (reserve) | 0.10 |
| MLA latent cache (6 layers, a quarter of 56320 x 576 bf16 per chip) | 0.09 |
| indexer weights (3 layers) | 0.05 |
| router (fp32) | 0.03 |
| indexer key cache (layers 0, 1, 5) | 0.01 |
| iHC, norms, sinks | 0.01 |
| **Total** | **20.60 of 27.20 budget** |

That leaves 6.6 GiB of slack. It covers a larger activation peak than estimated, or a bf16 copy of a layer's experts
if the fallback loop needs one. bf16 routed experts would not fit (4.51 GiB x 5 = 22.5 GiB for the experts alone).

## Collectives per layer (one 5120-token chunk; per chip 2560 rows; [S/2, 6144] bf16 = 31 MB)

| Layer type | Collectives |
|---|---|
| dense_full | iHC: 2 x all_reduce [S/2, 32] fp32 over axis 1. attn_norm: stats all_gather over axis 1. q_a: reduce_scatter + all_gather [S/2, 2048] over axis 1. Indexer: reduce over axis 1 for wk / weights_proj; key ring over axis 0 inside `ring_indexer_score_dsa`, plus a TP-inner gather over axis 1. Attention: reduce over axis 1 for kv_a; full-mesh prefix gather of the latent cache (up to 65 MB at 56k); all_gather attn_norm [S/2, 6144] over axis 1 for the gate; reduce_scatter o_proj over axis 1. FFN: all_gather ffn_x over axis 1; reduce_scatter mlp_out over axis 1. |
| moe_full | As dense_full, but the FFN part is: all_gather ffn_x over axis 1; offset_cumsum histograms over axis 0; dispatch and combine over axis 0 (fabric); reduce_scatter over axis 1 for the experts and for the shared expert (one if the partials are added first). |
| moe_shared | As moe_full, without the indexer's collectives. |

Per chunk there is one embedding lookup, the hc_head and final norm, and the LM head on the last token.

## Activation estimate (5.0 GiB)

It is sized for the 8192-token ladder chunk, which is 4096 rows per chip.

- **Residual streams.** fp32 `[4096, 12288]` is 201 MB per copy; in, h_mid and out can be alive together, 0.6 GB.
- **MoE.** The worst-case dispatch buffer is 8 x (2 x 4096) rows x 6144 bf16 = 0.81 GB. A chip receives from both
  chips of its column, and the capacity factor is 8 (all 8 experts of a token on one chip). The combine output
  `[4096, 8, 6144]` bf16 is 0.40 GB, plus the reduce output.
- **Attention.** Absorbed q `[32, 4096, 576]` bf16 in tile and row-major copies is 0.15 GB each, sparse_sdpa's output
  0.13 GB, and the q_b and gate outputs 67 MB each.
- **Indexer.** Logits for 1280 query rows per chip over 56320 keys in fp32 are 0.29 GB at the target.
- **Dense MLP (layer 0).** fp32 gate / up / h `[4096, 9216]` are 151 MB each.

About 2.4 GB at the MoE peak, doubled for fragmentation.

## Departures from the references, and why

- **SP2 x TP2 instead of MiMo 2x2's flat TP=4 with a replicated residual.** Hy4's KV cache is a single 576-wide latent
  that cannot be split by head, and 64 heads over 4 chips (16) would need ttMLA's head->sequence all_to_all around
  every sparse_sdpa. ttMLA and TtIndexer are built and tested for SP x TP on 2x2 at 32 heads per chip.
- **bf16 attention and indexer weights instead of ttMLA's bfp8.** Sinks and the top-k selection are
  precision-sensitive (known issues), and the memory is there (1.5 GiB for 6 layers).
- **bf16 index-key cache instead of TtIndexer's bfp8.** Top-k stability at a 56k-wide selection costs 7 MB per chip.
- **fp32 residual streams.** HF runs iHC in fp32, and a copy costs 126 MB per chip at 5120 tokens.
- **The indexer's 128 dims are permuted on the host.** Hy4 ropes the last 64 of 128 dims and TtIndexer's op ropes the
  first 64. So the wq_b / wk rows and the k_norm weight / bias are reordered to `[64..127 | 0..63]` (q.k, LayerNorm
  and ReLU(q.k) do not change), and index_key is un-permuted on read-back.
- **No host RoPE permutation in the MLA.** This reverses `hy4_preview_findings.md`: Hy4 is served with interleaved
  RoPE, and the checkpoint is already in that order (R.2 finding, known issues).
- **The sink is passed as sink x 16 with an explicit scale of 1/16.** `sparse_sdpa` multiplies the sink by the scale,
  and 1/16 is a power of two, so both are exact in bf16.

## Open items for the component steps

1. ttMLA runs q_a, the indexer and the attention in one forward, but the framework tests them as three steps. Its CCL
   helpers (`get_tt_ccl`, persistent gather buffers) are sized for one fixed local chunk length, and the ladder uses
   1024, 4096 and 2560 rows per chip. The Hy4 module must build per-length buffers, or build at max_seq and slice.
2. Indexer: whether `ring_indexer_score_dsa` takes bf16 q (TtIndexer feeds bfp8), and the top-k selection overlap
   against the golden at 56k keys. The test compares index sets, because the device order differs from the golden's
   ascending, -1 padded format.
3. `sparse_sdpa` with a sink at HiFi4 + fp32 dest: the L1 fit and the accuracy of the sink-dominated rows.
4. `unified_routed_expert_moe` with ClampedSiluGlu and `high_precision=True` together: check on the layer-1 golden
   early. The fallback is the per-expert loop (extract, ttnn.linear, ttnn.clamp, insert).
5. The block-cyclic caches need at least 64 tokens per chip per chunk (every ladder rung has 1024 or more), and state
   read-back must gather and un-permute them.
