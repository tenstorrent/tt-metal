# GLM-5.3-Flash prefill (layers 0-4): sharding on 4x Blackhole p300c (mesh 2x2, FABRIC_2D)

The gate (PL.1, `python -m models.demos.common.bringup.plan.check_plan`) computes the per-chip numbers from `plan.yaml`
against the real checkpoint shapes and writes them to `results/plan_memory.json`. This page gives the reasoning. The
step-to-op mapping is in `components.yaml`.

Target: 56320 tokens in 5120-token chunks (ladder up to 8192-token chunks), 1 user, bf16 activations. Each chip has
32 GB of DRAM; the budget is 27.2 GiB (15% headroom). Text decoder only: the vision tower and the MTP layer (45) are
not loaded (owner rule).

## Scope: layers 0-4, not 45

The full model does not fit on this box. The routed experts of 42 MoE layers at bfp8 need 42 x 1.79 GiB = 75 GiB per
chip (at bfp4 it would still be 42 GiB per chip, and the owner rule rules out bfp4 anyway). Layers 0-4 cover all three
block types: kda_dense (0-2), dsa_moe (3) and kda_moe (4). Layers 5-44 are `skip` in `plan.yaml`. A subset result is
never reported as a full-model result.

## Scheme

Mesh coordinates are (row r, col c), with axis 0 = rows and axis 1 = columns. There are two chip numberings, both
fixed at load:

- **TP index d = 2r + c** (row-major, `ttnn.ShardTensorToMesh` over the 2x2 mesh). It is used for the dense MLP, the
  shared expert, the LM head and the DSA query quarters.
- **EP device index 2c + r** (DeepSeek `ExpertMapping`, column-major) for the routed experts.

What runs where:

- **Residual: 4 mHC streams, replicated.** The residual is `[S, 4 x 4096]` on every chip. That is the reference
  boundary `[S * 4, H]` token-major, which as a packed tensor is the DeepSeek mHC layout `[1, 1, T, n*C]`. The mHC
  coefficients (projection plus 20-step Sinkhorn), the stream collapse, the residual mixes, the norms and the router
  all run identically on every chip, with no CCL.
- **KDA attention (layers 0-2, 4): TP=2 by head on axis 1, SP=2 over the sequence on axis 0.** This is `ttKDA`'s 2D
  mode, which needs two distinct axes. Column c holds heads 32c..32c+31. Row r takes its half of the chunk: a local
  `mesh_partition`, then the conv halo and the recurrent-state prefix are exchanged over axis 0 inside the module. The
  output goes through a reduce-scatter on axis 1, then all_gathers on axis 1 and axis 0 restore `[S, 4096]`.
- **DSA attention (layer 3): replicated weights, queries split 4 ways by sequence.** Chip d takes query rows
  d*S/4 .. (d+1)*S/4 - 1, with all 64 heads (`sparse_sdpa` needs a multiple of 32 heads per chip). The MLA latent
  cache and the indexer's pooled-key cache are replicated. Every chip computes the latent and the pooled keys for all
  S rows, so writing the caches needs no CCL. One all_gather (axis 1, then axis 0) after o_proj.
- **Dense MLP (layers 0-2) and shared expert (layers 3, 4): TP=4 by intermediate** (3072 and 512 per chip), followed
  by `ttnn.all_reduce(cluster_axis=None)`, which runs on axis 1 and then axis 0.
- **Routed experts: EP=4 with the DeepSeek 2D dispatch.** Dispatch runs along axis 0 with a group size of 2; each
  column is one dispatch group of 144 experts. Chip (r, c) holds experts 144c + 72r .. 144c + 72r + 71. Each chip
  dispatches its row's S/2 tokens within its column. After combine and the weighted sum, `all_reduce(cluster_axis=1)`
  adds the other group, and `all_gather(cluster_axis=0)` restores `[S, 4096]`. This is the layout MiMo-V2.6 ran on this
  mesh shape (`mimo_v2_6_d_p_2x2`).
- **Embedding** is replicated. The **LM head** is vocab-sharded 4 ways (38720 rows per chip), and the logits are
  gathered on both axes, for the last tokens only.
- **Weights.** FP8 e4m3 tensors are multiplied by their 128x128 `weight_scale_inv` at load, giving bf16 for non-expert
  weights and bfp8 for the routed experts (never bfp4). KDA, indexer, kv_b, embedding and LM head weights are bf16 as
  stored. The mHC projections, the router, A_log and dt_bias are fp32. Every matmul runs at HiFi4.

## kda_dense layer (layers 0-2)

KDA: 64 heads x 128; q/k/v 4096 -> 8192, each with a 4-tap conv and SiLU; low-rank forget gate bounded at -5; beta;
chunked delta rule with carried state; gated RMSNorm; o_proj. The dense MLP is a clamped SwiGLU with intermediate 12288.

| Component | Placement | Each chip holds | Collective after | Per chip |
|---|---|---|---|---|
| attn_hc: mHC projection [16384 -> 24] fp32 + Sinkhorn | replicated | hc_attn_fn / base / scale | none | 1.6 MB |
| attn_collapse: sum of pre-weighted streams | replicated | nothing | none | 0 |
| attn_norm (input_layernorm) | replicated | [4096] | none | 8 KB |
| KDA q/k/v/b/f_b/g_b projections, conv taps, dt_bias, A_log (bf16 / fp32) | column-parallel on axis 1 (TP=2) | heads 32c..32c+31 | none | 101 MB |
| KDA f_a, g_a (rank 128), o_norm | replicated inside each TP rank's fused projection | full | none | 2 MB |
| KDA recurrence over S/2 rows (SP=2 on axis 0) | local heads, sequence half r | recurrent state 32 x 128 x 128 fp32, conv tail 3 x 12288 | SP halo + prefix exchange on axis 0 (small, inside ttKDA) | 2.1 MB state |
| KDA o_proj [8192 -> 4096] | row-parallel (TP=2) | 4096 input columns | **reduce_scatter axis 1**, **all_gather axis 1**, **all_gather axis 0** -> [S, 4096] | 34 MB |
| attn_residual: post * attn_out + comb^T @ in | replicated | nothing | none | 0 |
| ffn_hc, ffn_collapse, ffn_norm | replicated | hc_ffn_*, [4096] | none | 1.6 MB |
| mlp: gate/up [4096 -> 12288], down, fp8 -> bf16 | gate/up column-parallel, down row-parallel (TP=4) | intermediate 3072d..3072d+3071 | **all_reduce, axis 1 then axis 0** [S, 4096] | 72 MB |
| ffn_residual | replicated | nothing | none | 0 |
| **Layer total** | | | **KDA: RS + 2 AG; MLP: 1 all_reduce (2 stages)** | **about 0.21 GB** |

## dsa_moe layer (layer 3)

DSA: q_lora 1536, kv_lora 512, 64 heads, qk 256 / v 256, all NoPE. The indexer has 32 heads x 128 over keys pooled by 4
with a learned softmax; it picks the top 512 pools plus up to 3 tail tokens. MoE: 288 experts, top-8, 1 shared expert,
sigmoid noaux_tc router with routed scale 2.5, clamped SwiGLU 2048.

| Component | Placement | Each chip holds | Collective after | Per chip |
|---|---|---|---|---|
| attn_hc, attn_collapse, attn_norm | replicated | as above | none | 1.6 MB |
| q_a: q_a_proj [4096 -> 1536] + q_a_layernorm, all S rows | replicated | full | none | 13 MB |
| indexer keys: wk [4096 -> 128] + LayerNorm, gate [4096 -> 128] + ape, pooled by 4, all S rows | replicated | full | none | 2 MB |
| indexer queries: wq_b [1536 -> 32 x 128], weights_proj [4096 -> 32]; score over pools, top-512, expand to tokens | replicated weights, query rows split by sequence (quarter d) | S/4 query rows | none (the component test boundary gathers topk) | 13 MB |
| pooled-key cache [max_seq / 4, 128] bf16 | replicated | whole cache | none | 3.6 MB |
| MLA latent: kv_a_proj [4096 -> 512] + kv_a_layernorm, all S rows -> cache [max_seq, 512] | replicated | whole cache | none | 4 MB + 58 MB cache |
| MLA q_b [1536 -> 16384], absorbed w_uk / w_uv [64, 256, 512] (from kv_b) | replicated weights, S/4 query rows, all 64 heads | full | none | 84 MB |
| sparse_sdpa over the 2051 selected latent rows | local (quarter d) | nothing | none | 0 |
| o_proj [16384 -> 4096], fp8 -> bf16 | replicated, S/4 rows | full | **all_gather dim -2, axis 1 then axis 0** -> [S, 4096] | 134 MB |
| attn_residual, ffn_hc, ffn_collapse, ffn_norm | replicated | as above | none | 1.6 MB |
| router [4096 -> 288] + correction bias, fp32 | replicated | full | none | 4.7 MB |
| routed experts 288 x clamped SwiGLU 2048, bfp8; dispatch -> unified (high_precision, ClampedSiluGlu) -> combine | expert-parallel, dispatch axis 0, group = column | experts 144c + 72r .. + 71 | dispatch + combine over axis 0 (fabric); **all_reduce axis 1** [S/2, 4096]; **all_gather axis 0** | 1.79 GiB |
| shared expert [4096 -> 2048 -> 4096], fp8 -> bf16 | TP=4 by intermediate (512 per chip) | intermediate 512d.. | **all_reduce, axis 1 then axis 0** | 12 MB |
| moe_add, ffn_residual | replicated | nothing | none | 0 |
| **Layer total** | | | **attention: 2 AG; MoE: dispatch/combine + AR + AG; shared: 1 AR (2 stages)** | **about 2.1 GB** |

## kda_moe layer (layer 4)

The attention is the same as in kda_dense (layer 4's weights). The router, experts and shared expert are the same as in
dsa_moe.

| Component | Placement | Each chip holds | Collective after | Per chip |
|---|---|---|---|---|
| mHC steps, norms | replicated | as above | none | 3.2 MB |
| KDA attention | TP=2 on axis 1, SP=2 on axis 0 | heads 32c.., sequence half r | RS axis 1, AG axis 1, AG axis 0 | 137 MB + 2.1 MB state |
| router, routed experts, shared expert, moe_add, ffn_residual | as dsa_moe | experts 144c + 72r .. + 71 | as dsa_moe | 1.8 GiB |
| **Layer total** | | | **KDA: RS + 2 AG; MoE: dispatch/combine + AR + AG; shared: 1 AR** | **about 2.1 GB** |

## Model level

| Component | Placement | Each chip holds | Collective | Per chip |
|---|---|---|---|---|
| embed_tokens [154880, 4096] bf16, then ttnn.repeat to 4 streams | replicated | full table | none | 1.18 GiB |
| final norm: mean of the 4 streams, then RMSNorm | replicated | [4096] | none | 8 KB |
| lm_head [154880, 4096] bf16 (untied) | vocab-sharded over 4 chips | rows 38720d .. 38720d + 38719 | all_gather logits, axis 1 then axis 0 (last tokens) | 0.30 GiB |
| vision tower, MTP layer 45, layers 5-44 | skipped | none | none | 0 |

## Per-chip total (from the gate, GiB)

| Group | Per chip |
|---|---|
| activations (planner estimate, one 8192-token chunk) | 7.00 |
| routed experts, bfp8 (2 layers x 72 experts) | 3.59 |
| embedding (replicated) | 1.18 |
| KDA attention weights (4 layers, TP=2) | 0.52 |
| constant tables, dispatch tables, misc | 0.30 |
| LM head, vocab-sharded | 0.30 |
| DSA attention weights (layer 3, replicated) | 0.22 |
| dense MLP (layers 0-2, TP=4) | 0.21 |
| KDA state double buffer and contract state copy | 0.15 |
| MLA latent cache (layer 3, 56320 x 512 bf16) | 0.05 |
| shared expert, mHC, indexer, router, norms | 0.06 |
| KDA recurrent state + conv tail (4 layers), pooled-key cache | 0.01 |
| **Total** | **13.59 of 27.20 budget** |

The state at 56k is small: the whole subset holds less than 70 MB per chip. The activations dominate because the
4-stream residual is 4 times the hidden width (268 MB per copy at 8192 tokens) and it is replicated.

## Collectives per layer (one 5120-token chunk; [S, 4096] bf16 = 42 MB)

| Layer type | Collectives |
|---|---|
| kda_dense | KDA: reduce_scatter [S/2, 4096] on axis 1, all_gather on axis 1 (-> [S/2, 4096]), all_gather on axis 0 (-> [S, 4096]); small SP halo / state-prefix exchanges on axis 0 inside ttKDA. MLP: all_reduce [S, 4096] on axis 1, then axis 0 |
| dsa_moe | attention: all_gather [S/4 -> S/2] on axis 1, [S/2 -> S] on axis 0. Experts: offset_cumsum histograms [1, 288] on axis 0; dispatch and combine over axis 0 (about 2 x S/2 x 4096 bf16 = 21 MB out and back); all_reduce [S/2, 4096] on axis 1; all_gather [S/2 -> S] on axis 0. Shared expert: all_reduce [S, 4096] on axis 1, then axis 0 |
| kda_moe | KDA as kda_dense; experts and shared expert as dsa_moe |

Per chunk: one embedding lookup, the final norm, and the LM-head matmul on the last tokens.

## Accuracy notes the component tasks need

- **Router.** The top-8 over 288 experts is chosen on sigmoid + correction bias in fp32 (known issues: bf16 or TF32
  choice scores flip near ties). The weights come from the unbiased sigmoid, renormalised and multiplied by 2.5.
- **Experts.** Use the unified kernel with `high_precision=True`, bf16 x, HiFi4 and fp32 dest (known issues: bf16 L1
  partials and bfp8 activations on outlier channels). Check the MoE input's per-channel max on layers 3 and 4.
- **Indexer.** `indexer_score_dsa` outputs bf16 scores, while the reference ranks fp32 scores, so some near-tie pools
  will be selected differently. The indexer test gates selection overlap, not exact match.
- **mHC.** The Sinkhorn op is fp32-only, and the bf16 residual is cast to fp32 for the projection. Each of the 16
  addcmul terms of a residual mix adds rounding on the replicated 4-stream residual, so accumulate in fp32 where the
  reference does.
- **Constants.** Every table is built once at load for max_seq and sliced on device per chunk: KDA chunk-start
  scalars, the index tail table, the pool-causal mask, the dense index rows for queries before position 2047, the
  sentinel pad and the dispatch tables.

## Departures from the reference plans (MiMo 2x2, Kimi K2.7, ERNIE), one reason each

- **Layer subset, not the whole model.** The full model's experts need 75 GiB per chip at bfp8, and layers 0-4 already
  cover all three block types.
- **The residual is 4 streams wide and replicated, not `[S, H]`.** mHC mixes the streams with a per-token 4x4 matrix,
  and every component boundary in the reference is either the 4-stream residual or `[S, *]`. Replicating it keeps
  every boundary identical to the reference. Splitting it by sequence is a later perf option.
- **KDA is TP=2 x SP=2, not TP=4.** `ttKDA` shards heads over one mesh axis and the sequence over the other (the
  constructor rejects anything else), and a 2x2 mesh has no size-1 axis. SP=2 halves the recurrence per chip and uses
  the module's own SP state-prefix path. Fallback if its SP segment order does not match contiguous halves: SP=1 on
  each 1x2 row submesh.
- **DSA attention is split by sequence with replicated weights, not TP by head.** 64 heads over 4 chips is 16 per
  chip, below `sparse_sdpa`'s 32. With all heads on every chip there is no head-to-sequence reshard, the caches are
  tiny and replicated, and the only CCL is one all_gather. The cost is 0.22 GB of replicated weights per DSA layer.
- **The indexer is composed from existing ops, not deferred to op-gen.** Key pooling is a reshape, slices and an
  elementwise softmax over 4. `indexer_score_dsa` runs unmasked, and a constant mask applies the pool-level causal
  rule on the chunk's own pools. The sentinel compaction `sparse_sdpa` needs comes from the selection itself: rows at
  position 2047 and later have no sentinels in the pool part, and earlier rows are the constant "all tokens up to q"
  row.
- **The MLA latent cache is 512 wide, not 576.** GLM-5.3 has no RoPE columns, and `sparse_sdpa` takes the head dim from
  the tensors.
- **Routed experts use the MiMo 2x2 dispatch layout (axis 0, 2 chips per group).** It is the layout the DeepSeek
  dispatch was built for and is proven on this mesh shape. The forks behave as the source ops on a 2-device axis.
  Not verified: 72 local / 288 global experts (MiMo ran 64 / 256). Fallback: MiMo's per-expert loop.
- **The shared expert has its own TP=4 reduce.** Its boundary is separate in the reference graph. Adding its partial
  into the experts' reduce is a perf option for the assembled model.
- **Non-expert fp8 weights go to bf16, not bfp8.** bf16 holds e4m3 x block scale exactly, and all non-expert weights
  together are under 1 GB per chip.
