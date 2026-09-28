# MiMo-V2.6-Flash-RL prefill (layers 0-5): sharding on 4x Blackhole p150b (mesh 2x2, FABRIC_2D)

The machine-checked numbers come from `plan.yaml` (gate PL.1, `python -m models.demos.common.bringup.plan.check_plan`,
output in `results/plan_memory.json`). This page gives the reasoning. It is re-planned from the 1x4 prior
(`models/demos/mimo_v2_6_d_p/bringup/plan.md`, and the modules it ended with in `models/demos/mimo_v2_6_d_p/tt/`);
each table says what changes for 2x2 and what carries over. Step-to-op mapping is in `components.yaml`.

Target: 56320 tokens in 5120-token chunks (ladder up to 8192-token chunks), 1 user, bf16 activations. Per-chip DRAM
32 GB, budget 27.2 GiB (15% headroom). Text decoder only; vision, audio and MTP are not loaded.

## Scope: layers 0-5, not 48

Unchanged from 1x4. The spec runs layers 0-5 (layer 0 full_dense, 1-4 sliding_moe, 5 full_moe). The full model does not
fit this box: the routed experts of 47 MoE layers in bfp8 are 47 x 1.59 GiB = 75 GiB per chip. Layers 6-47 are `skip`
in `plan.yaml`; a subset result is never reported as a full-model result.

## Scheme

Mesh coordinates are (row r, col c). Two chip numberings are used, both fixed at load:

- **TP index d = 2r + c** (row-major, `ttnn.ShardTensorToMesh` over the 2x2 mesh) for attention, the dense MLP and the
  LM head.
- **EP device index 2c + r** (DeepSeek `ExpertMapping`, column-major) for the routed experts: chip (r, c) holds experts
  `128c + 64r .. 128c + 64r + 63`.

The scheme:

- Residual stream `[S, 4096]` is **replicated** on all 4 chips (as on 1x4). Norms, residual adds and the router run
  identically on every chip.
- **Attention: TP=4 by head over the whole mesh.** Chip d takes checkpoint TP rank d's fused qkv slab as stored
  (`tp_size: 4`): 16 Q heads plus 1 KV head (full) or 2 KV heads (sliding). o_proj is row-parallel, then
  `ttnn.all_reduce(cluster_axis=None)`, which on a 2x2 mesh runs an all_reduce on axis 1 and then on axis 0
  (`ccl/all_reduce/all_reduce.cpp`).
- **Dense MLP (layer 0): TP=4**, the same reduce.
- **Routed experts: EP=4 with the DeepSeek 2D-mesh dispatch.** Dispatch runs along mesh axis 0 (2 chips per dispatch
  group, over the fabric). Each mesh column is one dispatch group of 128 experts, 64 per chip. At the MoE entry each
  chip keeps only its row's half of the tokens (`ttnn.mesh_partition` along the sequence, cluster_axis 0, a local split
  with no transfer). Each chip then dispatches its S/2 tokens within its column. After combine and the weighted sum,
  chip (r, c) holds its row half summed over group c. `all_reduce(cluster_axis=1)` adds the other group and
  `all_gather(cluster_axis=0)` puts the two row halves back together. The output is replicated `[S, 4096]`, so the
  component boundary is the same as on 1x4.
- **Embedding** replicated; **LM head** vocab-sharded 4 ways, logits gathered on both axes for the last tokens only.
- Weights (unchanged): fp8 qkv and dense MLP dequantized to **bf16**; o_proj, norms, sinks, embedding and LM head bf16
  as stored; router fp32; experts mxfp4 x e8m0 dequantized on the host and stored as **bfp8** (never bfp4). Every
  matmul runs at HiFi4.
- The modules start from the 1x4 prior's final versions, which use the bring-up forks: `ttnn.bringup.rms_norm` (with
  `return_residual_sum`), `ttnn.bringup.(chunked_)scaled_dot_product_attention` (V at 128), and
  `ttnn.bringup.unified_routed_expert_moe` (`high_precision`). They also use `ttnn.bringup.dispatch` / `combine` /
  `offset_cumsum`, which behave exactly as the source ops on a 2-device axis.

## full_dense layer (layer 0)

64 Q heads x 192, 4 KV heads, QK 192 / V 128, partial RoPE (first 64 dims, rotate-half, theta 1e7), no sink,
V x 0.707 folded into the V rows, scale 192^-0.5. MLP SwiGLU 16384.

| Component | Placement | Each chip holds | Collective after | Per chip | vs 1x4 |
|---|---|---|---|---|---|
| attn_norm (input_layernorm) | replicated | full [4096] | none | 8 KB | same |
| qkv_proj [4096 -> 13568], fp8 -> bf16, as Q [16 x 192] + KV [192 + 128] matmuls | column-parallel, TP rank d slab | 16 Q + 1 K + 1 V head (3392 rows) | none | 28 MB | chip d = 2r + c; no V pad (fork SDPA) |
| partial RoPE on q, k (dims 0-63) | local | 16 Q + 1 K heads | none | 0 | same |
| KV cache (full length) | local, 1 KV head | K [1, 56320, 192], V [1, 56320, 128] | none | 36 MB | V 128, not 192 |
| SDPA causal (chunk 0) / chunked SDPA (later), `ttnn.bringup` fork | local, GQA 16:1 | 16 Q vs 1 KV head | none | 0 | same |
| o_proj [8192 -> 4096], bf16 | row-parallel | 2048 input columns | **all_reduce, axis 1 then axis 0** [S, 4096] | 17 MB | reduce on 2 axes |
| attn_residual | replicated | full | none | 0 | same |
| ffn_norm (post_attention_layernorm) | replicated | full | none | 8 KB | same |
| dense MLP 16384 (gate, up, down), fp8 -> bf16 | gate/up column, down row | intermediate 4096d..4096d+4095 | **all_reduce, axis 1 then axis 0** [S, 4096] | 101 MB | reduce on 2 axes |
| mlp_residual | replicated | full | none | 0 | same |
| **Layer total** | | | **2 all_reduces (4 axis stages)** | **about 0.18 GB** | |

## sliding_moe layer (layers 1-4)

64 Q heads x 192, 8 KV heads, QK 192 / V 128, window 128, partial RoPE theta 1e4, per-head sink logit. MoE 256
experts, top-8, SwiGLU 2048, sigmoid noaux_tc router.

| Component | Placement | Each chip holds | Collective after | Per chip | vs 1x4 |
|---|---|---|---|---|---|
| attn_norm | replicated | full | none | 8 KB | same |
| qkv_proj [4096 -> 14848], fp8 -> bf16 | column-parallel, TP rank d slab | 16 Q + 2 K + 2 V heads (3712 rows) | none | 30 MB | chip d = 2r + c |
| partial RoPE, V x 0.707 (in the V rows), true scale folded into Q rows (SDPA scale 2^-4) | local | 16 Q + 2 K heads | none | 0 | same |
| KV cache (full length) | local, KV heads 2d, 2d+1 | K [2, 56320, 192], V [2, 56320, 128] | none | 72 MB | V 128 |
| SDPA causal, window 128, attention_sink, `ttnn.bringup` fork, preset S | local, GQA 8:1 | 16 Q vs 2 KV heads over 128 cached positions + chunk | none | 0 | same |
| attention_sink_bias [64] | sharded by head | 16 values (pre-divided by 2^-4) | none | 32 B | chip d |
| o_proj | row-parallel | 2048 input columns | **all_reduce, axis 1 then axis 0** [S, 4096] | 17 MB | reduce on 2 axes |
| attn_residual, ffn_norm | replicated | full | none | 8 KB | same |
| router [4096 -> 256] + correction bias, fp32 (sigmoid, add, topk, gather) | replicated, same routing everywhere | full | none | 4 MB | same |
| mesh_partition of x, idx, wts | local split by row | S/2 rows (row r: rows r*S/2 ..) | none | 0 | new |
| routed experts 256 x SwiGLU 2048, bfp8, dispatch -> unified (high_precision) -> combine | expert-parallel, dispatch axis 0 (2 chips), group = column | experts 128c + 64r .. +63 | dispatch + combine over axis 0 (fabric); **all_reduce axis 1** [S/2, 4096]; **all_gather axis 0** -> [S, 4096] | 1.71 GB | fabric dispatch replaces the local one |
| ffn_residual | replicated | full | none | 0 | same |
| **Layer total** | | | **1 all_reduce (2 stages) + dispatch/combine + 1 all_reduce + 1 all_gather** | **about 1.83 GB** | |

## full_moe layer (layer 5)

Attention as layer 0 (4 KV heads, one per chip, theta 1e7, no sink); router and experts as the sliding layers. Layer 5's
MoE input has outlier channels (|x| up to 131), so the expert input stays bf16 and the unified kernel runs
`high_precision=True` at HiFi4 (known issues), as on 1x4.

| Component | Placement | Each chip holds | Collective after | Per chip | vs 1x4 |
|---|---|---|---|---|---|
| attn_norm, attn_residual, ffn_norm | replicated | full | none | 16 KB | same |
| qkv_proj, RoPE, KV cache, chunked SDPA, o_proj | as layer 0 | 16 Q + 1 KV head (chip d) | **all_reduce, axis 1 then axis 0** after o_proj | 81 MB | as layer 0 |
| router, mesh_partition, routed experts, ffn_residual | as the sliding layers | experts 128c + 64r .. +63 | dispatch/combine axis 0; all_reduce axis 1; all_gather axis 0 | 1.72 GB | as the sliding layers |
| **Layer total** | | | as sliding_moe | **about 1.80 GB** | |

## Model level

| Component | Placement | Each chip holds | Collective | Per chip | vs 1x4 |
|---|---|---|---|---|---|
| embed_tokens [152576, 4096] bf16 | replicated | full table | none | 1.16 GiB | same |
| final norm | replicated | full | none | 8 KB | same |
| lm_head [152576, 4096] bf16 (untied) | vocab-sharded over 4 chips | rows 38144d .. 38144d + 38143 | all_gather logits, axis 1 then axis 0 (last tokens) | 0.29 GiB | gather on 2 axes |
| vision, audio, speech embeddings, MTP | skipped | none | none | 0 | same |
| layers 6-47 | skipped (subset) | none | none | 0 | same |

## Per-chip total (from the gate, GiB)

| Group | Per chip |
|---|---|
| activations (planner estimate, one 8192-token chunk) | 4.00 |
| routed experts, bfp8 (5 layers x 64 experts; counted as two halves, see below) | 3.98 + 3.99 |
| embedding (replicated) | 1.16 |
| LM head, vocab-sharded | 0.29 |
| attention weights (qkv bf16 + o_proj, 6 layers) | 0.26 |
| RoPE tables, dispatch tables, misc | 0.25 |
| KV state, sliding (4 layers, 2 heads, K 192 + V 128, full length) | 0.16 + 0.11 |
| contract KV copy (bfp8) | 0.20 |
| dense MLP (layer 0, bf16) | 0.09 |
| KV state, full (layers 0 and 5, 1 head, K 192 + V 128) | 0.04 + 0.03 |
| router (fp32) | 0.02 |
| **Total** | **14.59 of 27.20 budget** |

The expert weights appear twice because the checkpoint stores mxfp4 packed 2 values per byte. The `expert` placement
counts bfp8 bytes on the packed shape (half the values), and `extra_gb_per_chip` adds the other half. The weight bytes
per chip are the same as on 1x4, because every tensor is still split 4 ways or replicated. The KV state is 0.06 GiB
smaller because V is 128 wide.

## Collectives per layer (one 5120-token chunk, [S, 4096] bf16 = 42 MB)

| Layer type | Collectives | vs 1x4 |
|---|---|---|
| full_dense | 2 x all_reduce [S, 4096] over the mesh, each as axis 1 then axis 0 (2-chip stages) | 1x4 ran each as one 4-chip line all_reduce |
| sliding_moe, full_moe | attention: all_reduce [S, 4096] on axis 1 then axis 0. MoE: offset_cumsum all_gather of [1, 256] histograms on axis 0; dispatch and combine over axis 0 (each chip sends the (token, expert) pairs whose expert lives on the other chip of its column, on average about half of the ~4 of a token's 8 pairs that land in its group, so about 2 x S/2 x 4096 bf16 = 21 MB out and back); all_reduce [S/2, 4096] on axis 1 (21 MB); all_gather [S/2 -> S, 4096] on axis 0 | 1x4: local dispatch (no fabric) and one all_reduce [S, 4096] over 4 chips |

Per chunk: one embedding lookup, the final norm, and the LM-head matmul on the last tokens.

Activation estimate (4.0 GiB, sized for the 8192-token ladder chunk, the same as on 1x4). The worst-case dispatch buffer
does not shrink: a chip receives from both chips of its column, `max_dispatched_tokens_per_expert` = 2 x S/2 = S, and
the buffer capacity factor is 8 (all 8 experts of a token on one chip), so 8 x 8192 x 4096 bf16 = 0.54 GB. The combine
output is [S/2, 8, 4096] bf16 (0.27 GB). Then add the expert fp32 partials, the dense MLP intermediates (67 MB each),
Q/K/V and residual copies (67 MB each) and CCL scratch, and double the sum for fragmentation.

## Departures from the 1x4 prior and the reference plans, one reason each

- **Layer subset, not the whole model (carried over).** The full model's experts need about 75 GiB per chip, and layers 0-5
  already cover all three block types.
- **Attention and dense MLP stay TP=4 over the whole mesh, not TP=2 x SP=2.** There are 4 KV heads on the full layers
  and 4 TP ranks in the checkpoint, so one head per chip needs no duplication and no reorder, while sequence parallel on
  a causal chunked prefill would need a KV all_gather or ring SDPA every layer.
- **The all_reduce after o_proj and the dense MLP becomes `cluster_axis=None` (axis 1 then axis 0).** A 2x2 mesh has
  no 4-chip line, and TTNN's all_reduce already runs a non-line mesh as one reduce per axis, so no new CCL is needed.
- **The routed experts use the DeepSeek 2D dispatch (axis 0, 2 chips per group, fabric), not the prior's local 1-chip
  dispatch.** The prior needed a size-1 mesh axis for its local dispatch, and a 2x2 mesh has none. This layout is the
  one the dispatch op was built for (DeepSeek runs (2, 2) on FABRIC_2D). The forks behave as the source ops on a
  2-device axis, so no op change is needed. It also halves the MoE CCL: an all_reduce of [S/2, H] on one axis plus an
  all_gather, instead of an all_reduce of [S, H] over 4 chips.
- **Tokens are split by row inside the experts step (`mesh_partition`), not in the residual stream.** Attention needs
  every token on every chip (TP by head). A local split at the MoE entry and a gather at the exit keeps the residual
  replicated and every component boundary identical to 1x4, so the prior's goldens, tests and swap harness apply
  unchanged.
- **Router stays replicated on all S rows.** The component boundary stays [S, 256]. The experts step uses only its row
  half, so a later perf step may partition the router input first (about half the router time).
- **V at 128 on device, not padded to 192 (as the prior ended).** The `ttnn.bringup` SDPA fork takes a V narrower than K,
  which removes the pad rows, the pad cache columns and the o_proj zero columns.
- **Fallback if the fabric dispatch fails on this box:** add a default-off option to the dispatch/combine/offset_cumsum
  forks for a 1-chip dispatch group on a 2-device axis (agent rule 6). Then run EP=4 locally, as on 1x4, with one
  `all_reduce(cluster_axis=None)`.
- **Carried over unchanged, with the reasons in the 1x4 plan:** qkv sharded by the checkpoint's own TP ranks (per-rank
  quantization blocks); value scale folded into V rows; full-length sliding KV (the contract migrates the whole
  cache); qkv and dense MLP in bf16; experts bfp8 (owner rule); router fp32 with SFPU sigmoid + bias + `ttnn.topk`
  (moe_grouped_topk sorts on TF32 keys); no shared expert, so the routed partial is reduced alone.
