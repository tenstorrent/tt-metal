# MiMo-V2.6-Flash-RL prefill (layers 0-5): context parallelism CP=4 on 4x Blackhole p150b (mesh 1x4, FABRIC_2D)

The machine-checked numbers come from `plan.yaml` (gate PL.1, `python -m models.demos.common.bringup.plan.check_plan`,
output in `results/plan_memory.json`). This page gives the reasoning. The step-to-op mapping is in `components.yaml`.
The plan starts from the 1x4 TP=4 prior (`models/demos/mimo_v2_6_d_p/bringup/plan.md` and the modules it ended with in
`models/demos/mimo_v2_6_d_p/tt/`) and the 2x2 run (`models/demos/mimo_v2_6_d_p_2x2`). Each table has a "vs 1x4 prior"
column that says what changes and what carries over.

Target: 56320 tokens in 5120-token chunks (1280 rows per chip), ladder chunks 2048 and 8192 (512 and 2048 per chip),
1 user, bf16 activations. Per-chip DRAM 32 GB, budget 27.2 GiB (15% headroom). Text decoder only: vision, audio and
MTP (`model.mtp.*`) are not loaded.

## Scope: layers 0-5, not 48

Unchanged from the prior. The spec runs layers 0-5 (layer 0 full_dense, 1-4 sliding_moe, 5 full_moe). The full model
does not fit this box: the routed experts of 47 MoE layers in bfp8 are 47 x 1.59 GiB = 75 GiB per chip. Layers 6-47
are `skip` in `plan.yaml`. A subset result is never reported as a full-model result.

## Scheme (owner-approved, implemented as given)

- **Residual stream split by sequence (CP=4).** Each chunk of S tokens is split into 4 contiguous slices of S/4. Chip c
  holds rows `[c S/4, (c+1) S/4)` of the chunk, which are global positions `start + c S/4 + j`. The residual never
  leaves its chip. Norms, residual adds, the dense MLP, the router and the embedding are per-row, so they run locally
  with no CCL.
- **KV cache split by sequence, chunk-major.** Chip c stores its slice of every chunk: local row `k S/4 + j` holds
  global position `k S + c S/4 + j`. This is the `gemma4_d_p/tt/attention/ring_prefill.py` layout, written by
  `update_padded_kv_cache(cluster_axis=1)`. Every KV head is on every chip.
- **Attention TP=1.** Every chip holds all 64 Q heads and all KV heads, with the whole qkv_proj and o_proj. There is no
  all_reduce after o_proj. Full attention runs `ttnn.transformer.ring_joint_scaled_dot_product_attention`
  (causal, chunked prefill, cluster_axis 1): the ring streams the other chips' cached K/V through each chip's
  online softmax. Sliding attention (window 128) runs the same op with `sliding_window_size=128` and the per-head
  sink. Its compact halo exchange brings in the previous 128 positions across slice boundaries: chip c reads the tail
  of chip c-1's slice, and chip 0 reads chip 3's slice of the previous chunk. This is the GPT-OSS SP4 path
  (`gpt_oss_d_p/tt/attention/dense_sp.py`).
- **Dense MLP (layer 0) TP=1**, local to the slice.
- **Routed experts EP=4, TP=1.** Chip c stores experts 64c..64c+63 as complete experts. Each chip dispatches its own
  S/4 tokens over the whole 1x4 mesh: one dispatch group of 4 chips on axis 1, over the fabric, through the
  `ttnn.bringup.dispatch` / `combine` / `offset_cumsum` forks that the 2x2 run used for a multi-chip axis. The results
  are combined back and weighted-summed per token, so the output is the chip's own [S/4, 4096] rows with no reduce.
- **Router** replicated; each chip routes its own slice.
- **Embedding** replicated; each chip embeds its own S/4 ids. The engine input `[sp=4, 1, chunk/4]` is already split
  this way. **LM head** vocab-sharded; the last tokens are gathered to every chip first.
- Weights: fp8 qkv and dense MLP are dequantized per 128x128 block to **bf16**. o_proj, norms, sinks, embedding and LM
  head are bf16 as stored. The router is fp32. Experts are mxfp4 x e8m0, dequantized on the host and stored as
  **bfp8** (never bfp4). Every matmul runs at HiFi4. The routed experts use `unified_routed_expert_moe`
  `high_precision=True`. Sliding SDPA keeps the 1x4 preset "S": HiFi4, fp32 dest off (the streaming kernel, which the
  ring sink path also requires), exact exp, q128/k128.
- All CCLs are on FABRIC_2D, axis 1, Topology.Linear.

## full_dense layer (layer 0)

64 Q heads x 192, 4 KV heads, QK 192 / V 128, partial RoPE (first 64 dims, rotate-half, theta 1e7), no sink, V x 0.707
folded into the V rows, scale 192^-0.5. MLP SwiGLU 16384. Sizes are per chip.

| Component | Placement | Each chip holds | Collective after | Per chip | vs 1x4 prior |
|---|---|---|---|---|---|
| attn_norm | replicated weight, local rows | full [4096]; runs on S/4 rows | none | 8 KB | same module, S/4 rows |
| qkv_proj [4096 -> 13568], fp8 -> bf16, as Q [64 x 192] + KV [4 x (192 + 192 padded)] | replicated (TP=1): all four stored rank slabs, dequantized per slab and reassembled by head | all 64 Q + 4 K + 4 V heads | none | 111 MB (+2 MB V pad) | was 1/4 (rank slab d); now whole |
| partial RoPE on q, k (dims 0-63) | local rows | CP-permuted cos/sin (chip c's chunk-major rows) | none | small | tables permuted so every chip slices the same local row |
| KV cache (full length, chunk-major CP slice) | sequence-split | K, V [4 heads, 14080, 192] bf16 (V 128 + 64 zero pad) | none | 43 MB (21.6 MB each) | was 1 head x 56320; now 4 heads x 56320/4 |
| ring_joint SDPA, causal, chunked | ring over axis 1 | 64 Q heads on S/4 rows vs the whole prefix | **ring K/V stream** (internal) | gather buffers 173 MB (shared) | was local chunked SDPA, GQA 16:1 |
| o_proj [8192 -> 4096], bf16 | replicated | whole (+ zero input columns for the V pad, or slice SDPA out to 128) | **none** | 67 MB (+34 MB) | all_reduce removed |
| attn_residual, ffn_norm | local rows | full | none | 8 KB | same |
| dense MLP 16384 (gate, up, down), fp8 -> bf16 | replicated (TP=1) | whole | **none** | 403 MB | was 1/4 + all_reduce |
| mlp_residual | local rows | | none | 0 | same |
| **Layer total** | | | **no all_reduce; ring K/V inside SDPA** | **about 0.66 GB** | was 0.19 GB, 2 all_reduces |

## sliding_moe layer (layers 1-4)

64 Q heads x 192, 8 KV heads, QK 192 / V 128, window 128, partial RoPE theta 1e4, per-head sink. MoE 256 experts,
top-8, SwiGLU 2048, sigmoid noaux_tc router.

| Component | Placement | Each chip holds | Collective after | Per chip | vs 1x4 prior |
|---|---|---|---|---|---|
| attn_norm | local rows | full | none | 8 KB | same module, S/4 rows |
| qkv_proj [4096 -> 14848], fp8 -> bf16 | replicated | 64 Q + 8 K + 8 V heads (V padded to 192), true scale folded into Q rows (SDPA scale 2^-4) | none | 122 MB (+4 MB) | was rank slab (2 KV heads) |
| partial RoPE, V x 0.707 (in the V rows) | local rows | | none | 0 | permuted tables |
| KV cache (full length, CP slice) | sequence-split | K, V [8, 14080, 192] **bfp8** | none | 46 MB | was bf16 [2, 56320, 192]; bfp8 is required by the ring sliding path |
| ring_joint SDPA, causal, window 128, sink [1, 64, 1, 1] | ring halo over axis 1 | 64 Q heads, S/4 rows; halo = predecessor's last 128 rows | **halo exchange** (internal, < 1 MB) | halo buffers < 1 MB | was a local read-back of 128 cached rows |
| attention_sink_bias [64] | replicated | all 64 (pre-divided by 2^-4) | none | 128 B | was 16 per chip |
| o_proj | replicated | whole | **none** | 67 MB (+34 MB) | all_reduce removed |
| attn_residual, ffn_norm | local rows | | none | 8 KB | same |
| router [4096 -> 256] + correction bias, fp32 | replicated weights | full; routes its S/4 rows | none | 4 MB | was all S rows on every chip |
| routed experts 256 x SwiGLU 2048, bfp8: masked_bincount -> offset_cumsum -> dispatch -> unified (high_precision) -> combine -> post_combine_reduce | expert-parallel, one dispatch group of 4 on axis 1 | experts 64c..64c+63 (complete) | **offset_cumsum all_gather [1, 256]; dispatch + combine over the fabric**; no reduce | 1.71 GB | was local dispatch of replicated tokens + all_reduce |
| ffn_residual | local rows | | none | 0 | same |
| **Layer total** | | | **dispatch/combine + halo** | **about 1.99 GB** | was 1.85 GB, 2 all_reduces |

## full_moe layer (layer 5)

Attention as in layer 0 (4 KV heads, theta 1e7, no sink); router and experts as in the sliding layers. Layer 5's MoE
input has outlier channels (|x| up to 131), so the expert input stays bf16 and the unified kernel runs
`high_precision=True` at HiFi4 (known issues), as on 1x4.

| Component | Placement | Each chip holds | Collective after | Per chip | vs 1x4 prior |
|---|---|---|---|---|---|
| attn_norm, attn_residual, ffn_norm | local rows | full | none | 16 KB | same |
| qkv_proj, RoPE, KV cache, ring SDPA, o_proj | as layer 0 | all heads; K/V CP slice | ring K/V (internal); no all_reduce | 0.26 GB | TP=1 + ring |
| router, routed experts, ffn_residual | as the sliding layers | experts 64c..64c+63 | dispatch/combine over axis 1 | 1.72 GB | as the sliding layers |
| **Layer total** | | | as above | **about 1.98 GB** | was 1.80 GB |

## Model level

| Component | Placement | Each chip holds | Collective | Per chip | vs 1x4 prior |
|---|---|---|---|---|---|
| embed_tokens [152576, 4096] bf16 | replicated | full table; looks up its S/4 ids | none | 1.16 GiB | ids already CP-split by the engine |
| final norm | local rows | full | none | 8 KB | same |
| lm_head [152576, 4096] bf16 (untied) | vocab-sharded | rows 38144c .. 38144c + 38143 | all_gather [32, 4096] last rows (axis 1), then all_gather logits (axis 1, dim -1) | 0.29 GiB | extra tiny gather: the last token is on one chip |
| vision, audio, speech embeddings, MTP | skipped | none | none | 0 | MTP tensors (`model.mtp.*`) are in this checkpoint map and are now skipped explicitly |
| layers 6-47 | skipped (subset) | none | none | 0 | same |

## Per-chip total (from the gate, GiB)

| Group | Per chip |
|---|---|
| activations (planner estimate, one 8192-token chunk = 2048 rows per chip) | 4.00 |
| routed experts, bfp8 (5 layers x 64 experts; counted as two halves, see below) | 3.98 + 3.99 |
| embedding (replicated) | 1.16 |
| attention weights, replicated (qkv bf16 + o_proj, 6 layers) | 1.04 |
| dense MLP, replicated (layer 0, bf16) | 0.38 |
| LM head, vocab-sharded | 0.29 |
| RoPE tables, dispatch tables, misc | 0.25 |
| attention V-pad rows and o_proj zero columns | 0.22 |
| contract KV copy (bfp8) | 0.20 |
| ring SDPA gather buffers (full layers) | 0.17 |
| KV state, sliding (4 layers, 8 heads, K + padded V, bfp8, S/4) | 0.09 + 0.09 |
| KV state, full (layers 0 and 5, 4 heads, K + padded V, bf16, S/4) | 0.04 + 0.04 |
| router (fp32) | 0.02 |
| **Total** | **15.95 of 27.20 budget** |

The expert weights appear twice because the checkpoint stores mxfp4 packed 2 values per byte. The `expert` placement
counts bfp8 bytes on the packed shape (half the values), and `extra_gb_per_chip` adds the other half (known issue
"Plan gate counts packed checkpoint shapes"). Compared with the prior's 14.65 GiB: attention and dense-MLP weights
are now whole on every chip (+1.07 GiB), and the KV state is a quarter of the sequence with all heads (about the same
bytes, less for the bfp8 sliding cache).

## Collectives per layer (one 5120-token chunk, 1280 rows per chip)

| Layer type | Collectives | vs 1x4 prior |
|---|---|---|
| full_dense, full_moe (attention) | ring K/V inside `ring_joint_scaled_dot_product_attention`. Each chip receives the other 3 chips' cached K/V for the prefix: at the last chunk 3 x 4 heads x 14080 x 192 x 2 (K + V) bf16 = 130 MB per layer, half that on average over the prompt | 1x4: all_reduce [S, 4096] bf16 after o_proj (42 MB) |
| sliding_moe (attention) | halo exchange inside the ring op: [8, 128, 192] K + V bfp8 from the predecessor chip, < 1 MB | 1x4: all_reduce after o_proj (42 MB) |
| full_dense (MLP) | none | 1x4: all_reduce (42 MB) |
| sliding_moe, full_moe (experts) | offset_cumsum all_gather of [1, 256] histograms over axis 1. Dispatch sends each (token, expert) pair whose expert is on another chip: about 3/4 of 8 x 1280 pairs x 4096 bf16 = 63 MB out, then 63 MB back through combine | 1x4: local dispatch, then all_reduce (42 MB) |

Per chunk: one embedding lookup per slice, the final norm per slice, one [32, 4096] gather and the LM-head matmul on
the last tokens. The engine ids arrive split by slice, so no gather of ids is needed.

Activation estimate (4.0 GiB, sized for the 8192-token ladder chunk). The dispatch buffer does not shrink: a chip
receives from all 4 chips, `max_dispatched_tokens_per_expert` = 4 x S/4 = S, and the capacity factor is 8, so
8 x 8192 x 4096 bf16 = 0.54 GB. The unified kernel's partials and intermediates are up to 0.27 GB, and the combine
output [S/4, 8, 4096] bf16 is 0.13 GB. The dense MLP intermediates [S/4, 16384] bf16 are 67 MB each. Q and the SDPA
output [64, S/4, 192] bf16 are 50 MB each. The sum is then doubled for fragmentation and CCL scratch.

## Departures from the prior and the reference plans, one reason each

- **CP=4 with TP=1 attention instead of TP=4 by head (owner-approved scheme).** The residual, KV cache and every per-row
  step stay on the slice's chip. All 4 all_reduces per layer are gone. The cross-chip traffic is the ring K/V stream,
  the sliding halo and the MoE dispatch/combine. The cost is replicated attention and dense-MLP weights (+1.07 GiB per
  chip), which fits the budget easily.
- **qkv reassembled from the stored TP-rank slabs at load.** The checkpoint stores `[q_r; k_r; v_r]` per rank, each
  quantized on its own 128-row blocks (known issue "Fused fp8 qkv stored per TP rank"). Each slab is dequantized
  separately and the heads are concatenated in rank order: Q heads 16r..16r+15, KV head r (full) or 2r, 2r+1
  (sliding). This gives the whole Q / K / V that TP=1 needs.
- **Ring SDPA (`ring_joint_scaled_dot_product_attention`) instead of local chunked SDPA.** Full attention must see the
  other chips' K/V. The ring op streams them, so no all_gather of the whole prefix has to sit in DRAM. The same op's
  compact halo gives sliding attention the 128 positions before each slice with its sink, which is exactly the GPT-OSS
  SP4 path already in the repo. No new op is needed.
- **V padded 128 -> 192 again (the prior's `MIMO_V_PAD=1` path).** The ring op's tensor-V mode requires V head dim ==
  QK head dim (`ring_joint_sdpa_device_operation.cpp`). The `ttnn.bringup` sdpa fork that took a narrow V covers only
  the non-ring ops. The pad costs 0.22 GiB of weights plus the cache columns. A ring_joint fork with a narrow-V option
  (default off) is a later perf step under agent rule 6.
- **Sliding KV cache in bfp8, not bf16.** The ring sliding path requires BFP8_B K/V (BF16 Q). This matches the engine's
  bfp8 contract format. Risk: the sink-dominated sliding layers are sensitive to QK error (known issues). If
  C.sliding_moe.attention fails on accuracy, the fix is a `ttnn.bringup` fork of ring_joint that accepts BF16 K/V on
  the sliding path behind a default-off option (rule 6). Full layers keep bf16 (the full-causal ring path accepts it).
- **The MiMo shape on the ring sliding path is unexercised.** The op documents the GPT-OSS specialization (8Q:1KV,
  D64). MiMo's 64Q:8KV at D192, q128/k128, SP4, B1 passes its validation, but no test covers it. The implement step
  probes it first at s4096 (512 rows per chip). A kernel-level failure there also goes through a fork, not an edit of
  the op.
- **CP-permuted RoPE tables.** Chip c's rows are positions `k S + c S/4 + j`. At load each chip gets cos/sin for its own
  chunk-major rows (per chunk size), so the per-chunk slice starts at the same local row on every chip. No per-chip
  host scalar is needed and the forward does no host work.
- **One MoE dispatch group of 4 chips on axis 1 (2x2 used 2 groups of 2 on axis 0; the prior dispatched locally).** The
  tokens are already split by sequence, so the dispatch has to cross chips. With a single group, combine +
  post_combine_reduce returns each chip's own rows fully summed, so no all_reduce or all_gather follows. The forks
  keep the source op's behaviour on a multi-device axis, and DeepSeek runs groups of 4 to 8 chips. Axis 1 with 4
  devices is new for these forks in this repo. Any fix goes behind a default-off fork option.
- **LM head gathers the last rows first.** The last token lives on one chip (chip 3 for a full chunk). A [32, 4096]
  all_gather is far cheaper than replicating the 1.16 GiB LM head.
- **Carried over unchanged, with the reasons in the 1x4 plan:** layer subset; value scale folded into V rows; true
  scale folded into Q with the sink pre-divided by the power-of-two SDPA scale (exact in bf16); full-length KV (the
  contract migrates the whole cache); qkv and dense MLP in bf16; experts bfp8 (owner rule); router fp32 with SFPU
  sigmoid + bias + `ttnn.topk` (moe_grouped_topk sorts on TF32 keys); no shared expert.

## Tasks

The ledger (`tasks.yaml`) is unchanged: the block graphs and component set are the same as the prior's. The
component and swap tests cross the host boundary with full [S, ...] goldens. The implement step splits the input
rows by chip on the way in (`ShardTensorToMesh` on the sequence dim) and concatenates them on the way out. It loads
and reads the KV state in the chunk-major ring layout. That code is harness-boundary work, not forward-path work.
