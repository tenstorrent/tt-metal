# MiMo-V2.6-Flash-RL prefill (layers 0-5): sharding on 4x Blackhole p150b (mesh 1x4, FABRIC_2D)

The machine-checked numbers come from `plan.yaml` (gate PL.1, `python -m models.demos.common.bringup.plan.check_plan`,
output in `results/plan_memory.json`). This page gives the reasoning. The shape follows `gemma4_a4b_d_p/bringup/plan.md`
and `ernie45_d_p/SHARDING.md`. Step-to-op mapping is in `components.yaml`.

Target: 56320 tokens in 5120-token chunks (ladder up to 8192-token chunks), 1 user, bf16 activations. Per-chip DRAM
32 GB, budget 27.2 GiB (15% headroom). Text decoder only; vision, audio and MTP are not loaded.

## Scope: layers 0-5, not 48

The spec runs layers 0-5 (layer 0 full_dense, 1-4 sliding_moe, 5 full_moe). The full model does not fit this box: the
routed experts of 47 MoE layers in bfp8 are 47 x 1.59 GiB = 75 GiB per chip, about 3x the budget. Layers 6-47 are
`skip` in `plan.yaml`; a subset result is never reported as a full-model result.

## Scheme

- Residual stream `[S, 4096]` is **replicated**. Norms, residual adds and the router run identically on every chip.
- **Attention: TP=4 by head.** The checkpoint stores the fused fp8 `qkv_proj` per TP rank (`tp_size: 4`, rank slab
  `[q_r; k_r; v_r]`), so chip r takes rank r's slab as stored: 16 Q heads plus 1 KV head (full) or 2 KV heads
  (sliding). o_proj is row-parallel, then one `all_reduce`.
- **Dense MLP (layer 0, SwiGLU 16384): TP=4.** gate/up column-parallel (4096 per chip), down row-parallel, one `all_reduce`.
- **Routed experts: EP=4.** Chip c holds experts 64c..64c+63 in bfp8. Every chip dispatches its (replicated) tokens
  locally to its own experts (ERNIE `moe_unified.py` scheme, dispatch group of 1 chip), then one `all_reduce`.
  No shared expert.
- **Embedding** replicated; **LM head** (untied) vocab-sharded, logits all-gathered for the last tokens only.
- Weights: fp8 qkv and dense MLP are dequantized (block scale folded) to **bf16**; o_proj, norms, sinks, embedding and
  LM head are bf16 as stored; router fp32; experts mxfp4 x e8m0 dequantized on the host and stored as **bfp8**.
- All CCLs are `ttnn.all_reduce` / `ttnn.all_gather` with `cluster_axis=1` on the FABRIC_2D mesh.

## full_dense layer (layer 0)

64 Q heads x 192, 4 KV heads, QK 192 / V 128, partial RoPE (first 64 dims, rotate-half, theta 1e7), no sink,
V x 0.707 before caching, scale 192^-0.5. MLP SwiGLU 16384.

| Component | Placement | Each chip holds | Collective after | Per chip |
|---|---|---|---|---|
| attn_norm (input_layernorm) | replicated | full [4096] | none | 8 KB |
| qkv_proj fused [4096 -> 13568], fp8 -> bf16 | column-parallel, TP rank slab as stored | 16 Q + 1 K + 1 V head (3392 rows; 3456 with V padded to 192) | none | 28 MB |
| partial RoPE on q, k (dims 0-63) | local | 16 Q + 1 K heads | none | 0 |
| KV cache (full length) | local, 1 KV head | K [1, 56320, 192], V [1, 56320, 192] (128 + 64 zero pad) | none | 43 MB |
| SDPA causal (chunk 0) / chunked SDPA (later chunks) | local, GQA 16:1 | 16 Q vs 1 KV head | none | 0 |
| o_proj [8192 -> 4096], bf16 | row-parallel | 2048 input columns (3072 with zero columns for the V pad) | **all_reduce** [S, 4096] | 17 MB (25 MB padded) |
| attn_residual | replicated | full | none | 0 |
| ffn_norm (post_attention_layernorm) | replicated | full | none | 8 KB |
| dense MLP 16384 (gate, up, down), fp8 -> bf16 | gate/up column, down row | 4096 of 16384 | **all_reduce** [S, 4096] | 101 MB |
| mlp_residual | replicated | full | none | 0 |
| **Layer total** | | | **2 all_reduces** | **about 0.19 GB** |

## sliding_moe layer (layers 1-4)

64 Q heads x 192, 8 KV heads, QK 192 / V 128, window 128, partial RoPE theta 1e4, per-head sink logit. MoE 256
experts, top-8, SwiGLU 2048, sigmoid noaux_tc router.

| Component | Placement | Each chip holds | Collective after | Per chip |
|---|---|---|---|---|
| attn_norm | replicated | full | none | 8 KB |
| qkv_proj fused [4096 -> 14848], fp8 -> bf16 | column-parallel, TP rank slab as stored | 16 Q + 2 K + 2 V heads (3712 rows; 3840 padded) | none | 31 MB |
| partial RoPE, V x 0.707 (folded into the V rows) | local | 16 Q + 2 K heads | none | 0 |
| KV cache (full length) | local, 2 KV heads | K and V [2, 56320, 192] | none | 87 MB |
| SDPA causal, window 128, attention_sink | local, GQA 8:1 | 16 Q vs 2 KV heads over the previous 128 cached positions + chunk | none | 0 |
| attention_sink_bias [64] | sharded by head | 16 values (pre-divided by the scale) | none | 32 B |
| o_proj | row-parallel | 2048 input columns | **all_reduce** [S, 4096] | 17 MB |
| attn_residual, ffn_norm | replicated | full | none | 8 KB |
| router [4096 -> 256] + correction bias, fp32 | replicated, same routing on every chip | full | none | 4 MB |
| routed experts 256 x SwiGLU 2048, bfp8 | expert-parallel | experts 64c..64c+63 | **all_reduce** [S, 4096] | 1.71 GB |
| ffn_residual | replicated | full | none | 0 |
| **Layer total** | | | **2 all_reduces** | **about 1.85 GB** |

## full_moe layer (layer 5)

Attention as layer 0 (4 KV heads, one per chip, theta 1e7, no sink); router and experts as the sliding layers.

| Component | Placement | Each chip holds | Collective after | Per chip |
|---|---|---|---|---|
| attn_norm, attn_residual, ffn_norm | replicated | full | none | 16 KB |
| qkv_proj, RoPE, KV cache, chunked SDPA, o_proj | as layer 0 | 16 Q + 1 KV head | **all_reduce** after o_proj | 88 MB |
| router, routed experts, ffn_residual | as the sliding layers | experts 64c..64c+63 | **all_reduce** after experts | 1.72 GB |
| **Layer total** | | | **2 all_reduces** | **about 1.80 GB** |

## Model level

| Component | Placement | Each chip holds | Collective | Per chip |
|---|---|---|---|---|
| embed_tokens [152576, 4096] bf16 | replicated | full table | none | 1.16 GiB |
| final norm | replicated | full | none | 8 KB |
| lm_head [152576, 4096] bf16 (untied) | vocab-sharded | 38144 rows | all_gather logits (last tokens) | 0.29 GiB |
| vision, audio, speech embeddings, MTP | skipped | none | none | 0 |
| layers 6-47 | skipped (subset) | none | none | 0 |

## Per-chip total (from the gate, GiB)

| Group | Per chip |
|---|---|
| activations (planner estimate, one 8192-token chunk) | 4.00 |
| routed experts, bfp8 (5 layers x 64 experts; counted as two halves, see below) | 3.98 + 3.99 |
| embedding (replicated) | 1.16 |
| LM head, vocab-sharded | 0.29 |
| attention weights (qkv bf16 + o_proj, 6 layers) | 0.26 |
| RoPE tables, dispatch tables, misc | 0.25 |
| KV state, sliding (4 layers, 2 heads, K + padded V, full length) | 0.32 |
| contract KV copy (bfp8) | 0.20 |
| dense MLP (layer 0, bf16) | 0.09 |
| KV state, full (layers 0 and 5, 1 head, K + padded V) | 0.08 |
| router (fp32) | 0.02 |
| **Total** | **14.65 of 27.20 budget** |

The expert weights appear twice because the checkpoint stores mxfp4 packed 2 values per byte: the `expert` placement
counts bfp8 bytes on the packed shape (half the values) and `extra_gb_per_chip` adds the other half.

Per layer: 2 `all_reduce` of [5120, 4096] bf16 (42 MB each per chunk). No all-to-all: the residual is replicated and
every chip routes on its own. Per chunk: one embedding lookup, the final norm, and the LM-head matmul on the last tokens.

Activation estimate (4.0 GiB, sized for the 8192-token ladder chunk): MoE dispatch and combine buffers in the worst case
(all 8 experts of a token on one chip) are 8 x 8192 x 4096 bf16, 0.54 GB each; expert gate/up intermediates up to
8 x 8192 x 2048 bf16, 0.27 GB each; dense MLP intermediates 8192 x 4096 per chip, 67 MB each; Q/K/V (16 x 8192 x 192)
and residual copies (67 MB each); CCL scratch; doubled for fragmentation.

## Departures from the reference plans (Gemma-4 A4B, ERNIE SHARDING.md), one reason each

- **Layer subset, not the whole model.** Only layers 0-5 are placed because the full model's experts need about
  75 GiB per chip, and layers 0-5 already cover all three block types.
- **qkv sharded by the checkpoint's own TP ranks.** The fused projection is stored per rank with per-rank 128-row
  quantization blocks (known issue "Fused fp8 qkv stored per TP rank"), and TP=4 on 4 chips puts rank r on chip r, so
  no reordering across chips is needed.
- **One KV head per chip on full layers, two on sliding layers, none duplicated.** There are 4 and 8 KV heads for 4
  chips, so unlike Gemma-4's global layers no head has to be held twice.
- **V padded from 128 to 192 on device.** Plain and chunked SDPA require V head dim == QK head dim (only the MLA path
  differs, and it reads V from K), so V rows are zero-padded at load, the cache holds 192, and o_proj gets zero columns;
  the state read-back returns the first 128 columns.
- **attention_value_scale folded into the V weight rows.** It is a linear scale applied before caching, so folding it
  removes one op per layer and keeps the cached V equal to the reference's scaled V.
- **Sliding KV kept at full length, not 128.** The prefill contract hands the whole cache to decode; the window only
  limits what SDPA reads (previous 128 positions from the cache, concatenated before a windowed SDPA with sinks).
- **qkv and dense MLP in bf16, not fp8/bfp8.** They are small (0.35 GiB per chip for 6 layers), so accuracy comes first;
  bfp8 is a possible later perf step.
- **Experts in bfp8, not bf16 (Gemma-4) or bfp4 (DeepSeek default).** The owner requires bfp8; bf16 would also fit
  (about 21.7 GiB total) but doubles expert DRAM traffic. bfp8 (7-bit mantissa per 16-value group) holds the 3-bit
  e2m1 values closely; it is not bit-exact because its groups run along the output dim, not along the mxfp4 blocks.
- **Router in fp32 through DeepSeek's `moe_grouped_topk`.** The router is DeepSeek-style (sigmoid, correction bias,
  noaux_tc) with one group and no routing scale (`route_scale` 1.0, not DeepSeek's 2.5); fp32 keeps near-tie top-8
  selections stable at 4 MB per layer.
- **Two all_reduces per layer, as ERNIE.** There is no shared expert and no post-MoE norm, so the routed partial is
  reduced alone and attention needs its own reduce before the (nonlinear) ffn_norm.
