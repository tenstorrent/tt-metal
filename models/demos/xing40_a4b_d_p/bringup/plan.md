# Xing4.0-29B-A4B prefill (all 40 layers): sharding on 8x Blackhole p150b (LoudBox, mesh 4x2, FABRIC_2D)

The gate (PL.1, `python -m models.demos.common.bringup.plan.check_plan`, output in `results/plan_memory.json`) computes
the numbers from `plan.yaml` and the real checkpoint shapes. This page gives the reasoning. `components.yaml` maps each
step to its ops.

Target: 56320 tokens in 5120-token chunks (the ladder also runs 2048- and 8192-token chunks), 1 user, bf16
activations, fp32 mHC streams. Each chip has 32 GB of DRAM; the budget is 27.2 GiB (15% headroom). Text decoder only:
the MTP layer (`model.layers.40.*`) is not loaded.

## Scheme

Mesh coordinates are (row r, col c). The layout is **SP=4 over rows (axis 0) x TP=2 over columns (axis 1)** (owner
rule). It is the Kimi K2.7 layout of `deepseek_v3_d_p` (dense ttMLA chunked path with `ring_mla`, TtMoe 2D dispatch)
with 2 columns instead of 4. `hy4_preview_d_p` (SP2 x TP2, MLA + hyper-connection gates) is the closest working code.

- **Activations.** Chip (r, c) holds chunk rows `[r*S/4, (r+1)*S/4)` (1280 of 5120) and hidden columns
  `[1792c, 1792(c+1))`. The engine's uint32 input arrives as `[sp=4, 1, chunk/4]` over axis 0, so the ids need no
  gather: each chip embeds its row's ids into its own hidden columns.
- **Residual.** The 4 mHC streams stay resident in fp32 as `[S/4, 4 x 1792]` per chip: each stream is split by row
  (like the tokens) and by column (like Kimi's TP-sharded hidden), packed stream-major along the last dim. No chip
  holds another row's tokens and nothing is replicated over all 8 chips. A copy is 37 MB at 1280 rows.
- **mHC.** The coefficient projection reduces over all 4 x 3584 values of a token. Each chip computes its partial
  mixes (its fn^T columns) and its partial sum of squares, then one `[S/4, 32]` fp32 all_reduce over axis 1 completes
  both (RMS is a per-token scalar, so it commutes past the linear; hy4 `TtHcGates`). The sigmoid gates and the 20-step
  Sinkhorn on the 4x4 comb then run redundantly on both chips of the row. Collapse and the residual mix are per token
  and per hidden column, so they are local.
- **Attention: TP=2 by head, SP=4 by sequence.** 32 heads over 2 columns is 16 heads per chip, the same per-chip count
  as Kimi (64 over TP=4), which is what ttMLA's `ring_mla` path runs. Weights are split over the columns and replicated
  over the rows. q_a / kv_a are K-split (the norm output is column-split) and all-reduced over axis 1; o_proj is
  row-parallel and reduce-scattered over axis 1, which returns the residual's column split.
- **MLA latent cache.** ttMLA's chunked layout: block-cyclic over the 4 rows with period = the chunk. Row r holds rows
  `[k*chunk + r*chunk/4, +chunk/4)` of every chunk k at local rows `[k*chunk/4, +chunk/4)`, so each row holds a quarter
  of the prefix, `[1, 1, 14080, 576]` bf16 per layer (16 MB). Both columns of a row compute the same all-reduced latent
  and write their own copy (the dense `ring_mla` reads a TP-replicated cache; ttMLA's TP dedup is wired only for the
  sparse path, and the memory is there). `update_padded_kv_cache` writes the chunk at the local offset derived on
  device from the chunk start.
- **Gathering the latent.** `ring_mla` (cluster_axis 0, Linear topology on FABRIC_2D, `per_axis_topology`) gathers
  the prefix around each column's 4 chips into one shared `[1, 1, 56320, 576]` scratch in natural order and runs
  causal flash attention for the chip's 1280 query rows x 16 heads against `logical_n = start + chunk` keys. The two
  columns run two independent rings. At the last 56k chunk each chip receives 3 x 14080 x 576 bf16 = 49 MB per layer.
- **Routed experts: EP=8 with the DeepSeek 2D dispatch.** Each column is one dispatch group of 4 chips (axis 0) with
  32 experts; chip (r, c) holds experts `32c + 8r .. +7` (the `deepseek_v3_d_p/tt/moe/README.md` 4x2 example). Each chip
  dispatches its row's 1280 tokens up and down its column, only to the chips that hold one of the token's in-group
  experts. After the experts, combine and the weighted sum, each column holds the partial output of its 32 experts, and
  a reduce_scatter over axis 1 adds the two groups and returns the column split.
- **Dense MLP (layers 0-1) and shared expert: TP=2 by intermediate.** gate/up are column-parallel and down is
  row-parallel. The input is the row's full hidden (ffn_norm gathers its bf16 output once over axis 1), and a
  reduce_scatter over axis 1 closes each one.
- **Router.** Replicated fp32; both chips of a row compute the same routing from the gathered ffn_norm.
- **Weight dtypes.** Attention, the dense MLP, the shared expert, the embedding and the LM head are bf16 as stored.
  The router and mHC are fp32. The routed experts are bf16 in the checkpoint and **bfp8** on the device (the fused
  expert op's tested dtype, never bfp4). bf16 experts would also fit (6.6 GiB instead of 3.3), kept as the fallback if
  the expert accuracy needs it. Every matmul runs at HiFi4 with fp32 accumulation, `ring_mla` included (owner rule).
- **Code.** The Xing modules go in `models/demos/xing40_a4b_d_p/tt`. They copy or wrap `deepseek_v3_d_p`,
  `hy4_preview_d_p` and `glm53_flash_d_p` code, which stays read-only. Op changes go through the forks in
  `ttnn/ttnn/bringup` (dispatch / combine / offset_cumsum, `unified_routed_expert_moe`, `rms_norm`, and a new fork of
  `mhc_split_sinkhorn` for Xing's Sinkhorn, see below).

## Causal balance over the 4 rows

The chunk is split into 4 contiguous quarters, one per row (Kimi's chunked layout; `ring_mla` asserts
`is_balanced=False` on the chunked path). Row r's queries are the positions `start + 1280r .. +1279`, and they attend
to `start + 1280r + 1 .. start + 1280(r + 1)` keys each.

- **Last chunk (s56320, start 51200).** Attention work per row is 1280 x (51200 + 1280r + 640.5) query-key pairs:
  66.4 M (row 0), 68.0 M, 69.6 M, 71.3 M (row 3). The slowest row does 1.074x the fastest and 1.036x the mean, so the
  rows are 97% busy on attention. The long prefix, which every row reads in full, dominates the triangle of the chunk.
- **Whole s56320 run (11 chunks).** Summing the slowest row per chunk against the mean gives 1.068x, 94% efficiency.
  Only the first chunk is badly skewed (640 vs 4480 keys per query on average, 1:7), and it is 1% of the attention work.
- **Why not zigzag.** A zigzag split (row r takes the 640-row blocks r and 7 - r) would balance each chunk exactly,
  but chunked `ring_mla` rejects `is_balanced=True`, the rotary / cache-write ops derive a contiguous per-row offset,
  and the engine's ids arrive contiguous per row. It would buy at most 6% of the attention time. Not planned; a perf
  item if the profile shows attention bound by the last row.
- **Everything else is balanced.** Every other step is per token (1280 rows on every chip). The MoE load depends on
  routing, not on position.

## dense layer (layers 0, 1)

MLA with 32 heads, q_lora 768, kv_lora 512, qk 128 + 64 (interleaved RoPE, YaRN factor 64), v 128, dense causal; a
dense SwiGLU 9216. Two mHC steps (4 streams) around attention and MLP. Collective sizes at chunk 5120 (1280 rows per
chip).

| Component | Placement | Each chip holds | Collective after | Per chip |
|---|---|---|---|---|
| attn_hc (hc_fn [24, 14336], base, scale; sigmoid gates, clamped comb, 20-step Sinkhorn) | fn split by column | fn [24, 7168] fp32 | **all_reduce axis 1**, [S/4, 32] fp32 (160 KB) | 0.7 MB |
| attn_collapse (sum_n pre_n x stream_n) | local | 4 x [S/4, 1792] fp32 | none | 0 |
| attn_norm (input_layernorm, distributed RMSNorm) | weight split by column | [1792] | all_gather of [S/4, 32] stats, axis 1 | 4 KB |
| q_a: q_a_proj [3584 -> 768] + q_a_layernorm | K-split over columns | [1792, 768] bf16 | **all_reduce axis 1** [S/4, 768] fp32 (3.9 MB) | 2.8 MB |
| attention: kv_a_proj_with_mqa [3584 -> 576] + kv_a_layernorm, RoPE on k_rope | K-split over columns | [1792, 576] bf16 | **all_reduce axis 1** [S/4, 576] fp32 (2.9 MB) | 2.1 MB |
| attention: MLA latent cache [56320, 576] bf16 | block-cyclic over the 4 rows, replicated over columns | a quarter of the rows | none (written locally) | 16.2 MB |
| attention: q_b_proj [768 -> 32 x 192], kv_b_proj (-> wkv_b1 128 -> 512, wkv_b2 512 -> 128) | split by head | 16 heads | none | 8.9 MB |
| attention: ring_mla (absorbed 576 / 512, causal, scale 192^-0.5 x mscale^2) | local queries, ring over axis 0 | 16 heads, 1280 query rows | **KV ring gather over axis 0** (up to 49 MB at 56k) | shared 65 MB scratch |
| attention: o_proj [4096 -> 3584] | row-parallel | [2048, 3584] bf16 | **reduce_scatter axis 1** -> [S/4, 1792] (18 MB fp32 in) | 14.7 MB |
| attn_residual (post_i y + sum_j comb[i, j] in_j) | local | fp32 streams | none | 0 |
| ffn_hc, ffn_collapse | as attn_hc, attn_collapse | fn [24, 7168] fp32 | **all_reduce axis 1**, [S/4, 32] fp32 | 0.7 MB |
| ffn_norm (post_attention_layernorm, distributed) + gather | weight split by column | [1792] | stats all_gather; **all_gather axis 1** of the bf16 output -> [S/4, 3584] (4.6 MB in) | 4 KB |
| mlp: SwiGLU 9216 (gate, up, down), fp32 intermediates | gate/up column, down row | intermediate 4608c .. +4607 | **reduce_scatter axis 1** -> [S/4, 1792] (18 MB fp32 in) | 99 MB |
| ffn_residual | local | fp32 streams | none | 0 |
| **Layer total** | | | **2 tiny all_reduces, 2 stats gathers, 2 latent all_reduces, KV ring, 1 all_gather, 2 reduce_scatters** | **about 145 MB** |

## moe layer (layers 2-39)

Attention and mHC are as in the dense layer. The MoE has 64 routed experts, top-4, SwiGLU 1024, and 1 shared expert
(SwiGLU 1024). The router is sigmoid + correction bias, one group, normalised top-4, scaled by 2.0.

| Component | Placement | Each chip holds | Collective after | Per chip |
|---|---|---|---|---|
| attn_hc .. attn_residual | as dense | as dense | as dense | 45.4 MB |
| ffn_hc, ffn_collapse, ffn_norm | as dense | as dense | all_reduce [S/4, 32]; stats gather; all_gather ffn_norm axis 1 | 0.7 MB |
| router [3584 -> 64] + correction bias, fp32 (linear, sigmoid, add, topk 4, gather, renorm, x 2.0) | replicated | full [64, 3584] fp32 | none (both chips of a row compute the same routing) | 0.9 MB |
| experts: masked_bincount, offset_cumsum, dispatch, unified_routed_expert_moe (SwiGLU, high_precision), combine, post_combine_reduce | expert-parallel, dispatch group = column (4 chips, axis 0) | experts 32c + 8r .. +7, bfp8 | offset_cumsum histograms over axis 0; dispatch and combine over axis 0 (about 1280 x 1.9 x 7 KB = 17 MB each way, 3/4 off-chip); **reduce_scatter axis 1** (18 MB fp32 in) | 93.6 MB |
| shared_expert: SwiGLU 1024 | gate/up column, down row | intermediate 512c .. +511 | **reduce_scatter axis 1** (fusable with the experts' one later) | 11 MB |
| moe_add (experts + shared) | local | | none | 0 |
| ffn_residual | local | fp32 streams | none | 0 |
| **Layer total** | | | **as dense, plus dispatch / combine over axis 0 and a second reduce_scatter** | **about 152 MB** |

## Model level

| Component | Placement | Each chip holds | Collective | Per chip |
|---|---|---|---|---|
| embed_tokens [131072, 3584] bf16 | hidden split over columns | [131072, 1792] | none (ids already split by row); broadcast to the 4 streams | 0.44 GiB |
| hc_mean (plain mean of the 4 streams) + final norm | norm weight split by column | [1792] | stats all_gather over axis 1 | 4 KB |
| lm_head [131072, 3584] bf16 (untied) | vocab-sharded over all 8 chips | rows 16384d .. +16383 | the last tokens' hidden gathered over axis 1 and axis 0 to every chip; logits gathered over both axes | 0.11 GiB |
| MTP layer 40 | skipped | none | none | 0 |

## Per-chip total (from the gate, GiB)

| Group | Per chip |
|---|---|
| routed experts, bfp8 (38 layers x 8 experts) | 3.31 |
| activations (planner estimate, one 8192-token chunk) | 2.50 |
| attention weights (bf16, 40 layers) | 1.06 |
| MLA latent cache (40 layers, a quarter of 56320 x 576 bf16) | 0.60 |
| contract KV copy (reserve) | 0.60 |
| embedding (hidden split over columns) | 0.44 |
| shared expert (38 layers) | 0.39 |
| ring_mla KV scratch, RoPE and dispatch tables, CCL buffers | 0.35 |
| dense MLP (layers 0-1) | 0.18 |
| LM head, vocab-sharded | 0.11 |
| mHC projection (fp32) | 0.05 |
| router (fp32) | 0.03 |
| norms, mHC scalars | 0.00 |
| **Total** | **9.63 of 27.20 budget** |

That leaves 17.6 GiB of slack. It covers bf16 routed experts (+3.3 GiB) if bfp8 is not accurate enough and a larger
activation peak; a TP-deduplicated latent cache is not needed.

## Collectives per layer (one 5120-token chunk; 1280 rows per chip; [S/4, 3584] fp32 = 18 MB)

| Layer type | Collectives |
|---|---|
| dense | mHC: 2 x all_reduce [S/4, 32] fp32 over axis 1. Norms: 2 x stats all_gather [S/4, 32] over axis 1. q_a: all_reduce [S/4, 768] fp32 over axis 1. kv_a: all_reduce [S/4, 576] fp32 over axis 1. Attention: KV ring over axis 0 inside `ring_mla` (the prefix, up to 49 MB received at 56k). o_proj: reduce_scatter over axis 1. ffn_norm: all_gather [S/4, 1792] bf16 over axis 1. MLP: reduce_scatter over axis 1. |
| moe | As dense, but the FFN part is: all_gather ffn_norm over axis 1; offset_cumsum histograms over axis 0; dispatch and combine over axis 0 (fabric, per column); reduce_scatter over axis 1 for the experts and for the shared expert (one if the partials are added first). |

Per chunk there is one embedding lookup, the stream mean and final norm, and the LM head on the last tokens.

## Activation estimate (2.5 GiB)

It is sized for the 8192-token ladder chunk, which is 2048 rows per chip.

- **Residual streams.** fp32 `[2048, 7168]` is 59 MB per copy; in, h_mid, out and a mix temporary: 0.24 GB.
- **MoE.** The worst-case dispatch buffer is (4 x 8192 + 7 x 32) rows x 3584 bf16 = 0.24 GB (capacity factor 4: all 4
  experts of a token on one chip), plus metadata and the expert intermediates (~0.15 GB). The combine output
  `[2048, 4, 3584]` bf16 is 59 MB, the fp32 sum and the reduce_scatter output 30 MB each.
- **Attention.** q_b out 13 MB, absorbed q `[16, 2048, 576]` bf16 38 MB (tile + concat copies), ring_mla out
  `[16, 2048, 512]` 34 MB, o_proj partial fp32 29 MB. The gathered-prefix scratch is under `extra_gb_per_chip`.
- **Dense MLP (layers 0-1).** fp32 gate / up / h `[2048, 4608]` are 38 MB each.

About 1.0 GB at the MoE peak, x2 for fragmentation, plus slack.

## Where the Kimi layout does not carry over, and why

- **2 columns, not 4.** TP=2 over 32 heads gives Kimi's 16 heads per chip, so `ring_mla` runs at its tested head
  count. The weights per chip are twice Kimi's share, which is fine at this model size.
- **4 SP rows, not 8.** Kimi's tuned matmul / SDPA configs are keyed on 640 local rows (5120 / 8); Xing has 1280 (and
  512 / 2048 on the other rungs), so the defaults are used until a perf step tunes them.
- **mHC residual (Kimi has a plain residual).** Four fp32 streams per chip, one extra `[S/4, 32]` all_reduce per mHC
  step (hy4's pattern), the Sinkhorn per token on both chips of a row. The stock `mhc_split_sinkhorn` differs from
  Xing's (known issue: logit cap 80 without the lower clamp, a row softmax + eps first, iters - 1 pairs, pre + eps),
  so the attn_hc task forks it into `ttnn.bringup.mhc_split_sinkhorn` with an opt-in Xing mode (clamp [-30, 30],
  exp(x - row max), 20 x (row, column), no pre eps), or composes the loop from elementwise ops. The residual mix uses
  comb (HF `matmul(comb, residual)`), not glm53's comb^T.
- **Experts: groups of 4 chips, 8 experts per chip** (Kimi: groups of 8 rows on 8x4). The forks' own tests cover
  groups of 1 and 2 (2x2 models); a group of 4 is the source op's native case and gets a fork test case at O.1.
- **HiFi4 everywhere.** Kimi's ttMLA runs HiFi2 without fp32 dest in the MLA matmuls and `ring_mla`; the owner rule
  is HiFi4 + fp32 accumulation. The `ring_mla` L1 fit at HiFi4 + fp32 dest is checked first in C.dense.attention.
- **YaRN factor 64, mscale 1.** ttMLA's `get_cos_sin_matrix` builds pure-rotation tables and folds mscale^2 into the
  softmax scale (192^-0.5 x 1.4159^2 = 0.14468), the same as the reference.
- **bf16 attention weights** (Kimi caches bfp8): they are stored bf16, memory is plentiful (1.06 GiB for 40 layers),
  and a lower weight dtype is a perf-step choice, not a plan one.

## Other departures from the references, and why

- **Residual split by column as well as by row (not hy4's "rows only" alternative of a column-replicated residual).**
  It halves the residual and the mHC work per chip, and it lets o_proj / MLP / experts end in a reduce_scatter instead
  of an all_reduce. It matches both Kimi and hy4.
- **ffn_norm gathers its output over axis 1 inside the step**, so the router, the dispatch, the dense MLP and the
  shared expert share one full-hidden tensor (Kimi gathers inside TtMoe and inside the dense FFN separately).
- **Latent cache replicated over the 2 columns.** Dense `ring_mla` is not TP-dedup wired, and the copy costs 0.3 GiB.

## Open items for the component steps

1. `ring_mla` on a 4-row FABRIC_2D ring at HiFi4 + fp32 dest, 16 heads, 1280 local rows: L1 fit and accuracy (tested
   upstream on fabric2d 2x2 / 2x4 with SP 2, and on 8x4). Fallback: all_gather the cache prefix over axis 0 +
   `ttnn.transformer.chunked_flash_mla_prefill` locally.
2. ttMLA's CCL helpers and gather buffers are sized for one fixed local chunk length; the ladder uses 512, 2048 and
   1280 rows per chip, so the Xing module builds per-length buffers (hy4's approach) or builds at the rung's chunk.
3. The block-cyclic latent cache has period = the chunk; state load / read-back at the harness boundary converts
   natural <-> block-cyclic (hy4 `_Geometry.load / read`), including the "last" rung's 50k golden prefix.
4. mHC Sinkhorn: fork (preferred, one op) or compose; test pre / post / comb separately (known issues on whole-output
   PCC hiding mHC bugs).
5. Dispatch / combine / offset_cumsum with a dispatch group of 4 on FABRIC_2D; `unified_routed_expert_moe` with SwiGLU
   (Silu) + `high_precision` at HiFi4 and bfp8 weights. Fallback: the per-expert loop (extract, ttnn.linear, insert).
6. The engine's padded tail: pad tokens are routed and attended like the others (causal, so valid tokens are not
   affected); the cache rows past `actual_end` are zeroed before the layer ack (ttMLA `zero_pad_and_ack`).
