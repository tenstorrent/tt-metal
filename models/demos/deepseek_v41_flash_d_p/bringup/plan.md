# DeepSeek-V4.1-Flash prefill: sharding plan for galaxy_4x4 at 1M tokens, brought up on quietbox_2x2

> **Owner decision, 2026-10-15 (supersedes this page where they differ):** every box keeps weights in a format that
> holds the reference's values exactly; the bfp8 override for galaxy_4x4 is dropped (owner question 1). The deployment
> context on galaxy_4x4 is 512K tokens (26.5 of 27.2 GiB per chip; the longest exact context there is ~720K);
> galaxy_8x4 reaches 1M (18.5 GiB). The per-chip numbers below were computed for 1M with bfp8; the gate's
> `results/plan_boxes.json` / `plan_memory.json` hold the current ones. An e4m3-decoding matmul (the checkpoint's fp8
> bytes, decoded in the op as the experts are) would bring 1M back to galaxy_4x4 exactly.
>
> **Owner decision, 2026-10-15 (2):** the serving chunk is 4096 tokens: 1024 per row on galaxy_4x4 (SP 4), 512 on
> galaxy_8x4; it never grows on a bigger box, and this 2x2 proxy runs 2048-token chunks (1024 per row, as on the
> deployment; F67), scaled and at real size. Deployment context 524,288 on galaxy_4x4 (26.5 of 27.2 GiB per chip),
> 1,048,576 on galaxy_8x4. Where this page says chunk 8192 / W = 2048, read 4096 / 1024 (this box: 2048 / 1024);
> every condition in plan.yaml holds (the gate's plan_boxes.json). Twice the chunks per prefill of the 8192 plan: the
> per-chunk sparse-table refill (an all_gather of the source group's compressed rows) is paid 2x as often; a perf
> item, not correctness.

The gate (PL.1, `python -m models.demos.common.bringup.plan.check_plan`) computes every per-chip number below from
`plan.yaml` and the real checkpoint shapes (`results/plan_memory.json`, `results/plan_boxes.json`). This page gives
the reasoning. `components.yaml` maps every step to its ops; `scaling.yaml` defines the scaled run.

Deployment: galaxy_4x4 (16 x Blackhole p150b, 32 GB each, FABRIC_2D_TORUS_XY), 1,048,576 tokens in 8192-token chunks,
1 slot. This run: quietbox_2x2 (4 x p300c, FABRIC_2D), layers 0-3, 14, 20, 21, 24 (one of each block type), at real size
(56,320 tokens in 5120-token chunks; ladder chunks 2048 / 8192 / 5120) and scaled (heads and experts cut so each chip
holds a deployment chip's share). Budget per chip: 27.2 GiB (15 % headroom). Text decoder only: vision, aligner, image
tokens and the MTP / DSpark layers are never loaded (owner rule).

## The model in one paragraph

40 layers, hidden H = 5120, every layer MoE (384 routed experts, top 6, sqrt-softplus router with a selection bias, one
shared expert; clamped SwiGLU 2304). Residual: mHC with 4 streams, single-pass (a sublayer collapses with the previous
sublayer's pre, DS41-3). Attention: MQA, 64 query heads of 512 over one 512-wide KV row per token (K = V), q via a
1280-rank bottleneck, a per-head sink, output through 8 groups of 8 heads (wo_a, 1024 each) and wo_b. Every query sees
its 128-token window; layers with a compress ratio (2 for layers 2-19, 1 for 20-39) also see the top 512 compressed
rows an indexer picks (32 heads x 128, ReLU-weighted scores). Compressed KV and index keys are made only by the KV
sources (2, 8, 14, 20) and shared by the layers after them; the top-k is made only by the index sources (2, 8, 14, 20,
24, 28, 32, 36); layer 20 also builds a candidate pool of 2048 blocks of 8 positions that 24-36 search within.
Engram (layers 1, 14): an n-gram hash of the token ids picks rows of two 384M-row tables (host), projected and gated
into the streams. Weights fp8 e4m3 with 32 x 32 ue8m0 block scales; experts fp4 e2m1 with a ue8m0 scale per 32; the
inference code fp8-quantizes every GEMM's activations per 32 and fp8 / fp4-quantizes the KV it stores.

## Scheme

Mesh coordinates (row r, column c); axis 0 = rows, axis 1 = columns. Roles (`plan.yaml`): `sequence` = axis 0,
`heads` / `hidden` = axis 1, `experts`, `kv_records`, `vocab` = both axes. This is the Kimi K2.7 4x4 layout
(`knowledge/references/kimi_k27_4x4_layer.md`, xing40's SP x TP) with DeepSeek-V4's sparse attention
(`deepseek_v3_d_p/tt/mla/compressed_sparse_attention.py:TtCSA`).

- **Tokens.** Row r computes the natural block [start + r W, start + (r + 1) W) of the chunk, W = chunk / rows (2048 on
  4x4, 2560 here). The engine's ids arrive in tt-d-gen's `ring_sdpa_reshuffle` order; the embedding step gathers the ids
  (32 KB) over axis 0 and takes its natural block (a device gather with start-dependent positions). Every op therefore
  sees contiguous positions per row, which is what the existing RoPE, indexer and causal-mask ops assume; the caches are
  written by position owner (Serving contract).
- **Residual.** 4 fp32 streams per chip `[W, 4 x H/TP]` (each chip its hidden slice of every stream, xing40 layout) and
  the pending pre `[W, 4]` fp32, replicated over TP. 42 MB per copy on 4x4.
- **mHC.** Partial projection (the chip's rows of `hc_fn^T`) + partial sum of squares, one `[W, 25]` exchange over axis
  1, then `mhc_split_sinkhorn` on every chip of the row; collapse and the residual mix are local.
- **Exact small exchanges.** Partial sums that feed quantizers or near-tie selections (mHC, norms, q_a / kv /
  compressor / weights_proj, engram dots) use all_gather + `ttnn.add` instead of all_reduce: the FPU adds fp32 at about
  TF32 (K-30). They are all small (`[W, <= 3000]` fp32).
- **Attention projections: TP.** q_a, kv, the compressor and the indexer's weights_proj are one K-split matmul on the
  norm output plus one exchange; wq_b by head (16 per chip on 4x4), wo_a by whole groups (2 per chip), wo_b K-split and
  reduce-scattered back to the resting hidden split.
- **Sparse attention: head -> sequence.** `sparse_sdpa` needs a multiple of 32 heads per chip and reads one ROW_MAJOR tensor. An
  all_to_all over axis 1 turns `[16 heads, W rows]` into `[64 heads, W / TP rows]` (TtCSA's reshard), and back after.
  Done whenever TP > 1, so every box runs one path.
- **The sparse table.** A query may pick any earlier compressed row, so every chip needs all of its KV source's
  compressed rows. One persistent buffer per chip (sized for layer 20's ratio-1 rows: 1 GiB at 1M), refilled once per
  chunk per source group (2-7, 8-13, 14-19, 20-39) by an all_gather of the served rows over the mesh, followed by the
  current layer's window region (the 128 rows before start from the served cache + the chunk's window KV gathered over
  axis 0). The top-k ids are mapped to table rows on device. TtCSA instead keeps a replicated table per layer: 2.5 GiB
  per chip at 1M for the 4 sources, and DS41's readers would need copies of their source's.
- **Indexer.** wq_b replicated (all 32 heads on every chip), query rows split over TP (W / TP per chip), scores by
  `ttnn.bringup.ring_indexer_score_dsa` (fp32 DEST, K-13) ringing the SP-sharded index keys over axis 0,
  `topk_large_indices` (512; layer 20's 2048 candidate blocks), all_gather over axis 1 (TtCsaIndexer's layout).
- **Experts: EP over 16 chips, DeepSeek 2D dispatch.** ffn_norm gathers its fp32 output over axis 1, so every chip of a
  row has its W tokens whole; the router runs replicated per row; dispatch / combine run along axis 0 inside each
  column (one group of 96 experts per column, 24 per chip); the column partials are added to the shared expert's and
  reduce-scattered over axis 1 (one CCL).
- **Shared expert and engram projection: TP** by intermediate / output, reduce-scattered with the experts or local.
- **Embedding** hidden-split over TP (`[129280, H/TP]`), replicated over rows; **LM head** vocab-split over all chips.
- **Precision.** Every matmul HiFi4 + fp32 accumulation; fp32 where the reference is fp32 (streams, mHC, norms, router,
  compressor pooling). The model's own quantizers run on the device (owner rule: the inference kernels are the ground
  truth): fp8 act-quant per 32 before every fp8 GEMM, fp8 on the window KV, fp4 on the compressed KV and index q / k
  (`tt/quant.py`, components.yaml header). With them every GEMM operand is bf16-exact and the device GEMMs match the
  reference's up to fp32 summation order; remaining differences are rounding-boundary flips (DS41-1).

## Weights

| Tensor | Checkpoint | Device | Exact? | Why |
|---|---|---|---|---|
| Routed experts w1 / w2 / w3 | e2m1 packed 2 / byte + e8m0 per 32 | the same bytes (uint8), decoded per tile inside the expert op | yes | bfp8 is exact too (measured on 16 experts of layers 0 and 20: every value, both orientations) but needs 1.06 B per value: 26.9 GiB per chip for this run's 8 layers on 2x2, 33.6 GiB on 4x4. The packed bytes are 0.53 B per value (DS41-10) |
| Attention, shared expert, engram wkv, indexer wq_b | e4m3 + 32 x 32 ue8m0 | bf16 (dequantized at load) | yes | e4m3 x 2^k fits bf16; galaxy_4x4 uses bfp8 (owner question 1) |
| Embedding, LM head, router weight, compressor, indexer wk / weights_proj, norms | bf16 | bf16 | yes | as stored |
| mHC fn / base / scale, attn_sink, router bias | fp32 | fp32 | yes | as stored |
| Engram tables | e4m3 + e8m0 per 32, 2 x 98 GB | host only (memory-mapped) | yes | owner rule |

Measured (CPU, `/tmp`-only script, 16-value shared exponent, round to nearest even): bfp4 changes 16-17 % of the
expert values (47-53 % of the 16-value blocks); bfp8 changes none. bfp8 on the fp8 dense weights changes 10 % of the
values by one step, relative Frobenius error 3.5e-3 (wq_a, wkv, wq_b, wo_a, wo_b, shared w1 / w2 of layers 0 and 20).

The expert decode is a fork extension of `ttnn.bringup.unified_routed_expert_ffn` (C.swa.experts): a `weight_format`
option reading the packed bytes and scales (decode to bf16 in L1 per tile, exact), and an `h_quant` option applying
the routing weight and the fp8 block-32 quantization of the gated activation before the down projection, as the
reference does (`q(r h) != r q(h)`). Until it lands, the fused-op-free loop (mimo `_loop_experts`) decodes each local
expert's bytes into a bf16 temporary with a composed chain every chunk: exact, slower, and a temporary departure from
agent rule 5 ("never rebuild a weight-derived constant per call"), behind the Setting `experts_impl`.

## Per layer type (per chip; galaxy_4x4 at 1M / quietbox_2x2 at real size)

S = chunk; W = S / rows (2048 / 2560); collectives are per chunk. Memory is weights per layer per chip (bf16 unless
said; deploy dense weights bfp8, roughly half) plus state.

### Every layer (window part, mHC, MoE): swa (layer 0), and the common rows of every other type

| Component | Placement | Each chip holds | Collective after | Per chip (4x4 / 2x2) |
|---|---|---|---|---|
| attn_hc: rms(4H streams) + projection [4H -> 24] fp32, Sinkhorn | fn columns split with the hidden | hc_attn_fn [24, 4 x H/TP] fp32 | all_gather [W, 25] fp32, axis 1 | 0.5 / 1.0 MB |
| attn_collapse: sum_n pre_n x_n (pending pre) | local | nothing | none | 0 |
| attn_norm (eps 1e-20) + fp8 act-quant | weight split with the hidden | [H/TP] | stats all_gather [W, 1], axis 1 | 0 |
| q_a: wq_a [5120 -> 1280] + q_norm (fused with kv, compressor, weights_proj) | K-split over TP | [H/TP, 1280] | all_gather of fp32 partials [W, 1792..2848], axis 1 (+ SFPU add) | 3.3 / 6.6 MB |
| kv: wkv [5120 -> 512], kv_norm, RoPE, fp8 -> window KV, served cache write | K-split; cache by position owner | [H/TP, 512]; window KV records of its sub-blocks | all_gather [W, 512] bf16, axis 0 (window region and owner writes) | 1.3 / 2.6 MB + 64 / 14 MB state |
| attention: wq_b (by head), RoPE, head->seq all_to_all, sparse_sdpa (+sink) over the window region, back, un-RoPE, fp8, wo_a (by group), fp8, wo_b (K-split) | heads / groups over TP | 16 / 32 heads, 2 / 4 groups | all_to_all x 2, axis 1; reduce_scatter [W, H], axis 1 | 59 / 118 MB |
| attn_residual: post_k y + sum_j comb[j,k] x_j | local (mhc_post) | nothing | none | 0 |
| ffn_hc, ffn_collapse | as attn_hc, attn_collapse | hc_ffn_fn | all_gather [W, 25], axis 1 | 0.5 / 1.0 MB |
| ffn_norm + gather + fp8 act-quant | weight split with the hidden | [H/TP] | stats all_gather; all_gather [W, H/TP] -> [W, H] fp32, axis 1 | 0 |
| router: sqrt(softplus(x gate^T)) + bias, top 6 | replicated per row | gate [384, 5120] bf16 | none | 3.9 MB |
| experts: dispatch, 24 / 96 local experts (fp4 decoded in-op, clamped SwiGLU, x weight, fp8 h), combine | EP over all chips; groups = columns, dispatch along axis 0 | experts 24k .. 24k + 23 (96k .. on 2x2) | offset_cumsum histograms + dispatch / combine along axis 0; reduce_scatter [W, H] fp32, axis 1 (with the shared expert) | 451 / 1805 MB |
| shared_expert: w1 / w3 column-, w2 row-parallel, fp8 h | TP by intermediate | 576 / 1152 of 2304 | (shares the experts' reduce_scatter) | 17.7 / 35.4 MB |
| moe_add, ffn_residual (keeps ffn_hc's pre as the pending pre) | local | nothing | none | 0 |
| **Layer total (swa)** | | | **4 small exchanges, 3 norm stats, 2 all_to_all, 1 reduce_scatter (attn), axis-0 KV gather, dispatch / combine, 1 reduce_scatter (MoE)** | **~0.53 / 1.97 GB + window KV** |

### swa_engram (layer 1): swa + engram first

| Component | Placement | Each chip holds | Collective after | Per chip |
|---|---|---|---|---|
| engram_rows: n-gram hash, table rows [S, 24 x 256] | host (owner rule) | the chunk's rows for its SP row, replicated over TP (input) | host -> device, once per chunk | 25 / 31 MB input |
| engram_kv: wkv [6144 -> 5 x 5120] (+ fp8 act-quant, an identity on fp8 rows) | output split with the hidden (a slice of each of the 5 blocks) | [6144, 5 x H/TP] | none | 79 / 157 MB |
| engram_add: per-stream gate from <h, q_w k_w key>, rstd(h), rstd(key) | local partials | q_w k_w [4, H/TP] fp32 | all_gather [W, 12] fp32, axis 1 | 0.04 MB |
| rest | as swa | | | ~0.53 / 1.97 GB |

### enc_source (layers 2, 8) / enc_source_engram (14): ratio-2 KV and index source

| Component | Placement | Each chip holds | Collective after | Per chip |
|---|---|---|---|---|
| compressor: kv, gate (K-split, in the q_a exchange), softmax over token pairs, norm, RoPE at the group start, fp4 (16 / e4m3) -> compressed KV; wk, k_norm, RoPE, fp4 (32 / ue8m0) -> index keys; served writes; table refill | K-split; caches by owner | wkv, wgate [H/TP, 512]; wk [512, 128] | all_gather of the group's compressed rows over the mesh into the sparse table (up to 512 MiB at 1M); served writes local | 2.6 / 5.2 MB + state |
| indexer: wq_b (replicated), RoPE, fp4 q; weights_proj (exchange); ring score over the SP-sharded index keys; top 512 | replicated weights, query rows split over TP | wq_b [1280, 4096] | ring over axis 0 (index keys, 128 MiB per r2 source at 1M / 4); all_gather top-k [W/TP, 512], axis 1 | 10.8 MB |
| attention over [own compressed rows | window], own top-k | as swa | | as swa | 59 / 118 MB |
| state | | served compressed (dedup) 1/16, index keys 1/4 (TP-replicated, the ring's cache), partial group | | 32 + 32 MB / 0.9 MB at 1M |
| rest | as swa (+ engram for 14) | | | |

### enc_reuse (3-7, 9-13, 15-19), dec_reuse (21-23, ...): readers

Window part, MoE and mHC as swa. The attention reads the sparse table its KV source refilled this chunk and its index
source's top-k of the same queries (F65 `state_sources`; in single-block tests the golden source state, K-43). No
compressor, no indexer, state = window KV only. **~0.53 / 1.97 GB + window KV.**

### dec_candidate (20): ratio-1 KV and index source, candidate pool

As enc_source with ratio 1 (no pooling: latent = norm(wkv x); groups = tokens). The indexer adds the candidate pool:
block max over 8 positions, the newest block pinned (a load-time table), top 2048 blocks (`topk_large_indices`, its
maximum k), kept on device for layers 24-36 of the same chunk. Ring over 256 MiB of index keys at 1M (/ 4 rows). The
sparse table holds layer 20's 1M compressed rows (1 GiB at 1M): its size sets the table buffer.

### dec_reindex (24, 28, 32, 36): own top-k inside layer 20's candidates

Indexer on layer 20's index keys (ring over axis 0), scores masked to layer 20's candidate blocks of the same queries
(scatter of the block ids into a block mask, expanded x 8), top 512; attention over layer 20's sparse table with its own
top-k. Indexer weights 10.8 MB.

## Model level

| Component | Placement | Each chip holds | Collective | Per chip (4x4 / 2x2) |
|---|---|---|---|---|
| embed: id gather + natural block, clamp (K-72), embedding, 4 streams + one-hot pre | table hidden-split over TP | [129280, H/TP] bf16 | all_gather of the ids (32 KB), axis 0 | 0.31 / 0.62 GiB |
| final_norm: collapse with the pending pre, distributed RMSNorm | weight split | [H/TP] | stats all_gather, axis 1 | 0 |
| lm_head (untied) | vocab over all chips | [H, 129280 / chips] | rows gathered over both axes; logits all_gather | 0.08 / 0.31 GiB |

## Per-chip totals (from the gate, GiB)

| Group | galaxy_4x4, 40 layers, 1M | quietbox_2x2, run layers, 56k |
|---|---|---|
| routed experts (packed fp4 + e8m0) | 16.81 | 13.45 |
| activations (planner estimate) | 3.00 | 6.00 |
| window KV, served (TP-dedup) | 2.50 | 0.11 |
| attention weights (deploy bfp8 / run bf16) | 1.25 | 0.94 |
| sparse-attention table (one source group at a time) | 1.00 | 0.05 |
| RoPE tables (fp32, both rope types, to max_seq) | 0.50 | 0.50 |
| window region, index / position / dispatch tables, CCL scratch | 0.45 | 0.45 |
| shared expert | 0.35 | 0.26 |
| embedding | 0.31 | 0.62 |
| engram projection | 0.08 | 0.29 |
| LM head | 0.08 | 0.31 |
| router, indexer, compressor, mHC, served compressed / index keys, norms | 0.58 | 0.14 |
| **Total** | **26.91 of 27.20** | **23.12 of 27.20** |

The run box at real size is dominated by its 8 layers x 96 experts (13.5 GiB even packed). The deployment fits at 1M
with 0.3 GiB to spare inside the 15 % headroom, and only with bfp8 dense weights (owner question 1); with bf16 dense
weights it needs 28.4 GiB. Further savings if needed: the embedding split 2D (0.23 GiB), index keys TP-dedup (0.12),
the router weight K-split (0.11), an fp8 window cache (1.2 GiB, lossless for this fp8 data but a contract format
change: serving_contract.md owner question 2).

### Activation estimate

Deployment (3.0 GiB, 2048 rows per chip at chunk 8192): residual copies 0.2 GB, attention 0.4 GB, indexer at 1M with
128-query score blocks 0.6 GB, MoE (dispatch buffer at ~4 x the mean 3072 rows, one decoded expert, combine) 0.5 GB,
engram 0.1 GB: ~2.0 GB, x1.5. This run (6.0 GiB): the s16384 rung's 8192-token chunk puts 4096 rows on a chip with
twice the hidden share and 96 local experts: ~3 GB, x2, plus the top-k / candidate history for the state check.

## Collectives per layer (one chunk)

| Layer type | Collectives (4x4, chunk 8192; [W, H] fp32 = 42 MB) |
|---|---|
| swa / readers | mHC: 2 x all_gather [W, 25] fp32 (axis 1). Norms: 3 stats gathers [W, 1] + the ffn_norm output all_gather [W, H/TP] -> [W, H] fp32 (axis 1). q_a / kv exchange: all_gather [W, 1792] fp32 (axis 1). Window KV: all_gather [W, 512] bf16 (axis 0) + the 128-row carry (one gather). Attention: 2 x all_to_all of q / out (axis 1), reduce_scatter [W, H] (axis 1). MoE: offset_cumsum histograms, dispatch + combine along axis 0, reduce_scatter [W, H] fp32 (axis 1) |
| + engram (1, 14) | all_gather [W, 12] fp32 (axis 1); one host input [W, 6144] bf16 |
| + KV source (2, 8, 14, 20) | the exchange also carries kv / gate (+1024 columns for ratio 2); all_gather over the mesh of the group's compressed rows into the sparse table (end / ratio rows x 1 KiB, up to 1 GiB at 1M) |
| + index source (2, 8, 14, 20, 24-36) | ring over axis 0 inside the score op (index keys, end / ratio x 256 B); all_gather of the top-k [W/TP, 512] int32 (axis 1) |

Per chunk also: the id gather, the final norm, the LM head on the rows the caller needs, and one ack per layer.

## Boxes

From the gate (sizes in GiB; "scaled" = the run's layers at the deployment's per-chip shapes and context on this box's
DRAM; "real" = the run's layers at real dimensions on the box at its context):

| Box | Mesh | Scale | Context (scaled) | Full model: weights / longest context | Run's layers: scaled / real |
|---|---|---|---|---|---|
| galaxy_4x4 (deploy) | 4x4 | sequence 1, heads 1, hidden 1, experts 1, kv_records 1, vocab 1 | 1,048,576 / 8,192 | 23.1 GB / 1,048,576 | 9.9 / 9.9 GB (budget 27.2) |
| galaxy_8x4 | 8x4 | sequence 2, heads 1, hidden 1, experts 2, kv_records 2, vocab 2 | 2,097,152 / 16,384 | 16.1 GB / 1,048,576 | 10.3 / 9.6 GB (budget 27.2) |
| loudbox_2x4 | 2x4 | sequence 0.5, heads 1, hidden 1, experts 0.5, kv_records 0.5, vocab 0.5 | 524,288 / 4,096 | 42.5 GB / 0 | 11.3 / 14.2 GB (budget 27.2) |
| quietbox_2x2 (run) | 2x2 | sequence 0.5, heads 0.5, hidden 0.5, experts 0.25, kv_records 0.25, vocab 0.25 | 524,288 / 4,096 | 81.8 GB / 0 | 13.3 / 23.1 GB (budget 27.2) |

| Condition | galaxy_4x4 | galaxy_8x4 | loudbox_2x4 | quietbox_2x2 |
|---|---|---|---|---|
| query rows per chip after the head->seq reshard (sparse_sdpa; indexer TP sub-shard): tile rows | 512: plan | 512: plan | 512: plan | 1280: plan |
| ratio-2 compressed rows per SP row per chunk (ring indexer block-cyclic chunk): tile rows | 1024: plan | 1024: plan | 1024: plan | 1280: plan |
| served-record sub-block per chip (W / TP tokens): whole 32-token records | 512: plan | 512: plan | 512: plan | 1280: plan |
| routed experts per chip (dispatch regions, expert op): whole experts | 24: plan | 12: plan | 48: plan | 96: plan |
| wo_a output groups per chip (whole groups of 8 heads) | 2: plan | 2: plan | 2: plan | 4: plan |

Every box takes the deployment's path on every condition (the run box's real-size row shows its real-size counts; its
scaled run has the deployment's 512 / 1024 / 24 / 2). Paths that are the same by construction rather than by count:
the head->seq reshard runs whenever TP > 1 (on 2x2 at real size 32 heads per chip could skip it; it does not, so the
run tests the deployment's reshard), and the expert decode and the sparse-table refill are one code path everywhere.

What differs from the deployment box, and what that leaves untested here:
- **Dense weight format.** galaxy_4x4 overrides the fp8 dense weights to bfp8 (plan.yaml `boxes`, owner question 1);
  every other box keeps bf16. Untested until then: accuracy with bfp8 dense weights (it can be measured on this box with
  the same override, at real size, before the Galaxy).
- **Mesh sizes.** Dispatch groups of 4 chips along axis 0 (here 2), the 4-chip SP ring of the indexer and of the KV
  gathers (here 2), all_to_all over 4 columns (here 2), FABRIC_2D_TORUS_XY (here FABRIC_2D). loudbox_2x4 checks the
  4-column TP paths (16 heads per chip at real size) before the Galaxy.
- **Per-chip shares the scaled run cannot match.** Hidden 5120 is not scaled (it is the residual): the hidden share per
  chip is 2560 here vs 1280 (norms, K-split matmuls, shared expert 1152 vs 576, engram projection). The scaled run
  matches heads (16 per chip), groups (2), experts (24) and the sequence per chip (chunk 4096 = 2048 rows per row).
- **Context.** 1M (the indexer over 512K-1M keys, its 128-query score blocks, the 1 GiB sparse table, the 0.3 GiB
  memory margin) only on the deployment box; this run reaches 56k real and the owner's 64k scaled cap.
- Deployment-only checks: a 1M prefill's memory high-water mark per chip, the indexer score time at 1M, dispatch with
  groups of 4 on the torus, bfp8 dense-weight accuracy (if chosen), the runner smoke (DS41-8).

### Scaled run (SG.1)

`scaling.yaml` keeps 32 of 64 heads, 4 of 8 wo_a groups and 96 of 384 experts (gate: "scaled run keeps
num_attention_heads 32 of 64, o_groups 4 of 8, n_routed_experts 96 of 384"); sequence factor 0.5 (chunk 4096, contexts
capped at 64k by the owner). The reference reads `inference/config.json` (n_heads, n_routed_experts, o_groups), which
`plan/scale.py` does not copy (top-level files only): SG.1 must make the reference and the hooks read the scaled counts.

## Serving contract

How the layout meets each section of `serving_contract.md`:

- **Input.** The uint32 `[SP, 1, chunk/SP]` ids (sharded on axis 0, replicated on axis 1) with the `{slot, actual_start,
  actual_end}` header: the embedding step gathers the ids over axis 0 and each row takes its natural block with a
  device gather whose positions follow from start (tables built at load, no per-chunk host work); positions >=
  actual_end are masked out of every cache write, the compressor and the indexer, and clamped before `ttnn.embedding`
  (K-72, `ttnn.minimum`: the vocab is not a power of two). Every row knows the absolute position of each local row, so
  RoPE, the window, compressor groups and engram n-grams follow positions. Engram: the host keeps each slot's ids by
  position and reads the 3 before start from it. Entry points `embed(tokens, start)`, `from_host(h, start)`,
  `layer(i, h, start, state, end)`, `begin_chunk(tokens, start)` as listed there.
- **KV cache.** The five configs, in table order, bf16 ROW_MAJOR, ND-sharded `[1, 1, 32 / r, width]` per record
  ROUND_ROBIN_1D (`init_kvpe_cache`). Position g lives on row (g / W) % SP (the contract's rule); within the row,
  column c holds sub-block c of every W block: TP-dedup for 0_window_kv and the compressed configs (local row
  (g div (W SP)) (W / TP) + (g mod W) mod (W / TP); device group = one chip), TP-replicated for the index-key configs
  (they are the ring indexer's key cache; device group = the row's chips). Per position, never overwritten except by
  the same rows; one region per slot and layer. Memory per chip per slot at 1M on 4x4: window 2.5 GiB + compressed
  0.16 + index keys 0.16 GiB. `state.served` / `load_served` / `load_tokens` read and write natural order through the
  same position map; `state.formats()` reports bf16 for all five.
- **Attention and cache writes.** Every 32-aligned start: rows compute natural blocks, so a chunk crossing a W block
  needs no special case in compute, and each chip writes the positions it owns from the axis-0-gathered window KV and
  compressed rows. Only [actual_start, actual_end) is written (compressed groups with (j + 1) r <= actual_end); the last
  record's pad rows are zeroed before the ack (`zero_pad_and_ack`). The window rows before start come from the served
  cache on every chunk (one path for cold, follow-up and pulled-back chunks); compressed rows before start come from the
  source's served cache through the sparse-table refill. The pulled-back last chunk recomputes the same rows (layer 0
  bit-identical: its window KV depends only on token and position). The compressor's partial group resets at every
  start; the indexer, candidates and top-k run per chunk.
- **Acks.** One per layer per chunk, in layer order, after the layer's cache writes and pad zeroing: host sink after
  `ttnn.synchronize_device`, or `outbound_socket_service_sync` on the same command queue for D2H (`kv_ack.py`). Every
  layer owns 0_window_kv, so all 40 ack (no `kv_slot_layer_ids`).
- **Adapter and table.** As listed there (contract step K.1); the table's per-record address and device group follow
  the position map above (dedup records: the owner chip; index keys: the row's chips). Config names sort in table order
  (DS41-6). FABRIC_PAYLOAD_SIZE in [4352, 15232] (K-79).
- **Deployment.** PREFILL_SP 4, PREFILL_TP 4, 40 layers, chunk 8192, max seq 1,048,576, max_slots 1 (the memory above
  is for one slot), `2d_torus_xy` on galaxy_4x4, `2d` elsewhere.
- **Questions.** The plan takes the defaults: bf16 row-major per position (1), TP-dedup of the window KV (2: needed at
  1M), 1 slot at 1M (3), prefix reuse on (4: any 32-aligned start works), d2p off (5: the engram id history), explicit
  geometry in our tools (6), the runner smoke on the deployment box (7, DS41-8).

## Accuracy notes for the component tasks

- **Quantizer flips.** With the model's quantizers on the device, a last-bit difference before an fp8 / fp4 rounding
  moves a value by a whole step (DS41-1). Calibrate the limits per step (K-35); compare steps on the same device inputs.
  `tt/quant.py` itself must match `reference/kernels.py` bit for bit on random and golden inputs (C.swa.q_a builds
  fp8_qdq, C.enc_source.compressor fp4_qdq).
- **Exact sums.** all_gather + `ttnn.add` for fp32 partials that feed a norm, a quantizer or a selection (K-30).
- **Router.** fp32 selection (K-23); the FPU reads fp32 operands at about TF32, so if routing flips beyond the
  reference's near ties, split x into bf16 hi + lo halves (two bf16 matmuls).
- **Indexer.** bf16 scores (topk_large_indices is bf16-only) tie where the reference's fp32 do not: gate top-k overlap
  plus exact structure (K-42); candidates as sets.
- **Attention sink.** High-precision sparse_sdpa with the sink is a fork extension (K-31); check the sink term's
  scale arithmetic (K-27).
- **mHC.** Sinkhorn per K-20; pending pre per DS41-3; fp32 streams.
- **Experts.** The routing weight multiplies h before its fp8 quantization; dispatch carries q(x) (bf16-exact); check
  the fp4 decode bit for bit against `reference/weights.py:fp4_dequant`.

## Departures from the reference plans (Kimi K2.7 4x4, xing40, glm53), one reason each

- **Sparse attention, not ring MLA:** DS41 reads 128 + 512 rows per query; dense chunked attention would read the whole
  prefix.
- **Head -> sequence reshard around attention:** sparse_sdpa needs a multiple of 32 heads per chip; 4x4 gives 16.
- **One sparse table refilled per source group per chunk, not a resident table per layer:** 2.5 GiB less at 1M, and
  readers share their source's rows.
- **Experts kept as the checkpoint's fp4 bytes and decoded in the op, not bfp8:** bfp8 does not fit (33.6 GiB on 4x4).
- **Window KV TP-dedup:** 10 GiB per chip at 1M if replicated.
- **Natural compute blocks + owner-mapped cache writes, not compute in the server's order:** keeps every existing
  position-dependent op unchanged; the owner writes come from tensors the attention gathers anyway.
- **The model's quantizers on the device:** they are part of the model (owner rule); with them the device GEMMs match.
- **Indexer heads replicated, queries split:** a TP head split would need an all_reduce of [W, keys] scores (GBs at 1M).
- **Engram on the host:** owner rule.

## Owner questions

1. **Dense weights on galaxy_4x4.** 1M tokens on 16 chips fit only with bfp8 for the fp8 dense weights (26.9 of 27.2
   GiB; bf16 needs 28.4). bfp8 changes 10 % of those values by one step (rel. error 3.5e-3, measured). Alternatives: an
   exact fp8-byte format with a matmul fork that decodes e4m3 (6 % smaller than bfp8, kernel work), an fp8 window cache
   (lossless, contract format change), deployment on galaxy_8x4 (bf16 dense, 16.1 GiB of weights per chip), or 4x4 at a
   shorter context. Plan default: bfp8 on galaxy_4x4 only.
2. **Routed experts.** The owner expected bfp8; it is exact but does not fit this run's box at real size (26.9 GiB per
   chip of experts for 8 layers) nor the deployment box (33.6 GiB). Plan: the checkpoint's packed fp4 + e8m0 bytes on
   the device, decoded exactly in the expert op (fork extension); bfp4 (lossy, 16-17 % of values) is not planned.
3. **Plan check.** The engram tables need a host placement in the plan check (DS41-9); until the framework has one the
   gate reports one plan error for them.
