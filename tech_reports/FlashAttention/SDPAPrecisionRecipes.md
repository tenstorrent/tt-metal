# SDPA Precision Recipes

`ttnn.transformer.scaled_dot_product_attention`, `chunked_scaled_dot_product_attention` and
`joint_scaled_dot_product_attention` take an optional `precision=ttnn.SDPAPrecision.<RECIPE>`. A recipe fixes every numerical choice in the attention kernel, so
callers pick an accuracy/throughput point instead of tuning compute-kernel fields. Omitting `precision`
keeps the streaming kernel and its `compute_kernel_config` / `exp_approx_mode` controls, except where a call would
reach one of the legacy loops: those calls run a recipe (see [Routing](#routing)). With a recipe both controls are
accepted and ignored (see [Legacy arguments](#legacy-arguments-with-a-recipe)).

```python
out = ttnn.transformer.scaled_dot_product_attention(
    q, k, v, is_causal=False, attn_mask=mask, precision=ttnn.SDPAPrecision.BALANCED
)
```

## Recipes

All recipes use the flash-attention online softmax: for each K chunk, S = QKᵀ, m = rowmax, P = exp(scale·(S − m)),
then the running numerator O and denominator l are rescaled by c = exp(scale·(m_old − m_new)) and the chunk's
PV and row sums are added. The recipes differ in the arithmetic of each step.

| | Legacy (no `precision`) | STANDARD | BALANCED | ACCURATE | FAST |
|---|---|---|---|---|---|
| QK matmul | HiFi2 | HiFi2 | HiFi4 | HiFi4 | LoFi |
| Scores S, P | BF16 | BF16 | FP32 | FP32 | BF16 |
| S − m | FPU, BF16 | FPU, BF16 | FPU, TF32 operands, FP32 dest | packer FP32 L1 add | FPU, BF16 |
| exp(P) | approximate exp | approximate exp | bit-trick exp + cubic | bit-trick exp + cubic (tighter fit) | approximate exp |
| Correction c | approximate exp | approximate exp | accurate FP32 exp | accurate FP32 exp | approximate exp |
| PV matmul | HiFi2 | HiFi2 | HiFi2 | HiFi4 | LoFi |
| Row sum l | BF16 | BF16 per chunk, FP32 L1 add | P·1 matmul, LoFi | P·1 matmul, HiFi2 | BF16 per chunk, FP32 L1 add |
| Running O, l | BF16 | FP32 in L1, reference max | FP32 | FP32 | FP32 in L1, reference max |
| O / l | reciprocal, multiply | reciprocal, multiply | reciprocal (2 Newton steps), multiply | same as BALANCED | reciprocal, HiFi2 multiply |
| K/V storage | BF16, BFP8 or BFP4 | BF16, BFP8 or BFP4 | BF16, BFP8 or BFP4 | BF16, BFP8 or BFP4 | BF16, BFP8 or BFP4 |
| Input rounding (caller) | none | none | none | none | `prepare_sdpa_input` |

- **Legacy** is the existing streaming kernel (`compute_streaming.hpp`) that runs when `precision` is omitted.
  It is not a recipe and is listed for reference.
- **STANDARD** keeps O and l in FP32 in L1 with a *reference* row maximum m_ref. P = exp(scale·(S − m_ref))
  is computed in BF16 as in the legacy kernel, but the packer adds each chunk's PV and row sums onto the FP32 state in L1
  (the add is exact in FP32), so long-K accumulation does not swamp and needs no per-chunk fold. m_ref changes
  only when a row's maximum exceeds it by θ = 16·ln 2; that chunk rescales O and l once by c =
  exp(scale·(m_ref,old − m_ref,new)), rounding the state to BF16 once. To keep P in range while S exceeds
  m_ref, the approximate exp's input offset is lowered by τ = 28·ln 2 (exactly 28 octaves, so P is 2⁻²⁸ times
  the legacy kernel's P bit for bit and cancels in O / l). Saturation then starts τ + 0.72 above m_ref, past θ. See
  `streaming/recipe_streaming.hpp`, and [Fused K chunks](#fused-k-chunks-standard-fast) for how K chunks
  after the first skip the row maximum.
- **BALANCED / ACCURATE** hold scores and state in FP32. The score exp is Schraudolph's bit trick on a 2⁻¹⁰
  grid in log2, refined by a cubic, and carries a constant factor of about 0.970 that cancels in O / l.
  Across the mantissa, the refined exp ripples ±0.24% (BALANCED) or ±0.10% (ACCURATE). See
  `streaming/recipe_sfpu.hpp`.
- **FAST** runs both matmuls at LoFi, which truncates SrcA to 5 and SrcB to 7 significant bits.
  `prepare_sdpa_input` rounds Q to 7 bits and K/V to 5 bits (or onto the BFP4 grid) with round-to-nearest-even
  beforehand, so the truncation loses nothing further. Its state is kept like STANDARD's, and the final
  O / l multiply runs at HiFi2 so O is not truncated.

## Fused K chunks (STANDARD, FAST)

With the reference maximum fixed, a K chunk does not need its row maximum before the exp. Every K chunk after a
Q chunk's first runs a fused chunk (`streaming/recipe_fused_chunk.hpp`) that streams row groups through
QK → exp → PV without the max reduction:

- **m_ref folded into QK.** The QK matmul gains one inner step, [Q | M] × [Kᵀ ; −e₀], where M holds m_ref in
  column 0 and −e₀ is −1 in row 0, so DEST receives S − m_ref directly. Both operands of that step are exact in
  one fidelity phase (m_ref is truncated to 7 bits, −1 has one), so STANDARD replays its HiFi2 image once and
  only its inner 0–15 half (`INNER_HALF`). The extra step costs 1/(2·D/32) of the QK work.
- **Exp on the pack thread.** The packer's SFPU takes the exp in place and packs P once, L1-accumulating it
  onto per-row partial sums.
- **Saturation check and redo.** A row whose chunk sums reach the redo threshold may have saturated the exp
  (S − m_ref beyond τ + 0.72). Its row group is redone on the reduce path: the real maximum with the θ select,
  P, PV, and one rescale of its O and l rows. Groups are checked in units (two groups for STANDARD, one for
  FAST); a unit that fires is re-checked group by group, so only groups that need it are redone and
  every other group keeps m_ref. The output is the same as checking each group alone.
- **Software pipeline.** QK of group g, the check of the unit ending at g − 1 and the PV of an older group are
  interleaved in K pieces, so the FPU's PV overlaps the pack thread's exp. Dense STANDARD runs Q chunks of up
  to six tiles in one-row groups so the pipeline has enough groups; the ring kernels and FAST keep
  two-row groups.

Fused chunks need a QK subblock of at least two tiles and no attn_mask; the factories drop their CBs
(29–31) and run the reduce path when they do not fit L1. Errors are unchanged against the unfused chunks
(19 input distributions: fused/unfused rel-L2 ratio geomean 0.94, at most 1.04), and every later optimization in this path is
bit-identical.

## Accuracy and throughput

Relative L2 error (%) against FP64 attention on the same BF16 inputs. Q is 256 rows, D128, one head, with
normally distributed inputs. "Outliers" multiplies each Q, K and V element by 10 with probability 0.1%.

Throughput is TFLOP/s per Tensix core. It was measured on one Blackhole core with K/V resident in L1 (no DRAM
traffic), D128 and a K sequence of 8192. The legacy kernel is 1.99 TFLOP/s at Q256/K512. "vs legacy" is the
geometric mean ratio over Q chunks 128–320 and K chunks 128–512. STANDARD is at or above the legacy kernel at every
chunk pair (lowest 1.01, at Q128/K384) and FAST at least 1.21× (Q128/K128); BALANCED and ACCURATE are furthest behind at
K512 (0.61 and 0.45).

| Recipe | K 4096 | K 32768 | K 262144 | K 262144 outliers | TFLOP/s/core (Q256/K512) | vs legacy |
|---|---:|---:|---:|---:|---:|---:|
| Legacy (no `precision`) | 2.54 | 2.75 | 18.2 | 5.27 | 1.99 | 1.00 |
| STANDARD | 2.37 | 2.42 | 3.04 | 4.39 | 2.09 | 1.07 |
| BALANCED | 0.39 | 0.39 | 0.38 | 0.59 | 1.21 | 0.64 |
| ACCURATE | 0.18 | 0.18 | 0.18 | 0.41 | 0.88 | 0.47 |
| FAST, BF16 K/V | 2.92 | 2.94 | 3.44 | 13.5 | 2.86 | 1.37 |
| FAST, BFP8 K/V | 3.00 | 3.00 | 3.51 | 13.4 | 2.86 | 1.37 |
| FAST, BFP4 K/V | 16.8 | 16.9 | 16.7 | 42.9 | 2.85 | 1.37 |

The legacy kernel's BF16 running state swamps at long K. With small logits (Q and K scaled by 0.25), its error is 3.7% at
K 32768 and 53% at K 262144, against 1.5% and 2.3% for STANDARD. BFP8/BFP4 K/V gain nothing over BF16 at the
compute level; they cut K/V bandwidth and L1 to about a half or a quarter.

## Support

- Blackhole. Noncausal attention with an optional additive `attn_mask` of shape [1|B, 1|H, Sq, Sk] (BF16, BFP8,
  BFP4, or FP32 for BALANCED/ACCURATE). Joint attention supports the `"rear"` strategy without a mask.
- `is_causal`, `sliding_window_size` (causal or centred), windowed attention (`cu_window_seqlens` with
  `windowed_q_token_offset` or its tensor form) and chunked prefill (`chunk_start_idx` or `chunk_start_idx_tensor`)
  through the [K-range model](#causal-sliding-window-chunked-and-windowed-attention). Causal and sliding windows
  need Sq == Sk, as in the legacy kernel. Chunked prefill reads a paged K/V cache (page table [B, blocks per
  sequence], `paged_cache_geometry`).
- [MLA prefill, attention sinks and concatenated heads](#paged-kv-mla-attention-sinks-and-concatenated-heads):
  `flash_mla_prefill` and `chunked_flash_mla_prefill` take `precision`; `attention_sink` and
  `output_concat_heads` work on dense SDPA (sinks also on chunked SDPA).
- Batch and GQA are supported, with any number of batch/heads (more than the grid's cores is fine). Q/K lengths
  need not be tile or chunk multiples. Head dim and chunk sizes must be tile multiples, and the chunks must fit in
  L1.
- Tiled, interleaved inputs, mask and output, in DRAM or L1 (sharded tensors are rejected). Q and K/V may be BF16,
  BFP8 or BFP4 (see below); the output has Q's dtype, like the legacy kernel.
- Any finite positive `scale` (default 1/√D). Unsupported arguments raise before dispatch; there is no fallback.

## Legacy arguments with a recipe

The legacy kernel's numerical controls are accepted with a recipe, so a caller can name a recipe without first
removing them. The rule is: **the recipe owns the numerics.**

- `compute_kernel_config` is ignored: its math fidelity, `math_approx_mode`, `fp32_dest_acc_en`, `packer_l1_acc`
  and `dst_full_sync_en` are what the recipe table above fixes per recipe. A config shared with other ops (for
  example a HiFi4 FP32-dest config used for the linears) therefore runs exactly the named recipe, bit for bit.
  Routing a call *without* `precision` by its config (FP32 dest to ACCURATE) is a separate, op-level decision.
- `exp_approx_mode=False` (in `SDPAProgramConfig`) is ignored. Each recipe already picks its exp: STANDARD and
  FAST the approximate exp the legacy kernel uses by default, BALANCED and ACCURATE an FP32 exp more accurate than
  the legacy kernel's non-approximate one. A caller that wants an accurate exp names BALANCED or ACCURATE.
- `scale` is honored. As in the legacy kernel, the kernels fold it into the exp, P = exp(scale·(S − m)), and never
  into Q or K; the reference-max thresholds (θ, τ) are in units of the scaled scores, so they hold at any scale.
  An `attn_mask` is pre-multiplied by 1/scale (0 and −inf stay exact; FP32 for BALANCED/ACCURATE), and the
  pre-scale is skipped at scale 1. When 1/scale is a power of two the pre-scale is exact in the mask's own format,
  so BALANCED/ACCURATE keep a BF16/BFP8/BFP4 mask narrow (same output, half the mask traffic).
- **Packed inputs.** Every recipe takes BF16, BFP8 or BFP4 K/V. The unpacker expands them into the matmul source
  registers, so the stored values enter the recipe's arithmetic unchanged and the recipe's error
  against FP64 on those values is the same as with BF16 K/V. On dense and joint SDPA, K and V may differ (Qwen-VL
  vision passes K BF16, V BFP8): K's storage names the recipe's K/V storage, and V's circular buffer takes V's
  format. A BFP8/BFP4 Q is widened to BF16 in DRAM before the
  kernel (exact), and the BF16 result is narrowed back to Q's dtype after it, matching the legacy output dtype.
  The narrowing adds that format's rounding (about 0.5% rel-L2 for BFP8).
- **FAST and scale.** `prepare_sdpa_input` rounds Q to 7 and K to 5 significant bits for the LoFi matmul; the
  scale multiplies S = QKᵀ afterwards, inside the exp, so the two do not interact. A caller that folds the scale
  into Q itself (and passes `scale=1`) must do so before `prepare_sdpa_input`, like any other Q transform. A BFP8
  Q already has at most 7 significant bits, so it is exact at LoFi without preparation; BFP8/BFP4 K/V for FAST
  still need it (the 5-bit rounding).
- **Memory.** Inputs, mask and output may be L1-interleaved. The op-selected blocking reserves an L1 output's share
  of each core's L1, and the L1 fit check counts every live L1 buffer.
- **Many heads.** Up to one batch/head per core, each head's Q chunks run on a chain of cores that forwards K/V.
  With more batch/heads than cores, all Q chunks of all heads split evenly over the grid without chains, and
  each core's reader follows the head of every chunk it reads.

## Routing

Without `precision`, a call that the BF16-DEST streaming kernels (`compute_streaming.hpp`) do not serve runs a recipe
(Blackhole and Wormhole B0; `sdpa.cpp`, "Precision routing"). The legacy loops these calls used to reach
(`sdpa_standard`, `sdpa_joint`) are deleted; only the ring joint FP32 gaps below still reach `sdpa_ring`
(`sdpa_legacy_loops.hpp`).
The DEST mode is read from `compute_kernel_config` as before (`fp32_dest_acc_en`, default off).

| Entry point | BF16 DEST (default) | `fp32_dest_acc_en=True` |
|---|---|---|
| `scaled_dot_product_attention` (any mask / causal / sliding window / windowed / sink / concat heads), `chunked_scaled_dot_product_attention`, `flash_mla_prefill`, `chunked_flash_mla_prefill` | streaming kernel | ACCURATE |
| `joint_scaled_dot_product_attention` | STANDARD | ACCURATE |
| `ring_joint_scaled_dot_product_attention` | streaming kernel | ACCURATE; legacy loop only for combinations the legacy FP32 loop rejects too (sink, sliding window, KV-pad rotation, circular cache) or a V wider than Q |
| `exp_ring_joint_scaled_dot_product_attention` | streaming kernel; STANDARD for a blocking the streaming kernel cannot build (QK subblock taller than two tiles, K chunk not a multiple of the subblock row, one Q subblock), which failed to compile before | ACCURATE (failed to compile before) |
| `ring_distributed_scaled_dot_product_attention` | STANDARD | ACCURATE |

- A routed call is the named recipe, bit for bit, with the blocking below. `compute_kernel_config` and
  `exp_approx_mode` are ignored as for any recipe call. ACCURATE is the recipe that keeps FP32 scores and state, the
  closest match to an FP32-DEST request; FP32 DEST was often a shared linear config rather than an SDPA choice.
- Routed dense, chunked, MLA and joint calls choose their blocking on the whole compute grid
  ([Blocking](#blocking)): `program_config` chunk sizes and grids were tuned for the legacy kernels, and the chooser
  beats them or ties (table below). Routed ring-distributed calls choose their chunks on the caller's grid (8/1 heads,
  D128, BFP8, 131072 rows: the caller's Q64/K64 143 ms, the choice 43 ms). Ring and exp ring calls keep the caller's
  chunks and grid when the recipe supports them.
  `sub_core_grids`, which prefill ignored, is dropped. Routed dense and joint calls also ignore
  `max_cores_per_head_batch` (a decode setting, default 16, that legacy prefill ignored): a head's K/V chain may span
  the grid, which single-head calls need (one head, D512, S16384: 57 ms on 16 cores, 7 ms on the grid). Zero chunk
  sizes still need an explicit `precision`.
- A routed joint call with empty joint tensors (`[B, H, 0, D]`, FLUX.2 single-stream blocks) runs as dense SDPA and
  returns an empty joint output with legacy's spec.
- Routed ring calls return the recipe's scratch as the third output, not an LSE (no caller reads it).
- Exp ring routes on Blackhole only.

Op time on a P150 (ms, median of five batches of five calls; the routed column is what the same call runs now, with
the blocking above, the legacy column the same call before routing):

| Caller (shape, caller's chunks) | Before (legacy) | Routed |
|---|---|---|
| tt_transformers prefill: causal, 32/8 heads, D128, BFP8 Q/K/V, S4096, Q256/K256 on 8x8, HiFi4 + FP32 | 7.60 | 2.38 |
| same, S8192 | 29.40 | 7.78 |
| tt_transformers chunked prefill: Q2048 at 6144, paged BFP8 cache (64-row blocks), tensor start | 12.74 | 3.99 |
| Gemma-style: causal + window 1024, scale 1, 16/8 heads, D256, S8192, Q128/K128 | 12.65 | 2.62 |
| bge_m3: 16 heads, D64, S8192, padding mask, Q128/K256, LoFi + FP32 | 10.03 | 7.61 |
| bge_m3: B8, S512, padding mask, Q256/K256, HiFi4 + FP32 | 0.470 | 0.468 |
| nomic embed v2: 12 heads, D64, S512, padding mask, scale 1, HiFi3 + FP32 | 0.095 | 0.087 |
| Qwen3-VL text: causal, 32/8 heads, D128, S4096, Q128/K512 | 4.99 | 2.17 |
| Qwen3-VL vision: 16 heads, D96, S4096, Q128/K128 | 4.63 | 2.03 |
| qwen_image joint: 24 heads, D128, N4096 + L128, Q512/K256, HiFi2 + FP32 (ACCURATE) | 10.63 | 3.48 |
| qwen_image joint, BF16 DEST (STANDARD) | 6.18 | 1.36 |
| Flux-style joint: 24 heads, D128, N4096 + L512, Q256/K512, BF16 DEST (STANDARD) | 5.33 | 1.55 |
| FLUX.2 single-stream joint: 6 heads, D128, N4608 + empty joint, Q128/K512, BF16 DEST (STANDARD) | 1.73 | 0.42 |
| Gemma-4 global: causal, 4/1 heads, D512, S4096, Q128/K128, HiFi4 + FP32 | 3.50 | 1.40 |
| SDXL VAE mid-block: 1 head, D512, S16384, Q64/K64, HiFi2 + FP32 | 8.26 | 7.18 |
| Wan2.2 VAE (720p): 1 head, D384, S14400, Q32/K256, HiFi2 + FP32 | 6.01 | 4.55 |
| Qwen2.5-VL vision full layers: 16 heads, D96, S16384, BFP8 Q, K BF16 / V BFP8, Q256/K256, HiFi4 + FP32 | 96.2 | 28.6 |

The routed FP32-DEST calls are faster than the legacy FP32 loop, the small encoders included (nomic S512) or level
with it (bge_m3 B8 S512: the same device time, 0.452 ms traced, bound by reading the head-broadcast mask once per
head). Recipe calls are a cached device operation, about 0.02 ms of host time per call (the program hash
counts the L1 the call's layout may use, its free L1 less its outputs capped at the layout's full size, so
back-to-back calls holding L1 outputs share one program, inside a trace capture too, until L1 runs short); while every call rebuilt
its program on the host these two rows ran 0.545 and 0.261 ms. For reference, the BF16-DEST streaming kernel
(unchanged) runs the tt_transformers rows in 2.28 / 8.72 ms.

## Causal, sliding-window, chunked and windowed attention

These run on the same compute loop as unmasked attention (`dataflow/recipe_key_range.hpp`). Query row q at global
position p = offset + q (the chunk start, or the windowed Q offset) sees one key interval [lo(p), hi(p)): causal
k ≤ p, a causal window p − w < k ≤ p, a centred window |k − p| ≤ w/2, a window of `cu_window_seqlens` the keys of
p's window. Neither end moves left as p grows, so for a Q chunk each K chunk is

1. outside every row's interval: skipped (no reads, no compute);
2. inside every row's interval: the recipe's unmasked chunk, fused for STANDARD and FAST;
3. otherwise an edge: the attn_mask path, with {0, −2^100} BFP4 mask tiles the writer generates (mixed tiles are
   cached by their diagonal offset, so a causal call writes one).

Compute reads each Q chunk's K range and its unmasked sub-range from a control page the writer sends, and processes
the edge chunks first: a Q chunk's first K chunk always runs the unfused reduce path, so starting on the (masked)
edge leaves every full chunk to the fused path. Masked keys add −2^100 instead of −∞, so a row with no visible key in
its Q chunk's first K chunk keeps a finite running max, and the next chunk's real keys replace it with a zero
rescale (STANDARD and FAST: θ is exceeded, the fused chunk's saturation check redoes the group). A paired recipe pads
an odd Q chunk to even (the single-row group mishandles a row whose keys are all masked in its first K chunk). The
chunk start (scalar) is part of the program hash, as in the legacy op; `chunk_start_idx_tensor` and
`windowed_q_token_offset_tensor` are read on device, so one trace serves every offset.

**Work split.** All heads' Q chunks, sorted by cost (the latest chunks first, heads interleaved), are dealt to the
whole grid in snake order: round r gives core c sorted entry r·cores + c (r even) or r·cores + cores − 1 − c
(r odd). Causal work is balanced to within a light Q chunk per core, and every core works even when the heads do
not divide the grid.

**K/V sharing.** There is no K/V chain, but in each round deal positions p and p + heads run Q chunks j and j − 1 of
the same head on two fixed partner cores. The full K chunks both Q chunks see travel from the first core to the
second (store and forward: the receiver passes its circular-buffer write pointer to the sender in a semaphore), so
a head's K/V is read from DRAM about once per round instead of once per Q chunk; edge chunks and chunks only one
side sees are read directly. FP32 recipes, which keep one K/V buffer slot, share only causal and chunked ranges (with
a window, neighbouring chunks' shared ranges start at different points and the sender waits for the receiver).

Op time on one P150b (ms, full grid, trace replay; causal unless noted; legacy = the kernels without `precision`,
BF16 dest; FP32-dest legacy runs where it builds):

| Shape | STANDARD | legacy BF16 | FAST BFP8 | ACCURATE | legacy FP32 |
|---|---|---|---|---|---|
| 10 heads × 8192², D128, Q256/K512 | 1.49 | 1.92 | 0.88 | 2.46 | does not build |
| same, Q128/K256 | 2.51 | 3.49 | 1.52 | 2.71 | 3.51 |
| 16 heads × 8192², D64, Q256/K512 | 1.27 | 1.73 | 1.00 | 3.00 | 2.16 |
| 32/8 heads (GQA) × 8192², D128 | 5.39 | 6.24 | 3.20 | 7.21 | does not build |
| 8 heads × 32768², D128, Q128/K256 | 29.1 | 43.9 | 16.4 | 39.4 | 44.4 |
| sliding window 1024, 10 heads × 8192², Q256/K256 | 0.64 | 0.64 | 0.49 | 0.88 | 2.48 |
| chunked: 8 heads, 2048 rows at 6144 of 8192 | 0.76 | 1.03 | 0.48 | 1.45 | does not build |

ACCURATE at D64 stays slower than the legacy FP32 kernel (its FP32 softmax costs more per score; noncausal D64 is
2.05× the legacy FP32 time, causal 1.4×).

## Paged K/V, MLA, attention sinks and concatenated heads

- **Paged K/V.** The reader translates each K/V tile row through the sequence's page-table row (cache block
  `table[b][t / block]`, row `t % block`), so blocks may be in any order and need not align with the K chunk. With
  `paged_cache_geometry` the cache was declared for another layer: the call's block size and KV heads address it,
  and they must cover each block's elements exactly (legacy rule; not with MLA).
- **MLA** (`head_dim_v` < the QK head dim). Q and K keep the QK width; V, the numerator state, PV and the output use
  `head_dim_v`, with their own matmul subblock width. Without a V tensor (`flash_mla_prefill(q, k, head_dim_v)`,
  `chunked_flash_mla_prefill`) the reader takes V as K's first `head_dim_v` columns, as the legacy kernel does.
- **Attention sinks** (`attention_sink` [1, H, 1, 1], unscaled logits as in the legacy op): each row's softmax
  denominator gains exp(scale·sink). One hook at normalization adds k·exp(scale·(sink − m)) to l (accurate FP32
  exp), where m is the maximum the row's P were taken against and k is the score path's mean factor
  (P ≈ k·exp(scale·(s − m)): 0.970 for ACCURATE, 0.965 for BALANCED, 1.005·2⁻²⁸ for STANDARD and FAST with their
  headroom; measured with a sink that takes nearly all of a row's weight). Accuracy is unchanged with or without a
  dominant sink.
- **`output_concat_heads`** writes [B, 1, Sq, H·Dv] from the writer (the same tiles at concatenated addresses).

Each is a compile-time switch that is off unless the call uses it; builds without them are unchanged. The largest
programs are STANDARD's fused ones with BFP8 K/V: Q256/K512 D128 with a sink takes 68,880 B of the 70,656 B kernel
config buffer (plain 66,464 B), Q256/K256 D64 with a sink 70,368 B; see [Blocking](#blocking) for the geometries the
chooser avoids.

## Blocking

With a recipe, the op chooses Q and K chunk sizes when the caller leaves them unset: no `program_config`,
or a chunk size of 0 in `SDPAProgramConfig`. For exp ring it also chooses the SDPA grid width. The chooser
(`sdpa_recipe_blocking.cpp`) scores every supported chunk pair that fits L1, counting the attn_mask buffer.
It uses a roofline cost model fitted to Blackhole timings, plus pipeline fill/drain terms that dominate
short-K cross attention. Causal, sliding-window and chunked calls cost the K chunks each Q chunk actually processes,
dealt over the grid as the kernels deal them, plus fitted per-K-chunk streaming, per-edge-chunk and per-Q-chunk
terms; sliding windows also consider 128-row K chunks. When nothing in the searched range (Q from 128 rows, K from
256 to 512 rows) fits L1, as for head dims of 512 and more, the search extends down to one-tile chunks. A chunked prefill whose start is a device tensor is costed at
the latest start its K/V length allows (one program serves every start). With more batch/heads than cores, every Q
chunk reads its head's K/V from DRAM (no forwarding chain), which a fitted term adds (bge_m3 B8 S512: Q256/K512
0.455 ms, where the roofline alone picked Q128 at 0.488). The chooser never picks a geometry whose program is known
to overflow the kernel config buffer (`recipe_program_fits`: STANDARD odd Q chunks of 7+ tiles with a joint segment,
a sink or a K tail, or a head dim of odd tile count; packed K/V with a head dim of odd tile count and one of those
features; measured program sizes are listed there). Explicit chunk sizes are honored ([routed](#routing) calls choose
theirs). Blocking never changes a recipe's arithmetic, only its rounding order. Without a recipe, chunk sizes must be
explicit.

## Ring and exp ring attention

`ring_joint_scaled_dot_product_attention` and `exp_ring_joint_scaled_dot_product_attention` take the same
`precision`. Without it the streaming ring kernels run (or a recipe, see [Routing](#routing)), with their chunk limits (ring: Q 128-320, K 256/384/512,
D 64/128/256; exp ring: K512, D128). The recipes keep one online-softmax state per Q
chunk in L1 across all ring steps. They mask key tails (shard padding, `logical_n`, the joint tail) and
normalize once, on the last active step. Exp ring rows with several head segments (up to three passes) run
pass-outer and ring-inner.

- Ring: noncausal, `is_causal` and `is_balanced` (see [Causal rings](#causal-rings)), chunked prefill, indexed
  caches (`kv_cache_batch_idx`, or `slot_id` with the metadata tensors outside chunked prefill) and a V head dim
  below Q's (MLA with a separate V); exp ring: noncausal. KV-pad rotation (`kv_actual_isl`, metadata on chunked
  prefill), circular caches, sinks and sliding windows are rejected (`sliding_window_size` with an explicit error:
  the legacy FP32 ring kernel ignores it). A BFP8/BFP4 Q is widened to BF16 first and the outputs narrowed back, as on the dense path;
  K/V may be BF16, BFP8 or BFP4 under every recipe. `scale`, `compute_kernel_config` and `exp_approx_mode` follow
  [Legacy arguments](#legacy-arguments-with-a-recipe).
- `ring_distributed_scaled_dot_product_attention` takes `precision` too (see
  [Ring-distributed attention](#ring-distributed-attention)).
- `logical_n` (and ring's `logical_l`) may be a host scalar or a single-value device tensor, so a captured
  trace can replay with new lengths.
- For FAST, prepare K/V before they are communicated.
- Ring's third output is internal scratch, not an LSE.
- With several Q chunks per core, each chunk's state (FP32 O and l, maxima) is checkpointed to the third output
  around every Q chunk on every ring step. With fused chunks (STANDARD, FAST) this is streamed: the last
  K chunk hands finished O row groups to the writer, a restore acks maxima and sums first and then each O row as
  it lands, and compute waits only for the rows it is about to accumulate onto. On a BH Galaxy this takes
  FAST BFP8 Wan 2.2 720p attention from 19.32 to 18.80 ms (480p 5.87 to 5.74 ms), bit-identical.

### Chunked prefill

With Q shorter than the K/V shard (and not `is_cross`), Q is the newest chunk group's slab on each device and K/V the
cache of every group so far: device d's shard holds its slab of each group back to back, so local K tile t is global
tile (t / slab) · group + d · slab + t mod slab. Every step masks in the sequence's frame
(`SDPA_RECIPE_RING_CHUNKED`): compute walks the K chunks the reader sends (before `logical_n`, or the device's last Q
row with the reader's dense causal skip), pops those past a Q chunk's last row, and stamps the per-tile mask on the
rest from that mapping, as the causal diagonal does. A first chunk group's step on a later device's K has no visible
key; the recipes mark it inactive on the host. The legacy FP32 tests of the feature (Kimi-style D576 / V128 with BFP8
K/V, and an indexed ND-sharded cache with V narrower than Q) now route to ACCURATE.

### Causal rings

A causal ring (`is_causal`) holds the sequence's chunk d on device d; balanced (`is_balanced`) holds chunks d and
2R − 1 − d, so devices do equal work. The ring reader and writer schedule both as for the legacy kernels; the recipe
compute (`SDPA_RECIPE_RING_CAUSAL`) follows them:

- On the device's own K shard, query q sees key k ≤ q in the shard's frame (a balanced shard's two chunks keep
  their order). QK tiles past a Q tile row are set to −∞ in the pack thread, the diagonal tiles above their
  diagonal, as the key-tail mask does; K chunks starting past a Q chunk's last row are popped unread. Every row
  sees key 0 in its first chunk, so its running max is always finite.
- Unbalanced: earlier devices' K is fully visible, later devices' steps are inactive.
- Balanced, K from an earlier device: only that shard's early chunk precedes this device's rows; the reader sends
  the K chunks up to the one straddling it, and compute masks the straddle as a key tail. K from a later device:
  the early chunk's Q chunks see none of it and are skipped (the reader sends them nothing), so they normalize on
  the last step that reaches them; compute and writer find that step by walking the ring order once.

Causal builds compile at −O2 (BALANCED/ACCURATE unpack and pack otherwise build at −O3), which keeps every tested
variant inside the kernel config buffer at no measurable cost. On a 1x2 P150b ring, 8 heads, D128, 16384 rows per
device:

| Recipe vs legacy (each at its best chunks) | Causal | Balanced |
|---|---|---|
| STANDARD Q256/K512 vs legacy BF16 dest | 12.47 vs 13.04 ms | 8.71 vs 8.89 ms |
| ACCURATE Q256/K512 vs legacy FP32 dest Q128/K256 | 32.58 vs 23.37 ms | 21.43 vs 17.35 ms |

The legacy FP32-dest kernel does not fit Q256/K512 at this shape; at Q128/K256 ACCURATE takes 35.64 ms (causal)
and 26.03 ms (balanced). The causal/noncausal time ratio of each recipe matches the legacy kernels'.

## Ring-distributed attention

`ring_distributed_scaled_dot_product_attention` gives each device of a ring of size R the causal attention of two
slabs of the whole sequence, chunks `ring_id` and 2R − 1 − `ring_id` of 2R, so every device does the same work. It
needs no communication: Q, K and V hold the whole sequence on every device. With `precision` it runs on the dense
recipe kernels and the [K-range model](#causal-sliding-window-chunked-and-windowed-attention): a head's Q chunks are
the two slabs' chunks, read from Q at their sequence rows (`SDPA_RECIPE_Q_SLAB_JOBS`; the reader and writer map a job
to its chunk of the sequence), and the causal key range of each follows from its global position. The snake deal
balances the early slab's chunks against the late slab's. Without an explicit `ring_id`, each
device takes its index along the mesh axis of length R, and the op runs one program per device that differs only
in the slabs' rows (runtime arguments, folded into the program hash).

- The legacy op's rules: R even, the sequence a multiple of 64R (tile-aligned slabs), Sq == Sk without prefix
  caching; Q chunks must divide the slab (the op-chosen blocking picks one that does). Q and K/V types follow the
  recipe (BFP8 Q is widened and the output narrowed back, as for dense SDPA).
- Prefix caching (`page_table` with `chunk_start_idx`): Q holds the rows after the cached prefix, K/V the paged
  cache, read through the chunked-prefill key range (the slabs' global positions start at `chunk_start_idx`).

Throughput on one P150b, Galaxy Llama 70B shape per device (8 Q heads, 1 K/V head, D128, BFP8 Q/K/V, R = 4,
grid 7x10, Q256/K512, mean of ring positions 0 and 3):

| Sequence | Legacy BF16 dest | STANDARD | Legacy FP32 dest | ACCURATE |
|---|---|---|---|---|
| 8192 | 1.82 ms | 0.66 ms | 2.75 ms | 1.34 ms |
| 32768 | 6.79 ms | 4.73 ms | 10.43 ms | 11.66 ms |

The legacy op walks both slabs on the same cores (two phases); the recipe spreads both slabs' chunks over the grid,
which wins at short sequences. Without a K/V chain each core reads its own K/V; with GQA (8 Q heads per K/V head)
BF16 K/V then becomes read-bound at 32k (STANDARD 6.66 ms), BFP8 K/V halves the traffic.

## FAST inputs

SDPA never rounds, checks or converts its inputs. For FAST, the caller calls:

```python
q = ttnn.transformer.prepare_sdpa_input(q, is_query=True)                        # RNE to 7 bits, BF16
k = ttnn.transformer.prepare_sdpa_input(k, is_query=False, dtype=ttnn.bfloat8_b)  # RNE to 5 bits
v = ttnn.transformer.prepare_sdpa_input(v, is_query=False, dtype=ttnn.bfloat8_b)
```

Call it after any Q transforms (RoPE, norms) and before K/V are cached or communicated. A plain cast to
BFP8/BFP4 is not equivalent: it skips the 5-bit rounding that LoFi relies on, and the packer's shared-exponent
rounding differs from the round-to-nearest-even with saturation in `prepare_bfp4.cpp`.

## Kernel map

| File | Role |
|---|---|
| `kernels/compute/sdpa_recipe.cpp` | Dense / joint compute kernel: Q-chunk loop over `sdpa_segment_v2` |
| `kernels/compute/ring_joint_sdpa_recipe.cpp`, `exp_ring_joint_sdpa_recipe.cpp` | Ring and exp ring compute: one recurrent state per Q chunk across ring steps (`streaming/recipe_ring.hpp`) |
| `streaming/recipe_streaming.hpp` | Shared K-chunk step: reduce path (first chunk, redo), FP32 / reference-max state, normalization |
| `streaming/recipe_fused_chunk.hpp` | Fused K chunk (above) |
| `streaming/recipe_sfpu.hpp`, `recipe_tail.hpp` | Exp variants, key-tail masking |
| `dataflow/recipe_key_range.hpp` | K-range model: per-row key intervals, chunk classes and order, edge mask tiles, snake Q deal |
| `streaming/recipe_checkpoint.hpp`, `dataflow/recipe_state_transfer.hpp` | Ring multi-Q state checkpoints (compute side, writer side) |
| `dataflow/reader_recipe.cpp`, `ring_joint_*_impl.hpp`, `exp_ring_joint_*_impl.hpp` | Readers / writers; the ring and exp ring bodies are shared with the legacy kernels through a `Policy` struct (the legacy kernels compile unchanged) |
| `sdpa_recipe.cpp`, `sdpa_recipe_blocking.cpp` | Host: recipe → CB layout and defines; chunk chooser |

Compile-time defines set by the host:

| Define | Set for | Meaning |
|---|---|---|
| `SDPA_RECIPE_FP32` | BALANCED, ACCURATE | FP32 scores and state (DEST in FP32) |
| `SDPA_RECIPE_ACCURATE` | ACCURATE | HiFi4 PV and the tighter exp fit |
| `SDPA_RECIPE_LOFI` | FAST | LoFi matmuls (prepared inputs) |
| `SDPA_RECIPE_FUSED` | STANDARD, FAST with QK width ≥ 2 | fused K chunks (inactive with an attn_mask) |
| `SDPA_RECIPE_MASK` | an attn_mask or a key range | additive mask on the reduce path |
| `SDPA_RECIPE_KRANGE` | causal, sliding window, chunked, windowed | K range per Q chunk; mask on edge chunks only |
| `SDPA_RECIPE_Q_SLAB_JOBS` (reader, writer) | ring-distributed | a head's Q chunks are two slabs of the sequence |
| `SDPA_RECIPE_KV_SHARE` (reader) | key ranges with more cores than heads | K/V passed between a round's partner cores |
| `SDPA_RECIPE_QK_W`, `SDPA_RECIPE_PV_W` | all | matmul subblock widths |
| `SDPA_RECIPE_RING` (in the ring kernels) | ring, exp ring | key-tail masking, resident state |
| `SDPA_RING_STREAM_STATE` | ring STANDARD and FAST with fused chunks | streamed multi-Q checkpoints |
| `SDPA_RECIPE_RING_CAUSAL` | causal (and balanced) ring | diagonal mask, skipped K and Q chunks |

**Code size.** Each program must fit the 70656 B kernel config buffer, and the ring and exp ring FAST
kernels sit within a few hundred bytes of it. The reduce path (`SDPA_RECIPE_COLD`), normalization and the ring
unpack copy of the fused chunk are size-optimized and out of line; the ring kernels build unpack/pack at -O2
with fused chunks. FAST's ring kernels use a single no-MOP matmul init (every reinit re-records), and
the streaming writer has one copy of the transfer loops. New code in these paths should be checked against the ring/exp ring FAST BFP8
tests (Q224 two-pass, Wan Q320), and then on a Galaxy model: Galaxy programs are 0.7-1.5 KB larger than the
1x2 test programs (multi-link reader/writer, joint text), so a 1x2 pass does not guarantee a fit.
