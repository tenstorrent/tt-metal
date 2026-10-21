# The ideal prefill attention op for DeepSeek V4.1

Status: op 1 is implemented as `ttnn.transformer.deepseek_v4_prefill_sdpa` (bf16, any H; test:
`tests/ttnn/unit_tests/operations/sdpa/test_deepseek_v4_prefill_sdpa.py`), passing on Blackhole; not yet used by the
model. Op 2 is not implemented. Figures are for the server's 1M-token context with 1024-token
traced prefill chunks and TP 4 (16 local query heads), from the V4.1-Flash `config.json`.

## Summary

V4.1 prefill attention is sparse by construction: each query reads its 128-token sliding window plus at most
`index_topk` = 512 compressed entries. Today's traced prefill still materialises the dense form of that pattern,
an additive mask and indexer scores `[T, sliding_window + T + cap]` per layer type per chunk, and runs dense
masked SDPA over every entry in the early position tiers. At 1M tokens the ratio-1 layers have `cap` = 1,048,576,
so one chunk's mask is 2.1 GB in bf16 (4.3 GB for the fp32 compare intermediates), and that allocation is what
fails today.

The ideal op takes the selected entry rows as indices, builds the causal window and validity in the kernel from
a device-side start position, and pairs with a fused indexer that streams over the entries keeping a running
top-512, so no `[T, E]` tensor exists anywhere. Per-chunk memory and compute become independent of the context
length (apart from the indexer's score pass, which stays O(E) per query).

## What V4.1 attention computes

| Layers | `compress_ratios` | Keys a query sees |
| --- | --- | --- |
| 0, 1 (and the MTP layers 40-42) | 0 | Its 128-token sliding window only |
| 2-19 | 2 | Window plus up to 512 of the ratio-2 entries, selected by the lightning indexer |
| 20-39 | 1 | Window plus up to 512 of the ratio-1 entries; past 16384 entries the selection is two-level |

Shared by all layers:

* MQA with K == V: one 512-wide latent head (`head_dim` 512, `qk_rope_head_dim` 64), 64 query heads (16 per chip
  at TP 4). V is the same row as K, so an op should read each row once.
* Per-head attention sinks (`sinks`), contributing to the softmax denominator only.
* Inverse RoPE on the output (applied after SDPA today; could stay outside the op).
* Kv sources at layers 2, 8, 14, 20 own the compressed entries; the other compressed layers read their group's.
* Index sources at layers 2, 8, 14, 20, 24, 28, 32, 36 run the lightning indexer (32 heads x 128 dim); the
  other compressed layers reuse their group's latest selection.
* Two-level select (see `PREFILL_TWO_LEVEL.md`): layer 20 keeps each query's best 2048 blocks of 8 entries
  (pinning the block holding its newest entry); layers 24, 28, 32, 36 take their top-512 only from those blocks.

Visibility of entry `w` to query `i` of a chunk starting at `start` (`start` a multiple of the rate):
`w < start / rate + (i + 1) // rate`. Window column `k` (token `start - 128 + k`) is visible iff
`i < k <= i + 128` and `start - 128 + k >= 0`.

## What the code does today

Traced prefill (`DeepSeekV41TracedPrefill` in `tt/prefill/model.py`, attention in `tt/prefill/attention.py` and
V4's `deepseek_v4_flash/tt/prefill/attention.py`):

1. `mask_tables_host` builds `static` and `threshold` `[T, 128 + T + cap]` once per compressed layer type.
2. Each chunk, `_step_inputs` builds the additive mask `static + (threshold > start) * NEG` on device for every
   compressed layer type (fp32 compare and multiply, then bf16), and slices the entry columns out as
   `index_cuts`.
3. Index sources score every entry (`indexer_score_dsa`, `[T, cap]`), add the cut, and either threshold at the
   512th score into an additive selection mask (`_select_from_scores`) or, in sparse tiers, take the top-512 row
   indices (`_select_rows`).
4. Attention, by layer type and tier:
   * Sliding layers: `_sliding_attend`, mask-free (`maskless=True`).
   * Compressed layers in dense tiers: `_attend_static`, which adds the selection onto the mask and runs
     `ttnn.transformer.scaled_dot_product_attention` over all `128 + T + cap` keys.
   * Compressed layers in sparse tiers: `_attend_sparse`, `ttnn.transformer.sparse_sdpa` over the 512 selected
     rows plus 128 window rows (`step.window_idx`), no mask. A tier is sparse only once its first query already
     sees 512 entries and a whole window, because `sparse_sdpa` takes sentinels only as a contiguous tail.

Problems at long context:

* Step 2 runs for sparse tiers too, although their attention ignores the mask; only `index_cuts` is used.
* The mask, the cut, the indexer scores and the selection mask are each `[T, ~cap]` per chunk.
* Dense tiers read and multiply every entry: cost per chunk grows with the position.
* `ttnn.topk` with k = 512 runs over up to 1M columns per query.

Why `ttnn.transformer.flash_mla_prefill` / `chunked_flash_mla_prefill` are not the answer: they are mask-free
but plain causal over one contiguous sequence, with no sliding window, no rate-dependent entry visibility, no
per-query selection and no attention sinks.

## Per-chunk cost at 1M tokens (ratio-1 layer, T = 1024, per chip)

| Tensor or work | Today | Ideal |
| --- | --- | --- |
| Additive mask | 2.1 GB bf16, 4.3 GB fp32 intermediates | None |
| Indexer cut (`index_cuts`) | 2.1 GB | None (computed in kernel) |
| Indexer scores | `[1024, 1M]`, about 2 GB | Running top-512 per query in L1 |
| Selection | `[1024, 1M]` mask or `[1024, 512]` uint32 | `[1024, 512]` uint32 (2 MB) |
| Attention keys read | Dense tier: every row (1M x 512 x 2 B = 1 GB) | 640 rows per query, gathered |
| Attention FLOPs (QK and PV) | Dense tier: about 3.4e13 | About 2.1e10 |
| Indexer FLOPs | About 8.6e12 per index source | Same: the score pass is inherent |

Persistent buffers stay as today: ratio-1 entries 1M x 512 x 2 B = 1 GB, index keys 1M x 128 x 2 B = 256 MB per
kv source copy.

## The ideal ops

### 1. Sparse window-plus-entries attention

```python
out = ttnn.transformer.sparse_window_mla_prefill(
    q,              # [1, H_local, T, 512] bf16
    window_kv,      # [1, 1, 128 + T, 512] sliding-window K == V rows: [tail | chunk]
    cache_kv,       # [1, 1, cap, 512] compressed KV cache: front-anchored entries (persistent buffer)
    cache_idx,      # [1, 1, T, 512] uint32 rows of cache_kv each query attends; sentinel anywhere = skip
    start,          # [1] int32 device tensor: the chunk's absolute start position
    *,
    window_size=128,
    attention_sink=sinks,
    scale=scale,
)                   # [1, H_local, T, 512]
```

The two KV inputs are separate tensors, read by separate rules:

| Input | Contents | Which rows query `i` reads | Lifetime |
| --- | --- | --- | --- |
| `window_kv` | The 128 tail rows carried from the previous chunk, then this chunk's `T` rows | The contiguous run `i + 1 .. i + 128`, minus rows before position 0 (from `start`); no indices | Built per chunk by the layer |
| `cache_kv` | The kv source's compressed entries, entry `w` in row `w` | The rows listed in `cache_idx[i]`, gathered; no contiguity | Persistent across chunks; written by the kv source |

Keeping them apart means:

* No per-layer `concat` of `[cache rows | tail | chunk]` into one key tensor (today's `_attend_sparse` and
  `_attend_static` both build it, copying up to `cap` rows each layer).
* `cache_idx` indexes `cache_kv` only, so it holds no window entries: `window_idx` and the concatenation offsets
  (`cap + i + 1`) go away, and the index width is exactly `index_topk`.
* The two inputs can have different layouts and dtypes. `cache_kv` can be large, in DRAM and later fp8 or
  sharded across chips; `window_kv` is small (`(128 + T) x 512 x 2 B` = 1.2 MB at T = 1024) and could live in L1.
* A layer type with no compressed entries (sliding-only layers) passes no `cache_kv`.

Both inputs feed one softmax: the kernel processes the window run and the gathered cache rows as KV chunks of the
same online-softmax loop, with the sink added once to the denominator.

What else it must do that `sparse_sdpa` does not:

* **Window in the kernel.** Query `i`'s window rows are a contiguous run of `window_kv`, read as a block
  instead of gathered. No `window_idx` tensor, no window columns in any mask.
* **Sentinels anywhere, or a per-query valid count.** An early chunk's query may see fewer than 512 entries. If
  the op skips invalid rows of `cache_idx` wherever they sit, or derives the count from `start`, `rate` and `i`,
  every tier can be sparse and the dense path disappears.
* **K == V read once**, as `sparse_sdpa` already does (`v_dim` = 512), for both inputs.
* **Any head count.** 16 local heads today need `q_pad` to reach a tile of 32; the op should accept 16, or
  process head groups without padding to 32.
* **Trace-safe.** All position dependence comes from `start` (device tensor) and `cache_idx`; shapes depend
  only on `T` and `cap`.

Sliding-only layers (0, 1, 40-42) can use the same op without `cache_kv`, or keep `_sliding_attend`.

### 2. Fused streaming indexer select

```python
cache_idx, blocks = ttnn.experimental.deepseek.indexer_select_prefill(
    q_idx,          # [1, 32, T, 128] indexer queries after RoPE
    idx_keys,       # [1, 1, cap, 128] front-anchored index keys
    weights,        # [1, 1, T, 32] per-head gate, already scaled
    start,          # [1] int32 device tensor
    *,
    rate=1,
    topk=512,
    candidate_blocks=None,   # consumer layers: [1, 1, T, 2048] uint32 block ids from the candidate source
    emit_blocks=False,       # candidate source: also return each query's best 2048 blocks of 8
)
```

* Streams over the entry rows in chunks, computing scores and keeping a running top-512 per query in L1. The
  causal cut comes from `start`, `rate` and the query index. No `[T, cap]` scores, cut or mask.
* Candidate source (layer 20): also keeps a running top-2048 over block maxima, pinning the block of the query's
  newest entry, and returns block ids (8 KB per query instead of a `[T, cap / 8]` mask).
* Candidate consumers (layers 24, 28, 32, 36): score only the entries in the given blocks, so their score pass
  costs 2048 x 8 = 16384 entries per query instead of `cap`.
* `cache_idx` feeds op 1 directly: it indexes the same `cache_kv` rows the indexer scored. Decode's `fused_lightning_select_kv` is the single-query form of this.

With both ops, `mask_tables_host`, `io.masks`, `index_cuts`, `index_pads`, `window_idx`, `q_pad`,
`candidate_pins` and the dense/sparse tier split (`_sparse_tiers`) all go away.

## Path from today's code

1. **Model-only, no new kernel.** In `_step_inputs`, skip the full mask for layer types in a sparse tier and
   build only the entry-column cut, slicing `threshold` before the compare and comparing in bf16. Removes the
   window columns and the fp32 intermediates.
2. **Extend `sparse_sdpa`.** A second KV input for the sliding window (read as a contiguous run per query, beside
   the gathered cache rows), sentinels anywhere (or a per-query valid count) and the window's position-0 cut from
   `start`. Every compressed tier becomes sparse; the dense masked SDPA, the full masks and the per-layer key
   `concat` go away.
3. **Fused indexer select.** Streaming top-512 (and top-2048 blocks for the two-level select). Removes the last
   `[T, cap]` tensors and the large-k `ttnn.topk`.
4. **Sequence parallel indexer.** The score pass is the remaining O(length) per query; split the entry rows
   across chips and merge the per-chip top-512.

## Risks

* Top-k ties: thresholding today keeps every entry tied with the 512th; a streaming top-k keeps exactly 512.
  The reference uses `topk`, so the streaming form is closer to it, but per-layer PCC must be rechecked.
* Running top-512 per query in L1: 512 scores and indices per query for a block of queries per core; size the
  query block against L1 alongside the score tiles.
* bf16 block maxima near ties can change which candidate blocks survive (also a risk for today's code).
* `sparse_sdpa` is Blackhole-only; so would the extensions be.
