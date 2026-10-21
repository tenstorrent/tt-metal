# Long-context prefill cost at 1 M positions: the growing term per chunk

The 8x4 1 M run (`_logs/mesh/long_8x4_N1048576.log`, 16 K-token chunks, 45 layers) spent a growing time per chunk:
58 s at 64 K positions, 62 at 131 K, 72 at 256 K, 84 at 384 K, 94 at 512 K (~10-12 s more per 131 K of prefix), ~98 min
for the whole prompt of which ~43 min is the growing term. This note attributes the growing term on one chip (chip 0,
one DSA layer, one 16 K-token chunk of real rows at the end of a synthetically seeded prefix, `StageTimer` wall times
with a device synchronization per stage) and records the two changes made: the whole-prefix latent copy (measured, not
the term) and the multi-row prefill scorer (the term).

## 1. The whole-prefix latent copy (`TtLatentCache.user_rows_rm`) is bandwidth-bound and negligible

`sparse_sdpa` needs the user's whole prefix as a ROW_MAJOR [1,1,start+T,512] bf16 tensor once per chunk per DSA layer.
`tests/test_prefix_rows.py` times the old path op by op (a `ttnn.slice` of the paged TILE cache over the contiguous block
range, a reshape view, `ttnn.to_layout` ROW_MAJOR = the multicore untilize) against the direct path
(`ttnn.untilize_with_unpadding` of the leading block range straight out of the paged cache, no slice), on a 1-slot cache
allocated for 1 048 640 positions and seeded with random rows; min of 2 warm runs. Both kv tensors are compared bitwise
and `sparse_sdpa` is run from each (128 random queries, random valid index rows); the outputs are bitwise identical.

| prefix N | RM bytes | old ms = slice + reshape + untilize | old GB/s | direct ms | direct GB/s | `user_rows_rm` now |
|---|---|---|---|---|---|---|
| 16 384 | 17 MB | 0.3 = 0.1 + 0.03 + 0.1 | 63 | 2.7 | 6 | old (slice + untilize) |
| 131 072 | 134 MB | 1.5 = 0.7 + 0.02 + 0.7 | 88 | 2.8 | 47 | old |
| 262 144 | 268 MB | 3.0 = 1.4 + 0.05 + 1.3 | 90 | 3.4 | 80 | old |
| 524 288 | 537 MB | 5.8 = 2.8 + 0.04 + 2.5 | 92 | 4.2 | 128 | direct |
| 1 048 576 | 1.07 GB | 13.0 = 5.5 + 0.03 + 4.9 | 83 | 8.0 | 135 | direct |

(`_logs/agents/prefixcopy/prefix_rows_v1.log`.) The copy moves 1.07 GB in 13 ms at 1 M: 11 layers x 13 ms = 0.14 s per
16 K chunk, 0.2 % of the chunk. It was never the growing term (its share of a 16 K chunk at 131 K is 11 x 1.5 ms = 17 ms
against the ~10 s per 131 K of growth). `user_rows_rm(direct=True)` (default) takes the direct path for slot-0
contiguous pages from 524 288 rows on (the direct op has a ~2.7 ms floor below that); bit-identical, so the chunked
prefill tests are unchanged (`test_dsa_attention_r4097` bit-exact; `test_dsa_attention_long[16384]` in the same state as
the base: 2 rows at position 10311 below the 0.9999 bar, the known M-dependence in q_a / w_qb).

## 2. The growing term: the multi-row prefill scorer (`custom_ops/index_score_decode`, prefill form)

`test_prefill_chunked.py::test_dsa_prefill_chunk_cost` at 16 K / 131 K / 262 K / 524 K / 1 M positions (the prefix seeded
through `prefill_state`, the measured chunk the first 16 384 real rows of R16385 placed at the end), one DSA layer
(layer 3, attention only), 8 sub-blocks of 2048 query rows. Its stage report had been wiped by a `timer.reset()` on the
pass after the timed one (fixed).

The 8x4 (a multi-row mesh) scored prefill queries through the custom scorer's prefill form: the T = 2048 queries of a
sub-block are T "users" of one cache slot, and every user streams the slot's pooled keys (N/4 pools x 128 x 2 B = 64 MiB
at 1 M) once: 128 GiB of DRAM reads per sub-block at 1 M, O(T x N). On chip 0 with GLM53_PREFILL_SCORER=custom
(`chunk_cost_custom_scorer.log`):

| positions N | chunk wall s | tok/s | kv_copy ms | scorer ms | topk ms | sparse_sdpa ms | stages total ms | DRAM peak GiB |
|---|---|---|---|---|---|---|---|---|
| 16 384 | 0.430 | 38 120 | 0.5 | 96 | 19 | 94 | 462 | 2.71 |
| 131 072 | 1.163 | 14 088 | 1.6 | 1 240 | 40 | 106 | 1 622 | 2.83 |
| 262 144 | 2.164 | 7 571 | 3.5 | 2 696 | 57 | 109 | 3 113 | 2.98 |
| 524 288 | 3.906 | 4 194 | 4.4 | 5 367 | 98 | 105 | 5 808 | 3.48 |
| 1 048 576 | 8.400 | 1 950 | 7.5 | 10 859 | 180 | 107 | 11 388 | 4.48 |

The scorer grows by ~1.3 s per 131 K of prefix per layer per chunk (the custom kernel streams at ~100 GB/s over the
re-read keys); x 11 layers = ~12 s per 131 K, the slope of the 8x4 log. Projection for a 1 M prompt in 64 chunks, 11 DSA
layers (the per-chunk cost interpolated over the measured points): 2 956 s = 49 min.

The shared scorer (`ttnn.experimental.indexer_score_dsa`, a matmul over the pooled keys: the keys stream once per query
tile block) on one chip (`chunk_cost_before.log`, chunk wall only): 0.391 / 0.582 / 0.610 / 0.667 / 0.789 s at the same
points -> 465 s = 7.8 min for 11 layers. The dense attention, kv copy, top-k, sparse_sdpa and projections do not grow
beyond the top-k's 19 -> 180 ms.

## 3. The change: the shared scorer on multi-row meshes, exact through a per-row bounded top-k

`indexer_score_dsa` treats the mesh rows as sequence-parallel ranks (`seq_shard_axes=[0]`: row r scores with the chunk
start + r*Sq, Sq = the query rows of the call), which is why multi-row meshes had been sent to the custom scorer: the v1
prefill replicates the prompt over the rows (`DecoderLayer.prefill_forward`: every row attends over the all-gathered
chunk, the rows' identical outputs reduce-scattered). With the shared op, row r >= 1 scores a SUPERSET of the pools its
queries may see (pool p visible iff 4p+3 <= start + r*Sq + t instead of start + t; row 0 exact); the extra columns hold
the pooled keys of later rows of the same chunk (written by `maintain_prefill` before the selection) or, past the chunk
end, stale / zero pools. `TtDSAIndexer.select` now ranks every score row over its exact causal prefix only
(`custom_ops/topk_rows`, per-row bounds floor((start + t + 1)/4) clamped to [512, kv_len]) on multi-row meshes, so every
row selects the ids a single chip selects, in the same order (the same top-k kernels). The scored width `kv_len` grows by
the largest offset so that every row's chunk start lies inside it (the op's contract); a sub-block whose superset would
start past the pooled-key allocation (only the last chunks of a prompt that fills its whole allocation) keeps the custom
scorer (`prefill_rows_form`). `GLM53_PREFILL_SCORER`: auto / shared (the default everywhere now) or custom.

Single-chip emulation: `TtDSAIndexer.row_offset_sim = r` passes the chunk start + r*Sq explicitly, i.e. exactly what row
r of a multi-row mesh executes; `tests/test_select_rows.py` compares the exact path with emulated rows 1 and 7 (and the
custom scorer) on a 16 K chunk at 131 K positions: id sets, id order and the chunk's attention output.

`_logs/agents/prefixcopy/select_rows_v1.log` (chip 0, layer 3, a 16 K chunk at [114688, 131072), 8 sub-blocks of
2048 rows, the allocation with the rows' offset headroom; "topk" = untilize + top-k + id expansion + concat):

| variant | chunk wall s | scorer ms | topk ms | sparse_sdpa ms | ids vs exact (16384 rows) | output vs exact |
|---|---|---|---|---|---|---|
| exact (single chip, shared scorer + `topk_large_indices`) | 3.75 (see below) | 31.5 | 47.6 | 107.5 | - | - |
| emulated row 1 (chunk start + 2048, bounded top-k) | 2.09 | 31.4 | 72.5 | 105.5 | same sets, same order | bitwise equal |
| emulated row 7 (chunk start + 14336, bounded top-k) | 1.98 | 31.3 | 67.4 | 103.6 | same sets, same order | bitwise equal |
| custom scorer (the previous multi-row path) | 3.14 | 1218.9 | 38.2 | 103.9 | 16384 rows differ | row pcc min 0.9698 |

The bounded top-k costs ~3 ms more per sub-block than the shared one (`topk_rows` at 2048 rows: 3.0 ms at 32 K
columns, 20.8 ms at 262 K, against 2.6 / 20.4 ms for `topk_large_indices`, `topk_bench.log`). The chunk walls of this
test (2-3.7 s) carry the test's own index recorder (a 17 MB readback and a host sort per sub-block, charged to the
absorb_q mark); the cost test below has the real walls. Side finding: the custom scorer's selections differ from the
shared op's on EVERY sparse row of the chunk (min row pcc 0.9698 of the attention output): the 8x4 run has been
selecting differently from the single-chip reference.

The cost test with GLM53_LONGCTX_ROW_SIM=7 (the slowest row of the 8x4) is the "after" of the multi-row path:

`chunk_cost_after_row7.log` (GLM53_LONGCTX_ROW_SIM=7, the allocation with the offset headroom):

| positions N | chunk wall s | tok/s | kv_copy ms | scorer ms | topk ms | sparse_sdpa ms | stages total ms | DRAM peak GiB |
|---|---|---|---|---|---|---|---|---|
| 16 384 | 0.428 | 38 301 | 3.5 | 66 | 40 | 94 | 471 | 3.54 |
| 131 072 | 0.593 | 27 646 | 1.8 | 207 | 58 | 106 | 611 | 3.77 |
| 262 144 | 0.621 | 26 369 | 3.0 | 207 | 85 | 105 | 640 | 4.02 |
| 524 288 | 0.678 | 24 181 | 4.2 | 209 | 145 | 105 | 707 | 4.52 |
| 1 048 576 | 0.797 | 20 560 | 7.1 | 219 | 258 | 106 | 840 | 5.52 |

The plain shared curve (row 0 / one chip, `chunk_cost_shared_stages.log`) is the same within noise: 0.391 / 0.582 /
0.611 / 0.667 / 0.785 s, scorer 35 / 204 / 205 / 206 / 215 ms, topk 19 / 45 / 73 / 125 / 234 ms. Per layer and 16 K chunk
the custom path grew from 1.16 s at 131 K to 8.40 s at 1 M; the new path from 0.59 s to 0.80 s. Projection for a 1 M
prompt, 11 DSA layers: 473 s (7.9 min) against 2 956 s (49 min): ~41 min of the 8x4 run's 98 min, if the board behaves
like the emulation. The terms left that grow with N are the top-k group (the untilize of the [2048, kv_len] scores +
the top-k: 40 -> 258 ms per chunk) and the shared scorer (66 -> 219 ms, flat from 131 K on); the DRAM peak at the stage
boundaries is 1 GiB higher than the custom path's (5.52 vs 4.48 GiB at 1 M: the TILE scores and their ROW_MAJOR copy).

## 4. What the 8x4 still needs

* A (2,4)-block check of the multi-row path before the next 1 M run (`test_mesh_sparse_partitioned`,
  `test_mesh_long_context` at 16 K): chip 0 emulates the offset, not the op's per-device offset itself.
* Memory: the shared path's transient is the TILE scores + their ROW_MAJOR copy (2 x [2048, kv_len] bf16 = 2 GiB at 1 M)
  against the custom scorer's 1 GiB; the 1 M single-slot budget (12.4 GB weights + 11.8 GB latent + 0.75 GB pools) has
  the room, to be confirmed on the board.
* The structural next step (8x less scoring per chip): sequence-parallel sparse rows, every row selecting and attending
  only its own 2048-row slab with the shared op's native offset (no superset, no reduce-scatter of identical copies for
  the sparse rows); it needs per-row tail ids (a row-sharded upload) and the dense/sparse slab bookkeeping of chunk 0.
