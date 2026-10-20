# topk_rows (model-local generic op)

Top-512 indices of every row of a `[B, 1, 1, W]` bf16 ROW_MAJOR score tensor, each row ranked over its own prefix
(`valid_rows[b]`, a `[B, 1]` uint32 tensor read on device: one 4-byte page per row), descending by score. The
sparse DSA decode (`tt/dsa.py`, `TtDSA._decode_unified`, `GLM53_TOPK_BATCHED=1`) concatenates the users' pool-score
rows and ranks them in ONE call under their own causal bounds (`SparseDecodeMeta.valid_len_rows`): a 262 K-wide
row costs 1.14 ms on its core and B rows cost the same together, against B x 1.14 ms of per-user calls of the
shared op (which takes a single bound). Design: `doc/mesh/sparse_1m.md` section 2.4.

| tensor | shape | dtype / layout | role |
|---|---|---|---|
| `scores` | `[B, 1, 1, W]`, W >= 512 | bf16 ROW_MAJOR DRAM | one page per row (read) |
| `valid_rows` | `[B, 1]` | uint32 ROW_MAJOR DRAM | per-row search width in elements, 512 <= v <= W (read; refreshed in place between trace replays) |
| `out` | `[B, 1, 1, 512]` | uint32 ROW_MAJOR DRAM, pre-allocated | descending top-512 indices per row; a -inf survivor is 0xFFFFFFFF |

Kernels (`kernels/`): the shared `topk_large_indices` reader / compute / writer (and their three headers) copied
next to the model with the per-row metadata change: the reader reads page `row` of `valid_rows`, validates and
clamps it exactly like the shared op, derives that row's chunk bounds and pushes them to the compute mailbox
before streaming the row's 512-element chunks; compute pops the bounds per row. K = 512 only (the LLK window;
indices CB with 16-row x 2-face geometry, `FusedEndToEnd` body up to 32 chunks per row, `Classic` beyond).
Compile-time / common / per-core runtime args follow the shared factory (`op.py`): reader
`[cb_in, chunk_bytes, tile_bytes, tiles_per_sequence, acc(scores), 1, cb_meta, 0, acc(valid_rows), acc(scores),
per_row=1]`, common `[scores_addr, valid_addr, W, row_bytes, 0]`, per core `[start_row, rows]`; compute
`[cb_in, cb_indices, 512, body_mode, 1, cb_meta, 1]`, common `[W]`, per core `[rows]`; writer
`[cb_indices, cb_scratch, row_bytes, 32, 32, 64, acc(out)]`, common `[out_addr]`, per core `[start_row, rows]`.
Rows are dealt over the whole compute grid row-wise (one row per core while B <= cores). Under trace all three
tensors are persistent (`TtDSA` keeps the output per batch size; the concatenated score rows are allocated inside
the trace). Test: `tests/custom_ops/test_topk_rows.py` (torch golden per row, W = 1056 and 262 176, B = 1/4/8/32,
trace replay with refreshed bounds, timing).
