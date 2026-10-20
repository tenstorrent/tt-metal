# index_score_decode (model-local generic op)

The DSA indexer's pool scores for one decode query per user, all users in one program:
`score[b, p] = sum_h relu(q[b,h] . k[b,p]) * w[b,h]` over the 32 index heads, one bf16 row per user. Replaces the
per-user `indexer_score_dsa` calls of the traced sparse decode (a 32-row query block with one real row per user,
0.38 ms per user at 1 M, plus a 16 MB 32-row output and a row slice each): the user's 32 heads are the 32 ROWS of
one query tile row, so each key tile is multiplied once, and the pooled keys stream from DRAM once. No causal
mask: the top-k's per-row bound (`custom_ops/topk_rows`) excludes the pools at and beyond the user's incomplete
pool. Design: `doc/mesh/sparse_1m.md` section 2.4.

| tensor | shape | dtype / layout | role |
|---|---|---|---|
| `q` | `[1, B, 32, D]` (D = 128) | bf16 TILE | user b's index-head queries as the rows of tile row b (the indexer softmax scale folded) |
| `w` | `[1, 1, B, 32]` | bf16 TILE | the per-head gate weights (n_heads^-0.5 folded); row b of the tile page b // 32 |
| `pools` | `[slots, 1, W, D]`, slots >= B, W % (32*KC) == 0 | bf16 or bfp8_b TILE | the pooled-key cache view (`TtIndexCache.pools_batched()`, dtype `GLM53_POOL_DTYPE`); user b = slot b |
| `out` | `[B, 1, 1, W]` | bf16 ROW_MAJOR, pre-allocated | the scores; column p = pool p |

Kernels: reader (per item = (user, k-band): the Dt query tiles, a w-row tile (zeroed; row 0 of each face column =
row b of the w tile, copied by the RISC out of the row's two faces read into the scratch CB 6, since DRAM reads are
64-byte aligned and a tile row is a 32-byte piece), then the band's k-tiles in chunks of KC laid d-major in the CB);
compute (per chunk: Dt `matmul_block(rt 1, ct KC, kt Dt, K transposed)` calls accumulated in DEST, one per inner
d-tile, with ReLU in the packer -> KC tiles of [32 heads x 32 pools], then KC single-tile matmuls `Wrow @ S_j`
whose row 0 is the head-weighted score of 32 pools; bf16 DEST half-sync, HiFi2, `SrcOrder::Reverse` startup);
writer (row 0 of each out tile, two 32-byte face pieces, assembled into a 64*KC-byte line and written at the
chunk's column of the user's row). Compile-time args: reader `[Dt, KC, Tt, 32, acc(q), acc(pools), acc(w)]`,
compute `[Dt, KC]`, writer `[KC, acc(out)]`; runtime per core `[item0, n_items, stride, ...addresses, n_bands,
band_chunks, total_chunks]`. Items (user, band) are dealt round-robin over `min(B*n_bands, grid)` cores with
`n_bands ~ grid / B` bands of whole KC-chunks per user. Under trace the four tensors are persistent (`TtDSA` keeps
the score rows per batch size). Test: `tests/custom_ops/test_index_score_decode.py` (torch golden, W = 1280 /
262400, B = 1/4/8/32, trace replay, bandwidth).
